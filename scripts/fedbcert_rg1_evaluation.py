"""Label-separated RG-1 evaluation; no training or checkpoint selection."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from openfgl.flcore.fedbcert.certificate import PolynomialSpectralClassifier, retained_mass_uncertainty_basis, row_normalized_shift
from openfgl.flcore.fedbcert.geometry_baselines import evaluate_geometry_baselines
from openfgl.flcore.fedbcert.model_contract import _FIELDS, _modules, _CLASS_NAMESPACES, _CLASS_BASES, _FUNCTIONS, model_fingerprint
from openfgl.flcore.fedbcert.on_demand import potential_exposure
from scripts import run_fedbcert_on_demand_20261015 as legacy

GUARD,ATOL=1e-6,1e-10
STRATEGIES=(("L",None),("F","full_reference"),("E","exposure_only"),("S","ball_pairwise"),("D","nonnegative_diameter"),("N","cone"))


def _require(condition,message):
    if not bool(condition): raise RuntimeError(message)


def _hash_tensor(tensor):
    value=tensor.detach().cpu().contiguous(); digest=hashlib.sha256(str((value.dtype,tuple(value.shape))).encode()); digest.update(value.numpy().tobytes()); return digest.hexdigest()


def _fingerprint(model):
    if model.encoder_type!="feature_only_mlp": return model_fingerprint(model)
    # Inference-only identity for the control; never signs a certificate.
    _require(type(model) is PolynomialSpectralClassifier,"nonstock MLP control")
    for cls,trusted in _CLASS_NAMESPACES:
        current=vars(cls)
        _require(set(current)==set(trusted) and all(current[k] is v for k,v in trusted.items()),"control class changed")
    for cls,bases,mro in _CLASS_BASES: _require(cls.__bases__==bases and cls.__mro__==mro,"control inheritance changed")
    for f,code,defaults,kw in _FUNCTIONS: _require(f.__code__ is code and f.__defaults__ is defaults and f.__kwdefaults__==kw,"control function changed")
    for key,value in vars(nn.modules.module).items():
        if key.startswith("_global_") and "hooks" in key: _require(type(value) in {dict,OrderedDict} and not value,"global hooks unsupported")
    modules={}
    for name,module in _modules(model):
        config=dict(type=type(module).__qualname__,training=module.training,children=list(module._modules))
        if type(module) is nn.Linear:
            _require(tuple(module.weight.shape)==(module.out_features,module.in_features),"Linear shape")
            _require(module.bias is None or tuple(module.bias.shape)==(module.out_features,),"Linear bias")
            config.update(in_features=module.in_features,out_features=module.out_features,bias=module.bias is not None)
        elif type(module) is nn.LayerNorm:
            _require(math.isfinite(module.eps) and module.eps>0,"LayerNorm eps")
            shape=tuple(module.normalized_shape)
            _require((module.weight is not None)==module.elementwise_affine,"LayerNorm affine")
            for p in (module.weight,module.bias): _require(p is None or tuple(p.shape)==shape,"LayerNorm shape")
            config.update(normalized_shape=shape,eps=module.eps,affine=module.elementwise_affine,bias=module.bias is not None)
        modules[name]=config
    state={}
    for name,value in model.state_dict().items():
        _require(value.device.type=="cpu" and value.dtype==torch.float64 and torch.isfinite(value).all(),"control requires finite CPU float64")
        state[name]=_hash_tensor(value)
    payload=dict(config={key:getattr(model,key) for key in _FIELDS},modules=modules,state=state)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _observed(allowed):
    edges=allowed["full_edge_index"]; owners=allowed["membership"]
    return dict(features=allowed["features"],membership=owners,
                local_edge_index=edges[:,owners[edges[0]]==owners[edges[1]]])


def _views(observed,targets,identity=False):
    # Local inference gets only the observed induced graph, not remote edges.
    x,owners=observed["features"],observed["membership"]
    for owner in sorted(set(owners[targets].tolist())):
        nodes=torch.where(owners==owner)[0]; features=x[nodes]
        if identity:
            diagonal=torch.arange(nodes.numel()); shift=torch.sparse_coo_tensor(torch.stack((diagonal,diagonal)),torch.ones(nodes.numel(),dtype=x.dtype),(nodes.numel(),nodes.numel())).coalesce(); degree=torch.ones(nodes.numel(),dtype=x.dtype)
        else:
            edges=observed["local_edge_index"]; keep=owners[edges[0]]==owner
            index=torch.full((x.size(0),),-1,dtype=torch.long); index[nodes]=torch.arange(nodes.numel())
            shift,degree=row_normalized_shift(index[edges[:,keep]],nodes.numel(),device=torch.device("cpu"),dtype=x.dtype)
        yield nodes,features,shift,degree


@torch.no_grad()
def _local(model,observed,targets,identity=False):
    _require("full_edge_index" not in observed,"local inference cannot accept remote topology")
    if identity: observed={k:observed[k] for k in ("features","membership")}
    before=_fingerprint(model); data_before={k:_hash_tensor(v) for k,v in observed.items()}
    output=torch.empty((observed["features"].size(0),model.classifier.out_features),dtype=torch.float64)
    for nodes,features,shift,_ in _views(observed,targets,identity): output[nodes]=model(features,shift)
    _require(_fingerprint(model)==before,"local model changed")
    _require(data_before=={k:_hash_tensor(v) for k,v in observed.items()},"local data changed")
    _require(torch.isfinite(output[targets]).all(),"nonfinite local logits"); return output[targets]


def _publish(path,payload,binary=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); descriptor,temporary=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream:
            stream.write(payload if binary else json.dumps(payload,indent=2,allow_nan=False).encode()); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary,path)
    finally: os.unlink(temporary)


def _save_tensor(path,payload):
    stream=io.BytesIO(); torch.save(payload,stream); _publish(path,stream.getvalue(),True)


@torch.no_grad()
def _interval_margins(model,hidden,shift,degree,counts,nominal):
    classes=nominal.size(1); prediction=nominal.argmax(1); spectral=model.classifier(model.filter(hidden,shift))
    alpha=model.spectral_fusion_scale; invariant=nominal-alpha*spectral; result=torch.empty_like(nominal); mass=(degree/(degree+counts))[:,None]
    for a in range(classes):
        direction=model.classifier.weight[a]-model.classifier.weight
        ls=-torch.linalg.vector_norm(direction.clamp(max=0),dim=1); us=torch.linalg.vector_norm(direction.clamp(min=0),dim=1)
        lower=upper=hidden@direction.T; total=torch.zeros_like(lower)
        for k in range(model.order_count):
            theta=model.theta[k]; total+=theta*(lower if theta>=0 else upper)
            if k+1<model.order_count:
                pl=torch.sparse.mm(shift,lower); pu=torch.sparse.mm(shift,upper)
                lower=torch.minimum(pl,mass*pl+(1-mass)*ls); upper=torch.maximum(pu,mass*pu+(1-mass)*us)
        bias=model.classifier.bias[a]-model.classifier.bias
        bound=invariant[:,a:a+1]-invariant+alpha*(total+bias); selected=prediction==a; result[selected]=bound[selected]
    return result


@torch.no_grad()
def _diagnostics(model,allowed):
    x,edges,owners=(allowed[k] for k in ("features","full_edge_index","membership")); nodes=torch.arange(x.size(0))
    full_shift,full_degree=row_normalized_shift(edges,x.size(0),device=torch.device("cpu"),dtype=x.dtype)
    full=model(x,full_shift); hidden=model.bounded_encoding(x)
    _require(torch.isfinite(hidden).all() and (hidden>=0).all(),"hidden-domain failure"); _require((hidden.norm(dim=1)<=1+ATOL).all(),"hidden norm failure")
    powers=[hidden]
    for _ in range(1,model.order_count): powers.append(torch.sparse.mm(full_shift,powers[-1]))
    local=torch.empty_like(full); masks={n:torch.zeros(x.size(0),dtype=torch.bool) for n in ("ball_pairwise","nonnegative_diameter","cone")}; slacks={n:torch.empty(x.size(0),dtype=x.dtype) for n in masks}
    exposed=torch.zeros(x.size(0),dtype=torch.bool); errors=torch.empty((x.size(0),model.order_count),dtype=x.dtype); bounds=torch.empty_like(errors); margins=torch.empty_like(full)
    for ids,features,shift,degree in _views(_observed(allowed),nodes):
        counts=full_degree[ids]-degree; _require((counts>=0).all() and (counts==counts.floor()).all(),"invalid external count")
        budget=2*counts/(degree+counts); basis=retained_mass_uncertainty_basis(shift,budget,model.order_count); matrix=torch.stack(basis,dim=1)
        logits=model(features,shift); encoded=model.bounded_encoding(features)
        geometry=evaluate_geometry_baselines(model,features,shift,budget,guard=GUARD,logits=logits,basis=basis)
        local[ids]=logits; exposed[ids]=potential_exposure(shift,counts>0,model.order_count); propagated=encoded
        for k in range(model.order_count):
            error=(powers[k][ids]-propagated).norm(dim=1); _require((error<=matrix[:,k]+ATOL).all(),"representation bound violation")
            errors[ids,k],bounds[ids,k]=error,matrix[:,k]; propagated=torch.sparse.mm(shift,propagated)
        lower=_interval_margins(model,encoded,shift,degree,counts,logits); pred=logits.argmax(1); actual=full[ids].gather(1,pred[:,None])-full[ids]
        competitor=torch.arange(full.size(1))[None,:]!=pred[:,None]
        _require((lower[competitor]<=actual[competitor]+ATOL).all(),"pairwise interval margin violation")
        worst=lower.masked_fill(~competitor,float("inf")).min(1).values
        _require(torch.allclose(worst,geometry["cone"].worst_slack,atol=ATOL,rtol=0),"interval diagnostic disagrees")
        margins[ids]=lower
        for name in masks: slacks[name][ids]=geometry[name].worst_slack; masks[name][ids]=geometry[name].worst_slack>GUARD
    for lo,hi in (("ball_pairwise","nonnegative_diameter"),("nonnegative_diameter","cone")):
        _require((slacks[lo]<=slacks[hi]+ATOL).all(),"slack hierarchy violation"); _require((~masks[lo]|masks[hi]).all(),"certificate hierarchy violation")
    for name,mask in masks.items(): _require((~mask|(local.argmax(1)==full.argmax(1))).all(),"certified class flip: "+name)
    for value in (local,full,margins,errors,bounds,*slacks.values()): _require(torch.isfinite(value).all(),"nonfinite diagnostic")
    return dict(nodes=nodes,owners=owners.clone(),local_logits=local,full_logits=full,local_softmax=local.softmax(1),full_softmax=full.softmax(1),masks=masks,slacks=slacks,exposed=exposed,representation_errors=errors,representation_bounds=bounds,interval_margin_lower_bounds=margins)


def _metrics(prediction,labels,certified,owners,selection,classes):
    if not bool(selection.any()): return None
    p,y,c,o=(v[selection].cpu() for v in (prediction,labels,certified,owners))
    f1=lambda a,b:float(f1_score(a.numpy(),b.numpy(),labels=list(range(classes)),average="macro",zero_division=0))
    return dict(n=y.numel(),macro_f1=f1(y,p),accuracy=float((p==y).double().mean()),certified_coverage=float(c.double().mean()),certified_accuracy=float((c&(p==y)).double().mean()),worst_client_macro_f1=min(f1(y[o==owner],p[o==owner]) for owner in o.unique()))


def evaluate_case(models,allowed,test_nodes,test_labels,outdir,case_index,batch_seed,timing=None,checkpoint=None):
    timing={"warmup":1,"repetitions":3} if timing is None else dict(timing)
    _require(set(models)=={"P","I","MLP"},"P/I/MLP required"); _require(set(allowed)=={"features","full_edge_index","membership"},"label-free input contract")
    _require(timing=={"warmup":1,"repetitions":3},"timing protocol changed")
    x=allowed["features"]; _require(x.device.type=="cpu" and x.dtype==torch.float64,"CPU float64 required")
    owners=allowed["membership"]; edges=allowed["full_edge_index"]
    _require(x.ndim==2 and torch.isfinite(x).all(),"finite node feature matrix required")
    _require(owners.device.type=="cpu" and owners.dtype==torch.long and owners.shape==(len(x),) and (owners>=0).all(),"invalid ownership")
    _require(edges.device.type=="cpu" and edges.dtype==torch.long and edges.ndim==2 and edges.size(0)==2 and ((edges>=0)&(edges<len(x))).all(),"invalid edge endpoints")
    targets=torch.as_tensor(test_nodes,dtype=torch.long,device="cpu"); labels=torch.as_tensor(test_labels,dtype=torch.long,device="cpu")
    _require(targets.ndim==labels.ndim==1 and targets.numel()==labels.numel()>0,"test labels alignment"); _require(targets.unique().numel()==targets.numel(),"duplicate test nodes")
    _require(((targets>=0)&(targets<x.size(0))).all(),"test node out of range"); classes=models["P"].classifier.out_features; _require(((labels>=0)&(labels<classes)).all(),"test label range")
    order=targets.argsort(); targets,labels=targets[order],labels[order]
    ping=(lambda stage:checkpoint(stage)) if checkpoint is not None else (lambda stage:None)
    before={n:_fingerprint(m) for n,m in models.items()}; data_before={k:_hash_tensor(v) for k,v in allowed.items()}
    ping("diagnostics:start"); diagnostics=_diagnostics(models["P"],allowed); ping("diagnostics:complete")
    observed=_observed(allowed); all_nodes=torch.arange(x.size(0)); controls={n:_local(models[n],observed,all_nodes,True) for n in ("I","MLP")}
    sampled=np.random.default_rng(batch_seed).choice(targets.numpy(),min(32,targets.numel()),replace=False)
    workloads={"all_test":targets,"small_batch":torch.tensor(sorted(sampled.tolist()))}; records=[]; outputs={}; metrics={}
    for wi,(workload,selected) in enumerate(workloads.items()):
        metrics[workload]={}
        for repetition in range(-1,3):
            warmup=repetition<0; rotation=(case_index+wi+repetition)%len(STRATEGIES); sequence=STRATEGIES[rotation:]+STRATEGIES[:rotation]
            for method,strategy in (*sequence,("I",None),("MLP",None)):
                ping(f"{workload}:{method}:{repetition}:start"); started,process=time.perf_counter(),time.process_time()
                if method in {"L","I","MLP"}:
                    logits=_local(models["P" if method=="L" else method],observed,selected,method!="L")
                    evidence=dict(targets=selected.clone(),final_logits=logits,nominal_logits=logits.clone(),certified=torch.full((selected.numel(),),method!="L",dtype=torch.bool),fallback_nodes=torch.empty(0,dtype=torch.long),fallback_logits=torch.empty((0,classes),dtype=torch.float64))
                    row={k:0 for k in ("certified_n","fallback_n","metadata_wire_bytes","aggregate_request_bytes","aggregate_response_bytes","aggregate_wire_bytes","total_wire_bytes","remote_vectors","request_response_packets")}; row["target_n"]=selected.numel(); row["certified_n"]=selected.numel() if method!="L" else 0
                else: row,evidence=legacy.run_strategy(models["P"],allowed,selected,strategy,GUARD)
                row["cpu_seconds"]=time.perf_counter()-started; row["process_seconds"]=time.process_time()-process
                ping(f"{workload}:{method}:{repetition}:complete")
                if method not in {"L","I","MLP"}: legacy.check_result(row,evidence,diagnostics["full_logits"],diagnostics["local_logits"],selected,{"numerics":{"reference_logit_atol":ATOL}})
                elif method=="L": row["class_mismatches"]=int((evidence["final_logits"].argmax(1)!=diagnostics["full_logits"][selected].argmax(1)).sum())
                else:
                    _require(torch.allclose(evidence["final_logits"],controls[method][selected],atol=ATOL,rtol=0),"control inference changed"); _require(row["total_wire_bytes"]==0,"control payload")
                row.update(workload=workload,method=method,repetition=repetition,warmup=warmup,direct_local_n=(selected.numel() if method in {"L","I","MLP"} else row["certified_n"]))
                records.append(row)
                # Retain completed evidence immediately; later failure cannot erase it.
                _publish(Path(outdir)/"timing_records"/f"{workload}_{method}_{repetition+1}.json",row)
                if not warmup and repetition==0:
                    outputs[f"{workload}/{method}"]=evidence; positions=torch.searchsorted(targets,selected)
                    metrics[workload][method]={g:_metrics(evidence["final_logits"].argmax(1),labels[positions],evidence["certified"],allowed["membership"][selected],mask,classes) for g,mask in (("all",torch.ones(selected.numel(),dtype=torch.bool)),("exposed",diagnostics["exposed"][selected]))}
    # Repeated bytes/masks/counts must be deterministic; no failed repeats dropped.
    for workload in workloads:
        for method in ("L","F","E","S","D","N","I","MLP"):
            rows=[r for r in records if r["workload"]==workload and r["method"]==method and not r["warmup"]]
            for key in ("target_n","certified_n","fallback_n","total_wire_bytes","metadata_wire_bytes","aggregate_wire_bytes","remote_vectors","request_response_packets"):
                _require(len({r[key] for r in rows})==1,"repeated accounting changed")
    _require(before=={n:_fingerprint(m) for n,m in models.items()},"evaluation model changed")
    _require(data_before=={k:_hash_tensor(v) for k,v in allowed.items()},"evaluation input changed")
    test_mask=torch.zeros(len(x),dtype=torch.bool); test_mask[targets]=True
    changed=diagnostics["local_logits"].argmax(1)!=diagnostics["full_logits"].argmax(1)
    probability_changed=(diagnostics["local_softmax"]-diagnostics["full_softmax"]).abs().max(1).values>=.05
    structural={}
    for group,selection in (("all_graph",torch.ones(len(x),dtype=torch.bool)),("exposed_graph",diagnostics["exposed"]),("all_test",test_mask),("exposed_test",test_mask&diagnostics["exposed"])):
        count=int(selection.sum())
        structural[group]=dict(n=count,class_change_rate=float(changed[selection].double().mean()) if count else None,
            probability_change_at_least_005=float(probability_changed[selection].double().mean()) if count else None,
            coverage={label:float(diagnostics["masks"][name][selection].double().mean()) if count else None for label,name in (("S","ball_pairwise"),("D","nonnegative_diameter"),("N","cone"))})
    summary=dict(schema_version=1,case_index=case_index,batch_seed=batch_seed,guard=GUARD,diagnostic_atol=ATOL,metrics=metrics,structural_diagnostics=structural,diagnostics_passed=True,timing_records=records)
    _save_tensor(Path(outdir)/"diagnostics.pt",diagnostics); _save_tensor(Path(outdir)/"node_results.pt",dict(test_nodes=targets,test_labels=labels,controls=controls,routed_outputs=outputs))
    _publish(Path(outdir)/"timings.json",records); _publish(Path(outdir)/"summary.json",summary); return summary
