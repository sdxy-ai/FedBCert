"""RG-1.0 local-input-only FedAvg training; no test/full-graph interface."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from openfgl.flcore.fedbcert.certificate import PolynomialSpectralClassifier, row_normalized_shift
from openfgl.flcore.fedbcert.federated import cpu_state, weighted_average_states

DEFAULTS = dict(order_count=3, dropout=.2, residual_scale=.5, alpha=.25,
                rounds=100, local_epochs=1, lr=.01, weight_decay=.0005)
CLIENT_KEYS = {"features", "local_edge_index", "train_targets", "validation_targets"}


def _spec(spec):
    required = {"name", "input_dim", "num_classes", "hidden_dim"}
    if not required <= set(spec) or set(spec) - required - set(DEFAULTS):
        raise ValueError("invalid RG1 specification keys")
    result = dict(DEFAULTS, **spec)
    if result["name"] not in {"P", "I", "MLP"}: raise ValueError("name must be P, I, or MLP")
    for key in ("input_dim", "num_classes", "hidden_dim", "rounds"):
        if type(result[key]) is not int or result[key] < 1: raise ValueError(f"{key} must be a positive integer")
    if result["num_classes"] < 2: raise ValueError("at least two classes required")
    for key, value in DEFAULTS.items():
        if key != "rounds" and result[key] != value: raise ValueError(f"RG1 fixes {key}={value}")
    return result


def make_model(spec, device="cpu", dtype=torch.float32):
    p = _spec(spec)
    return PolynomialSpectralClassifier(
        input_dim=p["input_dim"], hidden_dim=p["hidden_dim"], num_classes=p["num_classes"],
        order_count=p["order_count"], dropout=p["dropout"], encoder_depth=2,
        encoder_type="feature_only_mlp" if p["name"] == "MLP" else "dual_feature_spectral",
        encoder_residual_scale=p["residual_scale"], spectral_fusion_scale=0. if p["name"] == "MLP" else p["alpha"],
        certificate_margin_mode="pairwise_difference").to(device=device,dtype=dtype)


def _clone(state): return {key:value.detach().cpu().clone() for key,value in state.items()}


def _tensor_hash(state):
    digest=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic(path,value):
    path=Path(path); temporary=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            if path.suffix==".pt": torch.save(value,stream)
            else: stream.write((json.dumps(value,indent=2,allow_nan=False)+"\n").encode())
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def _prepare_clients(p,clients,device):
    if len(clients)!=3: raise ValueError("RG1 requires exactly three clients")
    prepared=[]
    for client in clients:
        if set(client)!=CLIENT_KEYS: raise ValueError("clients accept only local features/edges and train/validation targets")
        if not all(isinstance(value,torch.Tensor) for value in client.values()): raise ValueError("client values must be tensors")
        features=client["features"]
        if features.ndim!=2 or features.size(1)!=p["input_dim"] or not features.is_floating_point(): raise ValueError("invalid local feature shape/type")
        features=features.detach().to(device=device,dtype=torch.float32).clone(); n=features.size(0)
        if n==0 or not bool(torch.isfinite(features).all()): raise ValueError("local features must be nonempty and finite")
        edge=client["local_edge_index"]
        if edge.dtype!=torch.long or edge.ndim!=2 or edge.size(0)!=2: raise ValueError("local edges must be a 2-by-E int64 tensor")
        if edge.numel() and (bool((edge<0).any()) or bool((edge>=n).any())): raise ValueError("edge endpoint is not local")
        targets={}
        for key in ("train_targets","validation_targets"):
            value=client[key]
            if value.dtype!=torch.long or value.shape!=(n,): raise ValueError("targets must be node-aligned int64 vectors")
            if bool(((value< -1)|(value>=p["num_classes"])).any()): raise ValueError("targets must be -1 or a valid class")
            targets[key]=value.detach().to(device=device).clone()
        if bool(((targets["train_targets"]>=0)&(targets["validation_targets"]>=0)).any()): raise ValueError("training and validation targets overlap")
        if not bool((targets["train_targets"]>=0).any()): raise ValueError("each RG1 client needs a labelled training node")
        if p["name"]=="P": shift,_=row_normalized_shift(edge,n,device=device,dtype=torch.float32)
        else:
            diagonal=torch.arange(n,device=device)
            shift=torch.sparse_coo_tensor(torch.stack((diagonal,diagonal)),torch.ones(n,device=device,dtype=torch.float32),(n,n),dtype=torch.float32,device=device).coalesce()
        prepared.append(dict(features=features,shift=shift,**targets))
    if not any(bool((c["validation_targets"]>=0).any()) for c in prepared): raise ValueError("pooled validation set is empty")
    return prepared


@torch.no_grad()
def _validation_score(model,clients):
    model.eval(); targets=[]; predictions=[]
    for client in clients:
        mask=client["validation_targets"]>=0
        if bool(mask.any()):
            logits=model(client["features"],client["shift"])
            targets.append(client["validation_targets"][mask].cpu()); predictions.append(logits[mask].argmax(1).cpu())
    return float(f1_score(torch.cat(targets).numpy(),torch.cat(predictions).numpy(),average="macro",zero_division=0,
                         labels=list(range(model.classifier.out_features))))


def _event(checkpoint,phase,**fields):
    if checkpoint is not None: checkpoint(dict(phase=phase,**fields))


def train_one(spec,clients,initial_state,model_seed,output_dir,device="cpu",checkpoint=None):
    p,device=_spec(spec),torch.device(device)
    if type(model_seed) is not int or model_seed<0: raise ValueError("model_seed must be a nonnegative integer")
    prepared=_prepare_clients(p,clients,device); state=_clone(initial_state)
    if any(v.is_floating_point() and (v.dtype!=torch.float32 or not bool(torch.isfinite(v).all())) for v in state.values()): raise ValueError("initial floating state must be finite float32")
    evaluator=make_model(p,device); evaluator.load_state_dict(state,strict=True)
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=False)
    counts=[int((c["train_targets"]>=0).sum()) for c in prepared]; initial_hash=_tensor_hash(state)
    trajectory=[]; best=None; best_f1=-math.inf; best_round=0; aggregated_round=0; started=time.monotonic()
    try:
        _atomic(output/"initial_state.pt",state); _event(checkpoint,"start",model=p["name"],round=0)
        for round_id in range(1,p["rounds"]+1):
            _event(checkpoint,"round_start",model=p["name"],round=round_id); local_states=[]
            for client_id,client in enumerate(prepared):
                seed=int(np.random.SeedSequence([model_seed,round_id,client_id]).generate_state(1)[0])
                _event(checkpoint,"client_start",model=p["name"],round=round_id,client=client_id,rng_seed=seed)
                local=make_model(p,device); local.load_state_dict(state,strict=True); torch.manual_seed(seed); local.train()
                optimizer=torch.optim.Adam(local.parameters(),lr=p["lr"],weight_decay=p["weight_decay"]); optimizer.zero_grad(set_to_none=True)
                fused,_,feature,spectral=local.training_logits(client["features"],client["shift"])
                mask=client["train_targets"]>=0; labels=client["train_targets"][mask]
                loss=F.cross_entropy(fused[mask],labels) if p["name"]=="MLP" else (F.cross_entropy(feature[mask],labels)+F.cross_entropy(spectral[mask],labels))
                if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite training loss")
                loss.backward()
                if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in local.parameters()): raise RuntimeError("nonfinite gradient")
                optimizer.step()
                if any(not bool(torch.isfinite(v).all()) for v in local.parameters()): raise RuntimeError("nonfinite local state")
                local_states.append(cpu_state(local))
                _event(checkpoint,"client_end",model=p["name"],round=round_id,client=client_id,rng_seed=seed,loss=float(loss.detach().cpu()))
            candidate=weighted_average_states(local_states,counts)
            if not all(bool(torch.isfinite(value).all()) for value in candidate.values()): raise RuntimeError("nonfinite aggregated state")
            state,aggregated_round=candidate,round_id; evaluator.load_state_dict(state,strict=True); score=_validation_score(evaluator,prepared)
            if not math.isfinite(score): raise RuntimeError("nonfinite validation score")
            improves=score>best_f1
            if improves: best,best_f1,best_round=_clone(state),score,round_id
            row=dict(round=round_id,validation_macro_f1=score,is_new_best=improves,tensor_sha256=_tensor_hash(state))
            _atomic(output/f"round_{round_id:03d}.json",row); trajectory.append(row); _event(checkpoint,"round_end",model=p["name"],**row)
        assert best is not None
        _atomic(output/"selected_state.pt",best); _atomic(output/"last_state.pt",state); _atomic(output/"validation_trajectory.json",trajectory)
        record=dict(status="complete",spec=p,model_seed=model_seed,dtype="float32",client_training_counts=counts,
            rounds_completed=len(trajectory),selected_round=best_round,selected_validation_macro_f1=best_f1,
            initial_tensor_sha256=initial_hash,selected_tensor_sha256=_tensor_hash(best),last_tensor_sha256=_tensor_hash(state),
            applied_spectral_scale=0. if p["name"]=="MLP" else p["alpha"],checkpoint_rule="pooled validation Macro-F1; exact ties earliest",
            rng_rule="SeedSequence([model_seed, round, client_id])",training_seconds=time.monotonic()-started)
        record["artifact_sha256"]={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ("initial_state.pt","selected_state.pt","last_state.pt","validation_trajectory.json")}
        _event(checkpoint,"end",model=p["name"],round=p["rounds"]); _atomic(output/"training_record.json",record); return record
    except BaseException as error:
        for name,value in (("last_state_partial.pt",state),("failure.json",dict(status="failed",error_type=type(error).__name__,error=str(error),rounds_completed=len(trajectory),last_aggregated_round=aggregated_round,initial_tensor_sha256=initial_hash,partial_tensor_sha256=_tensor_hash(state),partial_state_scope="last complete FedAvg aggregation, not unfinished local update"))):
            try: _atomic(output/name,value)
            except Exception: pass
        raise
