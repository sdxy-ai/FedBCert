"""Exclusive RG-1 artifacts, input preparation, seal gates and resource checks."""
from __future__ import annotations
from pathlib import Path
import ast
import hashlib
import json
import os
import signal
import sys
import time
import traceback
import uuid
import typing
import numpy as np
import psutil
import torch
from torch_geometric.utils import to_undirected
from scripts.fedbcert_rg1_training import _atomic, _tensor_hash, make_model, DEFAULTS
from openfgl.flcore.fedbcert.certificate import canonical_undirected

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
OLD_PREFLIGHT=PROJECT/"results_boundary_certified/fedbcert_integration_preflight_20261020/preflight_complete"
PREPARATION=PROJECT/"results_boundary_certified/fedbcert_rg1_runner_preparation_20261020"
PROTOCOL=PROJECT/"research/fedbcert/pre_submission_20261020/02_FedBCert_Real_Graph_Frozen_Protocol.md"


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""): digest.update(b)
    return digest.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def load(path): return torch.load(path,map_location="cpu",weights_only=True)
def publish(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); _atomic(path,value)
def seed(base,ds,stream): return int(np.random.SeedSequence([base,ds,stream]).generate_state(1)[0])
def code_index():
    paths=list((ROOT/"scripts").glob("*rg1*.py"))+list((ROOT/"openfgl/flcore/fedbcert").glob("*.py"))
    paths += [ROOT/"scripts/run_fedbcert_on_demand_20261015.py",ROOT/"scripts/fedbcert_e28b_common.py",PROTOCOL]
    return {str(p.resolve()):sha(p) for p in sorted(paths)}
def verify(index):
    for path,digest in index.items():
        if not Path(path).is_file() or sha(path)!=digest: raise RuntimeError("frozen file mismatch: "+path)
def environment_index():
    paths=[PREPARATION/n for n in ("requirements.lock.txt","requirements.wheels-hashed.txt","environment_validation.json","environment_build.json")]
    return {str(p):sha(p) for p in paths}


def existing_functions(path,names):
    tree=ast.parse(path.read_text()); funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in funcs}!=set(names): raise RuntimeError("source function missing")
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    selected=ast.fix_missing_locations(ast.Module(body=[future,*funcs],type_ignores=[]))
    namespace=dict(np=np,torch=torch,to_undirected=to_undirected,**vars(typing)); exec(compile(selected,str(path),"exec"),namespace); return namespace


def specifications(d,c,h=64,rounds=100):
    p=2*d*h+h*h+(5+2*c)*h+2*c+3
    width=min(range(1,4097),key=lambda w:(abs(w*w+(d+c+4)*w+c-p),-w))
    result={name:dict(DEFAULTS,name=name,input_dim=d,num_classes=c,hidden_dim=(width if name=="MLP" else h),rounds=rounds) for name in ("P","I","MLP")}
    for name,spec in result.items():
        expected=width*width+(d+c+4)*width+c if name=="MLP" else p
        if sum(v.numel() for v in make_model(spec).parameters())!=expected: raise RuntimeError("parameter count mismatch")
    return result


def _client_payload(x,edges,owners,y,masks):
    clients=[]
    for owner in range(3):
        nodes=torch.where(owners==owner)[0]; remap=torch.full((len(x),),-1,dtype=torch.long); remap[nodes]=torch.arange(len(nodes))
        keep=(owners[edges[0]]==owner)&(owners[edges[1]]==owner)
        client=dict(features=x[nodes].float().clone(),local_edge_index=remap[edges[:,keep]])
        for key,mask in (("train_targets",masks[0]),("validation_targets",masks[2])):
            values=torch.full((len(nodes),),-1,dtype=torch.long); selected=mask[nodes]; values[selected]=y[nodes][selected]; client[key]=values
        if not (client["train_targets"]>=0).any(): raise RuntimeError("empty local training set")
        clients.append(client)
    return clients


def prepare(out,*,toy=False):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    source_paths=[PROJECT/"openfgl/flcore/fedbeso/operator.py",PROJECT/"openfgl/flcore/fedsla_xn/data.py"]
    # Reuse the exact source functions pinned by the prior preflight.
    old=read(OLD_PREFLIGHT/"source_and_resources.json")["reused_source_sha256"]
    verify({str(PROJECT/k):v for k,v in old.items()})
    parts=existing_functions(source_paths[0],("metis_membership","balanced_random_membership"))
    split=existing_functions(source_paths[1],("_allocate_counts","stratified_four_way_masks"))["stratified_four_way_masks"]
    matrix=[]; inputs={}; cases=[]
    raw_manifest=read(OLD_PREFLIGHT/"upstream_raw_manifest.json")
    try:
        for ds,name in ((1,"Cora"),(2,"CiteSeer")):
            if toy:
                generator=torch.Generator().manual_seed(seed(42020000,ds,11))
                x=torch.randn(120,4,generator=generator); y=torch.arange(120)%3
                edges=canonical_undirected(torch.stack((torch.arange(120),torch.roll(torch.arange(120),1))),120)
                d,c,h,rounds=4,3,6,2; bases=[42020000]
            else:
                # Hash-check every raw and processed file before invoking loader.
                frozen=read(OLD_PREFLIGHT/(name+"_dataset.json"))
                verify({str(OLD_PREFLIGHT/k):v for k,v in frozen["raw_and_processed_sha256"].items()})
                from torch_geometric.datasets import Planetoid
                from torch_geometric.transforms import NormalizeFeatures
                data=Planetoid(root=str(OLD_PREFLIGHT/"datasets"),name=name,transform=NormalizeFeatures())[0]
                x,y=data.x.clone(),data.y.clone(); edges=canonical_undirected(data.edge_index,data.num_nodes)
                def tensor_sha(v): return hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                if tensor_sha(x)!=frozen["feature_sha256"] or tensor_sha(edges)!=frozen["canonical_edge_sha256"] or tensor_sha(y)!=frozen["labels_sha256"]: raise RuntimeError("standard tensor hash mismatch")
                d,c,h,rounds=x.size(1),int(y.max())+1,64,100; bases=list(range(42020101,42020109))
                if (len(x),d,c)!=((2708,1433,7) if name=="Cora" else (3327,3703,6)): raise RuntimeError("standard dataset dimensions changed")
            specs=specifications(d,c,h,rounds)
            for base in bases:
                masks=split(y,train_ratio=.3,calibration_ratio=.1,val_ratio=.2,test_ratio=.4,seed=seed(base,ds,11))
                if not torch.equal(torch.stack(masks).sum(0),torch.ones(len(x),dtype=torch.long)): raise RuntimeError("split overlap or gap")
                states={}
                for model_name,spec in specs.items():
                    torch.manual_seed(seed(base,ds,14)); states[model_name]={k:v.clone() for k,v in make_model(spec).state_dict().items()}
                if _tensor_hash(states["P"])!=_tensor_hash(states["I"]): raise RuntimeError("unpaired initialization")
                for partition,stream in (("metis",12),("random",13)):
                    f=parts["metis_membership" if partition=="metis" else "balanced_random_membership"]
                    args=(edges,len(x),3,seed(base,ds,stream)) if partition=="metis" else (len(x),3,seed(base,ds,stream))
                    owners=torch.as_tensor(f(*args),dtype=torch.long)
                    case_id=f"{name}_{partition}_{base}"; directory=out/"inputs"/case_id; directory.mkdir(parents=True)
                    clients=_client_payload(x,edges,owners,y,masks); publish(directory/"training_inputs.pt",dict(clients=clients))
                    # Separate vault is never deserialized by training.
                    nodes=torch.where(masks[3])[0]
                    publish(directory/"evaluation_vault.pt",dict(allowed=dict(features=x.double(),full_edge_index=edges,membership=owners),test_nodes=nodes,test_labels=y[nodes]))
                    publish(directory/"split_owners.pt",dict(masks=masks,membership=owners))
                    for model_name,spec in specs.items():
                        path=directory/f"initial_{model_name}.pt"; publish(path,states[model_name])
                        matrix.append(dict(case_id=case_id,dataset=name,partition=partition,seed=base,name=model_name,spec=spec,
                            training_inputs=str(directory/"training_inputs.pt"),initial_state=str(path),model_seed=seed(base,ds,14)))
                    cases.append(dict(case_id=case_id,dataset=name,partition=partition,seed=base,batch_seed=seed(base,ds,15),vault=str(directory/"evaluation_vault.pt")))
                    for p in directory.iterdir(): inputs[str(p)]=sha(p)
        manifest=dict(protocol="RG-1.0",mode="engineering_toy" if toy else "formal_pending_authorization",run_id=uuid.uuid4().hex,
            datasets=["Cora","CiteSeer"],partitions=["metis","random"],base_seeds=[42020000] if toy else list(range(42020101,42020109)),
            device="cpu",threads=1,source_sha256=code_index(),environment_sha256=environment_index(),input_sha256=inputs,
            reused_preparation_source_sha256={str(p):sha(p) for p in source_paths},matrix=matrix,cases=cases,expected_models=len(matrix),
            timing=dict(warmup=1,repetitions=3),guard=1e-6,diagnostic_atol=1e-10,
            budgets=dict(model_seconds=900,training_seconds=43200,evaluation_seconds=7200,record_seconds=120,rss_bytes=8*1024**3,storage_bytes=10*1024**3,cuda_allocated_bytes=6*1024**3),
            upstream_commit=raw_manifest["commit"],formal_training_authorized=False)
        if not toy and len(matrix)!=96: raise RuntimeError("incomplete planned matrix")
        publish(out/"manifest.json",manifest); return out/"manifest.json"
    except BaseException as e:
        publish(out/"preparation_failure.json",dict(type=type(e).__name__,reason=str(e),traceback=traceback.format_exc())); raise


def validate_manifest(path):
    manifest=read(path); verify(manifest["source_sha256"]); verify(manifest["environment_sha256"]); verify(manifest["input_sha256"]); verify(manifest["reused_preparation_source_sha256"])
    if manifest["protocol"]!="RG-1.0" or manifest["device"]!="cpu" or manifest["threads"]!=1: raise RuntimeError("unsupported frozen protocol/device")
    if len(manifest["matrix"])!=manifest["expected_models"]: raise RuntimeError("matrix incomplete")
    if manifest["mode"]=="formal_pending_authorization":
        expected={(d,p,s,m) for d in ("Cora","CiteSeer") for p in ("metis","random") for s in range(42020101,42020109) for m in ("P","I","MLP")}
        actual={(r["dataset"],r["partition"],r["seed"],r["name"]) for r in manifest["matrix"]}
        if len(manifest["matrix"])!=96 or actual!=expected: raise RuntimeError("formal matrix changed")
        for r in manifest["matrix"]:
            spec=r["spec"]
            if spec!=specifications(spec["input_dim"],spec["num_classes"])[r["name"]]: raise RuntimeError("formal architecture or budget changed")
    return manifest


def authorize(path,manifest_path,scope,execution=None):
    manifest=validate_manifest(manifest_path)
    if manifest["mode"]=="engineering_toy": return
    if path is None: raise PermissionError("separate explicit formal authorization required")
    authorization=read(path)
    if authorization.get("approved") is not True or authorization.get("manifest_sha256")!=sha(manifest_path) or scope not in authorization.get("scopes",[]): raise PermissionError("authorization does not bind this manifest and phase")
    if execution is not None and authorization.get("output_directory")!=str(Path(execution).resolve()): raise PermissionError("authorization does not bind this unique output directory")


def seal(manifest_path,execution):
    manifest=validate_manifest(manifest_path); execution=Path(execution); hashes={}; hashes_by_case={}
    for row in manifest["matrix"]:
        directory=execution/"training"/row["case_id"]/row["name"]; record=read(directory/"training_record.json")
        if record["status"]!="complete" or record["rounds_completed"]!=row["spec"]["rounds"] or record["spec"]!=row["spec"]: raise RuntimeError("incomplete or mismatched training record")
        initial=load(row["initial_state"])
        if record["initial_tensor_sha256"]!=_tensor_hash(initial): raise RuntimeError("initial state changed")
        for name,digest in record["artifact_sha256"].items():
            path=directory/name
            if sha(path)!=digest: raise RuntimeError("training artifact changed")
        trajectory=read(directory/"validation_trajectory.json")
        if len(trajectory)!=row["spec"]["rounds"] or record["selected_round"]!=max(trajectory,key=lambda r:r["validation_macro_f1"])["round"]: raise RuntimeError("invalid validation selection")
        if record["selected_tensor_sha256"]!=_tensor_hash(load(directory/"selected_state.pt")): raise RuntimeError("selected tensor mismatch")
        for p in directory.iterdir():
            if p.is_file(): hashes[str(p)]=sha(p)
        hashes_by_case.setdefault(row["case_id"],{})[row["name"]]=record["initial_tensor_sha256"]
    for values in hashes_by_case.values():
        if values["P"]!=values["I"]: raise RuntimeError("P/I pairing lost")
    value=dict(manifest_sha256=sha(manifest_path),all_models=len(manifest["matrix"]),artifact_sha256=hashes,
        sealed_before_evaluation=True,completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    publish(execution/"all_checkpoints_sealed.json",value); return value


def open_vault(manifest_path,execution,case):
    manifest=validate_manifest(manifest_path); sealed=read(Path(execution)/"all_checkpoints_sealed.json")
    if sealed["manifest_sha256"]!=sha(manifest_path) or sealed["all_models"]!=len(manifest["matrix"]): raise RuntimeError("all-model seal required")
    verify(sealed["artifact_sha256"])
    if case not in manifest["cases"]: raise RuntimeError("unknown case")
    return load(case["vault"])


class Progress:
    """Append-only progress plus callback-level budget checks; parent watchdog supplements it."""
    def __init__(self,root): self.root=Path(root); self.start=time.monotonic(); self.current={}; self.peak=0
    def __call__(self,event):
        event=dict(event) if isinstance(event,dict) else dict(phase=event)
        event["monotonic"]=time.monotonic(); self.peak=max(self.peak,psutil.Process().memory_info().rss)
        with (self.root/"events.jsonl").open("a") as f: f.write(json.dumps(event,allow_nan=False)+"\n"); f.flush()
        if self.peak>8*1024**3: raise RuntimeError("RSS budget exceeded")
    def begin(self,scope,seconds): self(dict(phase="budget_begin",scope=scope,seconds=seconds))
    def end(self,scope): self(dict(phase="budget_end",scope=scope))
