"""RG-1.0 preparation and gated execution. No command starts formal work implicitly."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import psutil
import torch
from scripts import fedbcert_rg1_common as common
from scripts.fedbcert_rg1_training import train_one,make_model,_tensor_hash
from scripts.fedbcert_rg1_evaluation import evaluate_case
from scripts.fedbcert_rg1_analysis import analyze


def configure():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    if Path(sys.prefix).resolve()!=common.PROJECT/"environments/FedBCert_RG1_20261020": raise RuntimeError("use the frozen independent RG1 environment")


def train_matrix(manifest_path,out,authorization=None,progress=None):
    manifest=common.validate_manifest(manifest_path); common.authorize(authorization,manifest_path,"train",out)
    out=Path(out); progress=progress or common.Progress(out)
    for row in manifest["matrix"]:
        progress.begin("model",manifest["budgets"]["model_seconds"])
        # Only this local-only capability is deserialized here. The evaluation
        # vault, test targets, full graph and certificate outputs are not passed.
        payload=common.load(row["training_inputs"])
        if set(payload)!={"clients"}: raise RuntimeError("training capability contains extra data")
        initial=common.load(row["initial_state"])
        train_one(row["spec"],payload["clients"],initial,row["model_seed"],out/"training"/row["case_id"]/row["name"],device=manifest["device"],checkpoint=progress)
        common.verify({row["training_inputs"]:manifest["input_sha256"][row["training_inputs"]],row["initial_state"]:manifest["input_sha256"][row["initial_state"]]})
        progress.end("model")
    common.validate_manifest(manifest_path); return common.seal(manifest_path,out)


def evaluate_matrix(manifest_path,out,authorization=None,progress=None):
    manifest=common.validate_manifest(manifest_path); common.authorize(authorization,manifest_path,"evaluate",out)
    out=Path(out); progress=progress or common.Progress(out); records=[]; case_metrics=[]
    for ci,case in enumerate(manifest["cases"]):
        vault=common.open_vault(manifest_path,out,case)
        models={}
        for name in ("P","I","MLP"):
            row=next(r for r in manifest["matrix"] if r["case_id"]==case["case_id"] and r["name"]==name)
            model=make_model(row["spec"],dtype=torch.float64).eval(); model.load_state_dict(common.load(out/"training"/case["case_id"]/name/"selected_state.pt")); models[name]=model
        def event(stage):
            progress(stage)
            if stage.endswith(":start"): progress.begin("record",manifest["budgets"]["record_seconds"])
            elif stage.endswith(":complete"): progress.end("record")
        result=evaluate_case(models,vault["allowed"],vault["test_nodes"],vault["test_labels"],out/"evaluation"/case["case_id"],ci,case["batch_seed"],timing=manifest["timing"],checkpoint=event)
        key={k:case[k] for k in ("dataset","partition","seed")}
        for r in result["timing_records"]:
            if r["warmup"]: continue
            name=r["method"]; records.append(dict(r,**key,model="P" if name in {"L","F","E","S","D","N"} else name,strategy=name if name in {"L","F","E","S","D","N"} else "L"))
        metrics=result["metrics"]["all_test"]
        utility={}
        for model,name in (("P","L"),("I","I"),("MLP","MLP")):
            a,e=metrics[name]["all"],metrics[name]["exposed"]
            utility[model]=dict(macro_f1=a["macro_f1"],accuracy=a["accuracy"],exposed_macro_f1=None if e is None else e["macro_f1"],worst_client_f1=a["worst_client_macro_f1"])
        case_metrics.append(dict(**key,utility=utility,coverage={n:metrics[n]["all"]["certified_coverage"] for n in ("S","D","N")},structural_diagnostics=result["structural_diagnostics"]))
        common.validate_manifest(manifest_path)
    summary=analyze(records,case_metrics,formal=manifest["mode"]!="engineering_toy")
    common.publish(out/"all_records.json",records); common.publish(out/"case_metrics.json",case_metrics); common.publish(out/"analysis.json",summary)
    common.publish(out/"evaluation_complete.json",dict(records=len(records),cases=len(case_metrics),manifest_sha256=common.sha(manifest_path),all_inputs_unchanged=True))
    return summary


def dry_run(out):
    out=Path(out); manifest_path=common.prepare(out/"toy_inputs",toy=True)
    for repeat in (1,2):
        execution=out/f"repeat_{repeat}"; execution.mkdir(); progress=common.Progress(execution)
        train_matrix(manifest_path,execution,progress=progress); evaluate_matrix(manifest_path,execution,progress=progress)
    manifest=common.read(manifest_path); state_checks=0
    for row in manifest["matrix"]:
        for name in ("initial_state.pt","selected_state.pt","last_state.pt"):
            a=common.load(out/"repeat_1/training"/row["case_id"]/row["name"]/name); b=common.load(out/"repeat_2/training"/row["case_id"]/row["name"]/name)
            if _tensor_hash(a)!=_tensor_hash(b): raise RuntimeError("deterministic toy state mismatch")
            state_checks+=1
    records1=common.read(out/"repeat_1/all_records.json"); records2=common.read(out/"repeat_2/all_records.json")
    def key(r): return r["dataset"],r["partition"],r["seed"],r["workload"],r["model"],r["strategy"],r["repetition"]
    second={key(r):r for r in records2}
    for r in records1:
        for field in ("total_wire_bytes","target_n","certified_n","fallback_n"):
            if r[field]!=second[key(r)][field]: raise RuntimeError("deterministic toy count mismatch")
    # Exact node results and diagnostics, excluding measured wall/process times.
    leaf_count=0; excluded_times=0
    def compare(a,b):
        nonlocal leaf_count,excluded_times
        if isinstance(a,torch.Tensor):
            if a.dtype!=b.dtype or a.shape!=b.shape or not torch.allclose(a,b,rtol=0,atol=0,equal_nan=True): raise RuntimeError("deterministic toy tensor mismatch")
            leaf_count+=1
        elif isinstance(a,dict):
            if a.keys()!=b.keys(): raise RuntimeError("toy output schema changed")
            for k in a:
                if k=="inference_seconds": excluded_times+=1
                else: compare(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            if len(a)!=len(b): raise RuntimeError("toy sequence changed")
            for x,y in zip(a,b): compare(x,y)
        elif a!=b: raise RuntimeError("toy non-timing output changed")
    for case in manifest["cases"]:
        for name in ("node_results.pt","diagnostics.pt"):
            compare(common.load(out/"repeat_1/evaluation"/case["case_id"]/name),common.load(out/"repeat_2/evaluation"/case["case_id"]/name))
    common.publish(out/"dry_run_validation.json",dict(passed=True,mode="engineering_toy",formal_data_read=False,
        models_per_repeat=len(manifest["matrix"]),rounds_per_model=2,repeats=2,
        exact_state_comparisons=state_checks,exact_tensor_leaves=leaf_count,excluded_inference_seconds_fields=excluded_times,formal_records_per_repeat=len(records1),
        all_six_P_policies=True,all_models_sealed_before_evaluation=True,deterministic_non_timing_outputs=True))


def worker(args):
    configure(); out=Path(args.out)
    if args.action=="prepare": common.prepare(out/"frozen_inputs"); return
    if args.action=="dry-run": dry_run(out); return
    progress=common.Progress(out)
    if args.action=="train": train_matrix(args.manifest,out,args.authorization,progress)
    elif args.action=="evaluate": evaluate_matrix(args.manifest,out,args.authorization,progress)


def supervise(args):
    out=Path(args.out).resolve()
    if args.action=="evaluate":
        if not (out/"all_checkpoints_sealed.json").is_file(): raise RuntimeError("no all-model seal")
        if (out/"evaluation").exists(): raise FileExistsError("evaluation already started; no automatic retry")
    else: out.mkdir(parents=True,exist_ok=False)
    if args.action in {"train","evaluate"}: common.authorize(args.authorization,args.manifest,args.action,out)
    common.publish(out/(args.action+"_execution_start.json"),dict(execution_id=uuid.uuid4().hex,
        action=args.action,output_directory=str(out),manifest_sha256=common.sha(args.manifest) if args.manifest else None,
        authorization_sha256=common.sha(args.authorization) if args.authorization else None,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONHASHSEED="0")
    cmd=[sys.executable,"-B",str(Path(__file__).resolve()),args.action,"--out",str(out),"--worker"]
    if args.manifest: cmd += ["--manifest",str(Path(args.manifest).resolve())]
    if args.authorization: cmd += ["--authorization",str(Path(args.authorization).resolve())]
    start=time.monotonic(); peak=0; deadlines={}; offsets={}; reason=None
    storage_roots=[out]
    if args.manifest:
        input_root=Path(args.manifest).resolve().parent
        if not input_root.is_relative_to(out): storage_roots.append(input_root)
    cap={"prepare":1800,"dry-run":900,"train":43200,"evaluate":7200}[args.action]
    rss_cap=(2 if args.action=="dry-run" else 8)*1024**3
    def interrupted(*_): raise RuntimeError("supervised interruption")
    with (out/(args.action+".log")).open("x") as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            last_storage=0; storage=0
            while process.poll() is None:
                now=time.monotonic()
                try:
                    p=psutil.Process(process.pid); live=[p,*p.children(recursive=True)]; rss=sum(v.memory_info().rss for v in live); peak=max(peak,rss)
                except psutil.NoSuchProcess: rss=0
                event_paths=out.glob("**/events.jsonl")
                for path in event_paths:
                    position=offsets.get(path,0)
                    with path.open() as stream:
                        stream.seek(position)
                        for line in stream:
                            event=json.loads(line)
                            if event["phase"]=="budget_begin": deadlines[(str(path),event["scope"])]=event["monotonic"]+event["seconds"]
                            elif event["phase"]=="budget_end": deadlines.pop((str(path),event["scope"]),None)
                        offsets[path]=stream.tell()
                if now-last_storage>2:
                    storage=sum(p.stat().st_size for root in storage_roots for p in root.rglob("*") if p.is_file()); last_storage=now
                if peak>rss_cap: reason="RSS budget exceeded"
                if now-start>cap: reason="phase timeout"
                if any(now>d for d in deadlines.values()): reason="per-model/record timeout"
                if storage>10*1024**3: reason="10 GiB output cap exceeded"
                if reason:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
                    break
                time.sleep(.05)
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM); raise
        code=process.wait()
    common.publish(out/(args.action+"_resources.json"),dict(exit_code=code,wall_seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak,stop_reason=reason,phase_seconds_cap=cap,rss_cap_bytes=rss_cap,storage_cap_bytes=10*1024**3,storage_roots=[str(p) for p in storage_roots]))
    if code or reason:
        common.publish(out/(args.action+"_supervisor_failure.json"),dict(exit_code=code,reason=reason or "worker failed; see retained full log"))
        raise SystemExit("STOP: "+args.action+" failed; see "+str(out))
    print(f"{args.action}: PASS; evidence={out}")


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("action",choices=("prepare","dry-run","train","evaluate")); parser.add_argument("--out",required=True); parser.add_argument("--manifest"); parser.add_argument("--authorization"); parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.action in {"train","evaluate"} and not args.manifest: parser.error("--manifest required")
    if args.worker:
        def stop(*_): raise RuntimeError("resource supervisor interrupted worker")
        signal.signal(signal.SIGTERM,stop)
        try: worker(args)
        except BaseException as error:
            path=Path(args.out)/(args.action+"_failure.json")
            if not path.exists(): common.publish(path,dict(type=type(error).__name__,reason=str(error),traceback=traceback.format_exc()))
            raise
    else: supervise(args)


if __name__=="__main__": main()
