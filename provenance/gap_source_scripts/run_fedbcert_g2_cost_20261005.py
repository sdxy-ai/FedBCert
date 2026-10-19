"""Exclusive, label-free timing replay. No training or scientific model selection.

Prepare pins inputs/code BEFORE measurement. Five persistent, separate stage
workers per case execute a rotating schedule, one active worker at a time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from openfgl.flcore.fedbcert.certificate import (
    PolynomialSpectralClassifier, boundary_uncertainty, row_normalized_shift,
    retained_mass_uncertainty_basis, certificate_statistics,
    cone_pairwise_certificate_statistics,
)
from openfgl.flcore.fedbcert.federated import build_boundary_ledger_views

OUT = ROOT / "results_boundary_certified/fedbcert_g2_cert_only_cost_20261005"
SOURCE = ROOT / "results_boundary_certified/boundary_certified_spectral_e28b_confirmation"
SPEC = ROOT / "configs/fedbcert_e28b_controlled_protocol_v2.json"
STAGES = ("forward", "symmetric", "cone", "combined", "forward_combined")
FIELDS = ("features", "membership", "full_edge_index", "missing_degree")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(state):
    h = hashlib.sha256()
    for key, t in sorted(state.items()):
        if t.is_sparse:
            t = t.coalesce()
            h.update(digest({"indices": t.indices(), "values": t.values()}).encode())
        else:
            h.update(f"{key}:{t.dtype}:{tuple(t.shape)}".encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def extract_allowed(bundle):
    # Never access labels, masks or metadata (even indirectly through clients).
    return {key: bundle[key].detach().cpu().clone() for key in FIELDS}


def validate_allowed(data):
    if set(data) != set(FIELDS):
        raise ValueError("worker input must contain ONLY whitelisted tensors")
    x, m, e, b = (data[k] for k in FIELDS)
    n = len(x)
    if x.ndim != 2 or m.shape != (n,) or b.shape != (n,) or e.shape[0] != 2:
        raise ValueError("input shape mismatch")
    if not torch.isfinite(x).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite input")
    if (b < 0).any() or (b != b.round()).any() or (e < 0).any() or (e >= n).any():
        raise ValueError("invalid ledger or edge")


def make_model():
    p = read(SPEC)
    s, t = p["models"]["observed_p"], p["training"]
    return PolynomialSpectralClassifier(
        p["generator"]["feature_dim"], s["hidden_dim"], 2, t["order_count"],
        t["dropout"], s["encoder"], s["encoder_depth"], t["residual_scale"],
        spectral_fusion_scale=t["alpha"], certificate_margin_mode="pairwise_difference",
    ).double().eval()


class CachedHidden:
    """No parameter mutation; frozen certificate implementation gets cached H."""
    def __init__(self, model, hidden):
        self.model, self.hidden = model, hidden

    def __getattr__(self, name):
        return getattr(self.model, name)

    def bounded_encoding(self, features):
        return self.hidden


def prepare():
    if OUT.exists():
        raise FileExistsError("exclusive output already exists; no rerun/overwrite")
    OUT.mkdir(parents=True)
    commit = read(SOURCE / "checkpoint_commit.json")
    cases = []
    pinned = {}
    for path in [Path(__file__), SPEC, ROOT / "configs/fedbcert_gap_tasks_20261005.json",
                 ROOT / "tests/test_fedbcert_g2_cost_20261005.py",
                 ROOT / "openfgl/flcore/fedbcert/certificate.py",
                 ROOT / "openfgl/flcore/fedbcert/federated.py",
                 SOURCE / "checkpoint_commit.json"]:
        pinned[str(path.relative_to(ROOT))] = sha(path)
    for seed in range(28101, 28109):
        case = "h0.90_rho0.40_seed%d" % seed
        paths = [SOURCE / "training" / case / "sealed_graph_bundle.pt",
                 SOURCE / "training" / case / "graph_manifest.json",
                 SOURCE / "training" / case / "observed_p/selected_state.pt",
                 SOURCE / "training" / case / "observed_p/training_record.json"]
        start = time.perf_counter()
        for path in paths:
            relative = str(path.relative_to(SOURCE))
            observed = sha(path)
            if observed != commit["artifact_sha256"][relative]:
                raise ValueError("committed hash mismatch: " + relative)
            pinned[str(path.relative_to(ROOT))] = observed
        # Trusted bundle deserialization materializes sealed labels in this
        # preparation process. They are never indexed, inspected or exported.
        bundle = torch.load(paths[0], map_location="cpu", weights_only=False)
        allowed = extract_allowed(bundle)
        del bundle
        validate_allowed(allowed)
        state = torch.load(paths[2], map_location="cpu", weights_only=True)
        record = read(paths[3])
        if digest(state) != record["selected_tensor_sha256"]:
            raise ValueError("checkpoint tensor commitment mismatch")
        payload = OUT / (case + "_label_free.pt")
        with payload.open("xb") as handle:
            torch.save(allowed, handle)
        cases.append(dict(case=case, payload=str(payload.relative_to(ROOT)),
                          payload_sha256=sha(payload), tensor_sha256=digest(allowed),
                          checkpoint=str(paths[2].relative_to(ROOT)),
                          checkpoint_sha256=sha(paths[2]),
                          artifact_loading_hashing_extraction_seconds=time.perf_counter()-start))
    unavailable = []
    ogb = ROOT / "results_boundary_certified/boundary_certified_spectral_e18c_ogb_cone_scale/runs/dual_alpha025"
    for seed in (76, 77, 78):
        folder = ogb / ("ogbn-arxiv_random_c3_seed%d" % seed)
        path = folder / "protocol.json"
        p = read(path)
        audit = p["immutable_split_bundle_audit"]
        if audit["enabled"] or audit["hash_verified"]:
            raise ValueError("OGB provenance changed; review before inclusion")
        unavailable.append(dict(case=folder.name, status="unavailable",
            reason="membership and checkpoint exist, but no historical hash-bound features/local-view/ledger bundle or full nominal-logit anchor; unchecked reconstruction prohibited",
            protocol_sha256=sha(path), immutable_split_bundle_audit=audit,
            inventory=[str(q.relative_to(ROOT)) for q in sorted(folder.rglob("*")) if q.is_file()]))
        pinned[str(path.relative_to(ROOT))] = sha(path)
    manifest = dict(task="G2", status="frozen_before_measurement", cases=cases,
        unavailable=unavailable, source_sha256=pinned,
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
            platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
            device="cpu", dtype="float64", intraop=1, interop=1, sparse="coalesced COO"),
        warmups=5, repeats=30, stages=STAGES,
        schedule="five separate persistent workers per case; repetition r starts at stage r mod 5; one worker active at a time",
        boundaries=dict(cone="cached H/logits; includes basis, existing symmetric-statistics bookkeeping and nominal spectral-logit recomputation required by frozen implementation",
            combined="cached H/logits; includes basis and both certificates",
            forward_combined="uncached frozen forward plus additional H encoding plus combined; no internal forward refactoring",
            rss="parent samples active worker RSS every 1 ms plus before/after; sampled peak is not guaranteed true instantaneous peak; separate process lifetime HWM also reported"),
        caps=dict(seconds=3600, total_worker_rss_bytes=8*1024**3),
        no_training=True, no_label_scoring=True, no_threshold_changes=True)
    dump(OUT / "execution_manifest.json", manifest)
    print("Prepared and frozen 8 label-free cases; 3 OGB cases unavailable", flush=True)


def verify_manifest():
    p = read(OUT / "execution_manifest.json")
    for path, expected in p["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("frozen source changed: " + path)
    for case in p["cases"]:
        if sha(ROOT / case["payload"]) != case["payload_sha256"]:
            raise ValueError("label-free payload changed")
    return p


def finite_result(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite logits")
    else:
        # Own-class infinities are never present in worst_slack (C=2).
        for key in ("worst_slack", "representation_radius", "class_logit_bounds"):
            if not torch.isfinite(getattr(value, key)).all():
                raise ValueError("nonfinite certificate arithmetic")


def worker(case_id, stage):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    manifest = verify_manifest()
    case = next(c for c in manifest["cases"] if c["case"] == case_id)
    t0 = time.perf_counter()
    data = torch.load(ROOT / case["payload"], map_location="cpu", weights_only=True)
    validate_allowed(data)
    model = make_model()
    model.load_state_dict(torch.load(ROOT / case["checkpoint"], map_location="cpu", weights_only=True))
    initial_model = digest(model.state_dict())
    loading = time.perf_counter()-t0
    t0 = time.perf_counter()
    views = []
    for ledger in build_boundary_ledger_views(data["full_edge_index"], data["membership"].numpy(), data["missing_degree"]):
        x = data["features"][ledger.global_nodes].double()
        p, d = row_normalized_shift(ledger.local_edge_index, len(x), device=torch.device("cpu"), dtype=torch.float64)
        delta = boundary_uncertainty(d, ledger.missing_degree.double())
        views.append(dict(x=x, p=p, delta=delta, nodes=ledger.global_nodes))
    sparse_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    view_state = {"%d_%s" % (i,k): v[k] for i,v in enumerate(views) for k in ("x","p","delta","nodes")}
    initial_view = digest(view_state)
    for v in views:
        v["logits"] = model(v["x"], v["p"]).detach()
        v["h"] = model.bounded_encoding(v["x"]).detach()
        finite_result(v["logits"])
    snapshot_seconds = time.perf_counter()-t0

    def operation():
        outputs = []
        for v in views:
            x,p,delta = v["x"],v["p"],v["delta"]
            if stage == "forward":
                outputs.append(model(x,p))
                continue
            logits, h = v["logits"], v["h"]
            if stage == "forward_combined":
                # Standard dual forward computes bounded H; avoid changing
                # the frozen forward by caching after it (additional encoding
                # is charged, disclosed as implementation overhead).
                logits = model(x,p)
                h = model.bounded_encoding(x)
            basis = retained_mass_uncertainty_basis(p,delta,model.order_count)
            if stage == "symmetric":
                outputs.append(certificate_statistics(model,logits,basis))
            else:
                outputs.append(cone_pairwise_certificate_statistics(
                    CachedHidden(model,h),x,p,delta,logits,basis,
                    count_semantics="upper_bound",combine_symmetric=stage!="cone"))
        return outputs

    # Semantic checks are after measurement, so reference computation doesn't
    # contaminate setup-memory baseline or measured stage outputs.
    print(json.dumps(dict(ready=True, pid=os.getpid(), loading_seconds=loading,
        sparse_construction_seconds=sparse_seconds, snapshot_seconds=snapshot_seconds,
        n=sum(len(v["x"]) for v in views), local_directed_edges_with_self=sum(v["p"]._nnz() for v in views),
        terms=model.order_count, hidden=64, classes=2)), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        if command["op"] == "run":
            start = time.perf_counter()
            outputs = operation()
            elapsed = time.perf_counter()-start
            for value in outputs:
                finite_result(value)
            import resource
            print(json.dumps(dict(seconds=elapsed, process_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)), flush=True)
        elif command["op"] == "check":
            cached = operation()
            for v,value in zip(views,cached):
                if stage == "forward":
                    torch.testing.assert_close(value,v["logits"],rtol=0,atol=0)
                elif stage != "symmetric":
                    basis = retained_mass_uncertainty_basis(v["p"],v["delta"],model.order_count)
                    original = cone_pairwise_certificate_statistics(model,v["x"],v["p"],v["delta"],v["logits"],basis,
                        count_semantics="upper_bound",combine_symmetric=stage!="cone")
                    torch.testing.assert_close(value.worst_slack,original.worst_slack,rtol=0,atol=0)
                    if not torch.equal(value.certified,original.certified):
                        raise ValueError("cached adapter changed accepted nodes")
            if digest(model.state_dict()) != initial_model or digest(view_state) != initial_view or digest(data) != case["tensor_sha256"]:
                raise ValueError("input or model mutation")
            print(json.dumps(dict(checked=True, input_model_unchanged=True, cached_matches_frozen=True)),flush=True)
            return


def receive(proc, deadline):
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    try:
        if not sel.select(max(0,deadline-time.monotonic())):
            raise TimeoutError("task deadline")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("worker failed: " + proc.stderr.read()[-3000:])
        return json.loads(line)
    finally:
        sel.close()


def run():
    manifest = verify_manifest()
    if (OUT / "run_started.json").exists():
        raise FileExistsError("scientific timing replay already started; no rerun")
    dump(OUT / "run_started.json",dict(execution_manifest_sha256=sha(OUT/"execution_manifest.json"), started=time.time()))
    deadline = time.monotonic()+3600
    results = []
    for case in manifest["cases"]:
        workers, records = [], []
        try:
            for stage in STAGES:
                proc = subprocess.Popen([sys.executable,__file__,"worker","--case",case["case"],"--stage",stage],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,
                    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))
                workers.append(proc)
                setup = receive(proc,deadline)
                records.append(dict(stage=stage, setup=setup, baseline_rss_bytes=psutil.Process(proc.pid).memory_info().rss,
                                    sampled_peak_rss_bytes=0, raw_seconds=[], process_hwm_bytes=0))
            active = [-1]
            halt = threading.Event()
            breach = []
            def sample():
                while not halt.is_set():
                    sizes = [psutil.Process(p.pid).memory_info().rss for p in workers if p.poll() is None]
                    if sum(sizes) > 8*1024**3:
                        breach.append("total worker RSS exceeded 8 GiB")
                    i = active[0]
                    if i >= 0:
                        value = psutil.Process(workers[i].pid).memory_info().rss
                        records[i]["sampled_peak_rss_bytes"] = max(value,records[i]["sampled_peak_rss_bytes"])
                    halt.wait(.001)
            monitor = threading.Thread(target=sample,daemon=True)
            monitor.start()
            for r in range(35):
                for i in [(r+j)%5 for j in range(5)]:
                    if breach:
                        raise MemoryError(breach[0])
                    active[0] = i
                    workers[i].stdin.write('{"op":"run"}\n')
                    workers[i].stdin.flush()
                    answer = receive(workers[i],deadline)
                    records[i]["process_hwm_bytes"] = max(records[i]["process_hwm_bytes"],answer["process_hwm_bytes"])
                    records[i]["sampled_peak_rss_bytes"] = max(records[i]["sampled_peak_rss_bytes"],psutil.Process(workers[i].pid).memory_info().rss)
                    if r >= 5:
                        records[i]["raw_seconds"].append(answer["seconds"])
                    active[0] = -1
            halt.set()
            monitor.join()
            for i,p in enumerate(workers):
                p.stdin.write('{"op":"check"}\n')
                p.stdin.flush()
                records[i]["checks"] = receive(p,deadline)
                p.wait(timeout=10)
                values = records[i]["raw_seconds"]
                records[i].update(dict(median_seconds=float(np.median(values)),p10_seconds=float(np.percentile(values,10)),
                    p90_seconds=float(np.percentile(values,90)),sampled_increment_bytes=max(0,records[i]["sampled_peak_rss_bytes"]-records[i]["baseline_rss_bytes"])))
            ratio = records[3]["median_seconds"]/records[0]["median_seconds"]
            result = dict(case=case["case"],status="executed_and_checked",stages=records,combined_to_forward_ratio=ratio)
        except Exception as error:
            if "halt" in locals():
                halt.set()
            result = dict(case=case["case"],status="failed_no_rerun",reason=str(error),partial_stages=records)
        finally:
            for p in workers:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=10)
                for handle in (p.stdin,p.stdout,p.stderr):
                    handle.close()
        dump(OUT/(case["case"]+"_timing.json"),result)
        results.append(result)
        print(case["case"]+": "+result["status"],flush=True)
    verify_manifest()
    dump(OUT / "summary.json",dict(task="G2",status="terminal",results=results,unavailable=manifest["unavailable"],
        new_independent_confirmation=False, no_low_overhead_claim=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command",choices=["prepare","run","worker"])
    parser.add_argument("--case")
    parser.add_argument("--stage",choices=STAGES)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    else: worker(args.case,args.stage)
