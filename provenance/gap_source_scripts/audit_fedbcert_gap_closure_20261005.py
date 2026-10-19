"""Read-only independent timing/statistics/provenance audit, optional exclusive seal."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "results_boundary_certified/fedbcert_g1_comparability_20261005"
G2 = ROOT / "results_boundary_certified/fedbcert_g2_cert_only_cost_20261005"
OUT = ROOT / "results_boundary_certified/fedbcert_gap_closure_20261005"
STAGES = ["forward","symmetric","cone","combined","forward_combined"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition: raise ValueError(message)


def validate_stage(stage):
    samples = stage["raw_seconds"]
    require(len(samples)==30 and all(math.isfinite(x) and x>0 for x in samples),"invalid repeat distribution")
    for key,value in [("median_seconds",statistics.median(samples)),
                      ("p10_seconds",float(np.percentile(samples,10))),
                      ("p90_seconds",float(np.percentile(samples,90)))]:
        require(math.isclose(stage[key],value,rel_tol=0,abs_tol=1e-14),"timing statistic mismatch: "+key)
    require(stage["checks"]==dict(checked=True,input_model_unchanged=True,cached_matches_frozen=True),"missing semantic/input check")
    for key in ("baseline_rss_bytes","sampled_peak_rss_bytes","sampled_increment_bytes","process_hwm_bytes"):
        require(isinstance(stage[key],int) and 0<=stage[key]<8*1024**3,"invalid RSS")
    require(stage["sampled_increment_bytes"]==max(0,stage["sampled_peak_rss_bytes"]-stage["baseline_rss_bytes"]),"RSS increment mismatch")
    setup=stage["setup"]
    require(setup["n"]==6000 and setup["terms"]==3 and setup["hidden"]==64 and setup["classes"]==2,"model dimension mismatch")


def audit():
    g1,m,s = read(G1/"gate_audit.json"),read(G2/"execution_manifest.json"),read(G2/"summary.json")
    require(g1["status"]=="terminal_not_comparable_historical_input_gate_failed","G1 not terminal")
    require(g1["numeric_comparisons"]==0 and g1["precision"] is None and g1["solved_instance_ratio"] is None,"fake comparative result")
    require(len(g1["trial_status"])==40 and all(x["status"]=="pre_execution_unavailable" and x["known_completions"]==4 for x in g1["trial_status"]),"G1 all-case accounting mismatch")
    require(g1["official_candidate_available"] and not g1["same_threat_superiority_claim_allowed"],"unsupported external-method conclusion")
    for group in (g1["pins"],g1["historical_analysis_inventory"],m["source_sha256"]):
        for path,expected in group.items():
            require(sha(ROOT/path)==expected,"frozen source changed: "+path)
    require(m["status"]=="frozen_before_measurement" and m["warmups"]==5 and m["repeats"]==30,"unfrozen timing protocol")
    started=read(G2/"run_started.json")
    require(started["execution_manifest_sha256"]==sha(G2/"execution_manifest.json"),"execution manifest changed after start")
    require([c["case"] for c in s["results"]]==["h0.90_rho0.40_seed%d"%i for i in range(28101,28109)],"case search or omission")
    require(len(s["unavailable"])==3 and all(c["status"]=="unavailable" for c in s["unavailable"]),"OGB availability accounting mismatch")
    require(s["unavailable"]==m["unavailable"],"OGB gating changed after measurement")
    for case in m["cases"]:
        require(sha(ROOT/case["payload"])==case["payload_sha256"],"label-free input changed")
    for case in s["results"]:
        require(case==read(G2/(case["case"]+"_timing.json")),"per-case output mismatch")
        require(case["status"]=="executed_and_checked","failed timing case requires explicit limitation review")
        require([r["stage"] for r in case["stages"]]==STAGES,"stage mismatch")
        for stage in case["stages"]: validate_stage(stage)
        require(math.isclose(case["combined_to_forward_ratio"],case["stages"][3]["median_seconds"]/case["stages"][0]["median_seconds"],rel_tol=0,abs_tol=1e-14),"ratio mismatch")
    aggregate=[]
    for i,name in enumerate(STAGES):
        rows=[c["stages"][i] for c in s["results"]]
        medians=[r["median_seconds"] for r in rows]
        aggregate.append(dict(stage=name,median_of_case_medians_ms=1000*statistics.median(medians),
            min_case_median_ms=1000*min(medians),max_case_median_ms=1000*max(medians),
            median_sampled_increment_mib=statistics.median([r["sampled_increment_bytes"] for r in rows])/1024**2,
            max_sampled_increment_mib=max(r["sampled_increment_bytes"] for r in rows)/1024**2))
    ratios=[c["combined_to_forward_ratio"] for c in s["results"]]
    return dict(status="both_tasks_terminal_and_audited", task_outcomes=dict(G1="not_comparable_archival_gate",G2="8 executed, 3 OGB unavailable"),
        aggregate=aggregate,combined_to_forward_ratio=dict(median=statistics.median(ratios),minimum=min(ratios),maximum=max(ratios)),
        scientific_scope="timing replay only; no new model training, test scoring, independent efficacy confirmation or same-threat external-verifier ranking",
        limitations=["G1 does not establish external-tool superiority or inability", "G2 does not establish large-graph cert-only overhead",
            "RSS sampling is not an instantaneous true-peak bound", "float64 replay is not historical float32 training or formal floating-point soundness"],
        final_experimental_freeze_complete=True,english_publication_sync_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seal",action="store_true")
    p.add_argument("--verify-seal",action="store_true")
    args=p.parse_args()
    require(not(args.seal and args.verify_seal),"choose one seal action")
    result=audit()
    if args.seal or args.verify_seal:
        # Earlier evidence-use seal is independently checked, never replaced.
        old=read(ROOT/"results_boundary_certified/fedbcert_evidence_use_20261005/manifest.json")
        for group in (old["artifact_sha256"],old["historical_protected_sha256"]):
            for path,expected in group.items(): require(sha(ROOT/path)==expected,"prior evidence seal changed: "+path)
        documents=[ROOT/"docs/FEDBCERT_GAP_CLOSURE_20261005_ZH.md",
            ROOT/"FedBCert：端点盲补全集认证（实验收口候选稿20261005）.md",
            Path(__file__),ROOT/"tests/test_fedbcert_gap_closure_20261005.py"]
        payloads=documents+[ROOT/"results_boundary_certified/fedbcert_evidence_use_20261005/manifest.json"]
        payloads += [q for folder in (G1,G2) for q in sorted(folder.rglob("*")) if q.is_file()]
        result["artifact_sha256"]={str(q.relative_to(ROOT)):sha(q) for q in payloads}
        result["prior_historical_protected_file_count"]=len(old["historical_protected_sha256"])
        if args.seal:
            require(not OUT.exists(),"refuse to overwrite final freeze")
            OUT.mkdir()
            with (OUT/"manifest.json").open("x",encoding="utf-8") as handle:
                json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
                handle.write("\n")
        else:
            require(read(OUT/"manifest.json")==result,"final freeze mismatch")
    print(json.dumps(result if not(args.seal or args.verify_seal) else {"status":result["status"],"sealed_files":len(result["artifact_sha256"])},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
