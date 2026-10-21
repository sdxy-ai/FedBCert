#!/usr/bin/env python
"""Read-only compact G1/G2 validation. Never load tensors or run experiments."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("forward", "symmetric", "cone", "combined", "forward_combined")
CASES = {f"h0.90_rho0.40_seed{seed}" for seed in range(28101, 28109)}
OGB = {f"ogbn-arxiv_random_c3_seed{seed}" for seed in range(76, 79)}
# These two files were already reviewer-portability derivatives in the prior
# package (license text and external dependency location only), not byte-exact
# research sources. Keep both identities distinct; do not claim original pins.
PORTABLE_DERIVATIVES = {
    "configs/fedbcert_e26_known_universe_verifier_bridge.json":
        "e6ca923931437454072be421c9def652a12a450befb18ab0ad9cefbedace1a11",
    "scripts/run_fedbcert_e26_gnnev_bridge.py":
        "071ec9e60c664a5503ca17ecff3486c5472c8b9c3ea34fdc872980b90aafd208",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_records(g1, manifest, summary, timing):
    require(g1["status"] == "terminal_not_comparable_historical_input_gate_failed",
            "G1 archival-input stop changed")
    trials = g1["trial_status"]
    require(len(trials) == 40 and {t["seed"] for t in trials} == set(range(2601, 2641)),
            "G1 trial inventory incomplete")
    require(all(t["known_completions"] == 4 and t["status"] == "pre_execution_unavailable"
                for t in trials), "G1 unavailable semantics changed")
    require(g1["numeric_comparisons"] == g1["translated_cases"] == 0 and
            g1["precision"] is None and g1["solved_instance_ratio"] is None and
            g1["official_candidate_available"] is True and
            g1["same_threat_superiority_claim_allowed"] is False and
            all(s["downloaded_or_executed"] is False for s in g1["official_sources"]),
            "G1 unavailable misreported as numeric performance")
    require(all(g1[key] is True for key in ("no_regeneration", "no_training", "no_easy_case_search")),
            "G1 stop rules changed")
    require(manifest["status"] == "frozen_before_measurement" and
            manifest["warmups"] == 5 and manifest["repeats"] == 30 and
            tuple(manifest["stages"]) == STAGES, "G2 frozen schedule changed")
    require(all(manifest[k] is True for k in
                ("no_training", "no_label_scoring", "no_threshold_changes")),
            "G2 scope changed")
    require(summary["status"] == "terminal" and
            summary["new_independent_confirmation"] is False and
            summary["no_low_overhead_claim"] is True, "G2 claim scope changed")
    rows = summary["results"]
    require(len(rows) == 8 and {r["case"] for r in rows} == CASES and
            set(timing) == CASES and {r["case"] for r in manifest["cases"]} == CASES,
            "G2 case inventory incomplete")
    require(len(manifest["cases"]) == 8, "G2 duplicate manifest cases")
    unavailable = summary["unavailable"]
    require(unavailable == manifest["unavailable"] and len(unavailable) == 3 and
            {r["case"] for r in unavailable} == OGB and
            all(r["status"] == "unavailable" and
                r["immutable_split_bundle_audit"]["enabled"] is False and
                r["immutable_split_bundle_audit"]["hash_verified"] is False for r in unavailable),
            "G2 OGB unavailable inventory changed")
    medians = {s: [] for s in STAGES}
    increments = {s: [] for s in STAGES}
    ratios = []
    for row in rows:
        require(row == timing[row["case"]], "G2 summary/raw record mismatch")
        require(row["status"] == "executed_and_checked", "G2 case status changed")
        stages = row["stages"]
        require(len(stages) == 5 and tuple(s["stage"] for s in stages) == STAGES,
                "G2 stage inventory incomplete")
        for stage in stages:
            raw = stage["raw_seconds"]
            require(len(raw) == 30 and all(math.isfinite(x) and x > 0 for x in raw),
                    "G2 repetitions nonfinite or incomplete")
            values = [statistics.median(raw), float(np.quantile(raw, .1)),
                      float(np.quantile(raw, .9))]
            for key, value in zip(("median_seconds", "p10_seconds", "p90_seconds"), values):
                require(math.isclose(stage[key], value, rel_tol=0, abs_tol=1e-14),
                        "G2 statistic mismatch: " + key)
            require(stage["sampled_increment_bytes"] == max(0,
                    stage["sampled_peak_rss_bytes"] - stage["baseline_rss_bytes"]),
                    "G2 RSS increment mismatch")
            require(all(stage["checks"].get(k) is True for k in
                        ("checked", "input_model_unchanged", "cached_matches_frozen")),
                    "G2 semantic-check flags failed")
            setup = stage["setup"]
            require(setup["ready"] is True and
                    (setup["n"], setup["terms"], setup["hidden"], setup["classes"]) ==
                    (6000, 3, 64, 2), "G2 workload changed")
            medians[stage["stage"]].append(values[0])
            increments[stage["stage"]].append(stage["sampled_increment_bytes"] / 2**20)
        ratios.append(stages[3]["median_seconds"] / stages[0]["median_seconds"])
    return {"formal_measurements": 1200, "measured_cases": 8, "unavailable_ogb_cases": 3,
            "g1_numeric_comparisons": 0,
            "median_ms": {s: 1000 * statistics.median(v) for s, v in medians.items()},
            "median_sampled_increment_mib": {s: statistics.median(v) for s, v in increments.items()},
            "median_same_case_combined_forward_ratio": statistics.median(ratios)}


def verify(root=ROOT):
    root = Path(root)
    ref = root / "reference_results/gaps"
    g1, manifest, summary = (read(ref / name) for name in
                            ("g1/gate_audit.json", "g2/execution_manifest.json", "g2/summary.json"))
    started = read(ref / "g2/run_started.json")
    require(started["execution_manifest_sha256"] == sha(ref / "g2/execution_manifest.json"),
            "G2 start/manifest binding mismatch")
    timing = {read(path)["case"]: read(path) for path in (ref / "g2").glob("*_timing.json")}
    require(len(list((ref / "g2").glob("*_timing.json"))) == 8,
            "G2 raw file inventory incomplete")
    # Only bundled sources are authenticated here. Omitted original tensors,
    # tests and checkpoint/input archives are never claimed verified.
    bindings = dict(manifest["source_sha256"])
    for original, expected in g1["pins"].items():
        require(original not in bindings or bindings[original] == expected,
                "G1/G2 source binding conflict")
        bindings[original] = expected
    verified = 0
    derivative_checks = []
    for original, expected in bindings.items():
        candidate = root / original
        # Earlier timing records bind the earlier source, not the B1/B2 runtime.
        # Preserve and verify that source separately instead of rewriting its pin.
        archived = root / "provenance/pre_b1_b2" / original
        if candidate.is_file() and sha(candidate) != expected and archived.is_file():
            require(sha(archived) == expected, "Archived source hash mismatch: " + original)
            candidate = archived
        if not candidate.is_file() and original.startswith("scripts/"):
            candidate = root / "provenance/gap_source_scripts" / Path(original).name
        if original.endswith("/checkpoint_commit.json"):
            candidate = root / "reference_results/e28b_confirmation/checkpoint_commit.json"
        if candidate.is_file():
            if original in PORTABLE_DERIVATIVES:
                require(sha(candidate) == PORTABLE_DERIVATIVES[original],
                        "Portable E26 derivative changed: " + original)
                derivative_checks.append(original)
                continue
            require(sha(candidate) == expected, "Bundled frozen source hash mismatch: " + original)
            verified += 1
    result = verify_records(g1, manifest, summary, timing)
    result["bundled_source_bindings_checked"] = verified
    result["portable_derivatives_not_original_source_pins"] = derivative_checks
    print("PASS: read-only G1/G2 compact records; no tensors loaded or measurements rerun.")
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    verify()
