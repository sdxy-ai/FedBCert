#!/usr/bin/env python3
"""Recompute statistics from saved messages; do not execute graph inference."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_fedbcert_on_demand_20261015 import (
    analyze_records, csv_text, read_json, require, sha256, summary_rows, verify_index,
)


def verify(root=ROOT):
    root = Path(root).resolve()
    # The current release index is separate from the original archive indexes.
    verify_index(root, read_json(root / "PACKAGE_SHA256.json"))
    base = root / "results_boundary_certified/fedbcert_on_demand_20261015"
    pilot = base / "pilot"
    runs = read_json(pilot / "runs.json")
    freeze = read_json(pilot / "freeze_manifest.json")
    audit = read_json(base / "input_audit.json")
    protocol = read_json(root / "configs/fedbcert_on_demand_20261015.json")
    require(runs["schema_version"] == 1, "unexpected run schema")
    require(sha256(root / "configs/fedbcert_on_demand_20261015.json")
            == runs["protocol_sha256"] == freeze["protocol_sha256"], "protocol binding")
    require(sha256(base / "input_audit.json")
            == runs["input_audit_sha256"] == freeze["input_audit_sha256"], "audit binding")
    require(sha256(pilot / "freeze_manifest.json") == runs["freeze_manifest_sha256"], "freeze binding")
    require(runs["frozen_input_sha256"] == freeze["frozen_input_sha256"]
            == audit["verified_file_sha256"], "historical input-index binding")
    require(runs["inputs_unchanged_after"] and runs["implementation_unchanged_after"], "recorded integrity failure")
    require(audit["passed"] and audit["source_hashes_unchanged_after_audit"], "recorded audit failure")
    require(tuple(protocol["seeds"]) == tuple(range(28101, 28109)), "seed mismatch")
    verify_index(root, freeze["source_sha256"])
    verify_index(root, runs["evidence_sha256"])
    report, rows = analyze_records(runs["records"])
    saved = read_json(pilot / "analysis/summary.json")
    # These two fields describe the recorded original archive audit, not this check.
    require(saved["engineering_gates"]["protocol_input_implementation_integrity"] is True,
            "original integrity gate was not recorded as passed")
    expected = {k:v for k,v in saved.items() if k != "integrity"}
    expected["engineering_gates"] = {k:v for k,v in saved["engineering_gates"].items()
                                     if k != "protocol_input_implementation_integrity"}
    require(report == expected, "saved statistics differ")
    require(csv_text(rows).splitlines() == (pilot / "analysis/per_seed.csv").read_text().splitlines(), "per-seed CSV differs")
    require(csv_text(summary_rows(report, rows)).splitlines()
            == (pilot / "analysis/workload_summary.csv").read_text().splitlines(), "summary CSV differs")
    primary = report["cone_comparisons"]["all_test"]
    reduction = primary["full_reference"]["relative_payload_reduction"]["mean"]
    ratio = primary["full_reference"]["cone_to_baseline_cpu_ratio"]["mean"]
    print(f"PASS: 240 records; payload reduction {100*reduction:.2f}%; CPU increase {100*(ratio-1):.2f}%.")
    print("Saved-record analysis only; original graph/checkpoint archive was not re-audited or executed.")
    return report


if __name__ == "__main__":
    verify()
