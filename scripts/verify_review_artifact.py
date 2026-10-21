#!/usr/bin/env python
"""Check compact review hashes/statistics; no training or sealed target access."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import numpy as np
from fedbcert_e28b_common import sha256
from fedbcert_e28b_confirmation_common import (
    estimate, exact_sign_p, holm_adjust, load_protocol, resample_indices,
    validate_specification,
)


def verify(root=ROOT):
    root = Path(root).resolve()
    index = json.loads((root / "PACKAGE_SHA256.json").read_text())
    for relative, expected in index.items():
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or sha256(candidate) != expected:
            raise RuntimeError(f"package hash/path mismatch: {relative}")
    historical = json.loads((root / "HISTORICAL_SOURCE_SHA256.json").read_text())
    locations_path = root / "HISTORICAL_SOURCE_LOCATIONS.json"
    locations = json.loads(locations_path.read_text()) if locations_path.exists() else {}
    for relative, expected in historical.items():
        source = (root / locations.get(relative, relative)).resolve()
        if not source.is_relative_to(root) or sha256(source) != expected:
            raise RuntimeError(f"historical source not preserved: {relative}")
    spec, base = load_protocol()
    validate_specification(spec, base)
    summary = json.loads((root / "reference_results/e28b_confirmation/summary.json").read_text())
    if len(summary["per_seed"]) != 96 or summary["training_runs"] != 288:
        raise RuntimeError("reference matrix incomplete")
    expected_units = {(seed, h, rho) for seed in spec["confirmation"]["seeds"]
                      for h in spec["confirmation"]["homophily"]
                      for rho in spec["confirmation"]["cross_client_mass"]}
    actual = [(r["seed"], r["h"], r["rho"]) for r in summary["per_seed"]]
    if len(set(actual)) != len(actual) or set(actual) != expected_units:
        raise RuntimeError("reference seeds/cells mismatch")
    indices = resample_indices(spec)
    for cell in summary["cells"]:
        rows = [r for r in summary["per_seed"] if r["h"] == cell["h"] and r["rho"] == cell["rho"]]
        rows.sort(key=lambda r: r["seed"])
        for metric, original in cell["estimates"].items():
            actual_estimate = estimate([r[metric] for r in rows], indices)
            if actual_estimate != original:
                raise RuntimeError(f"paired reference estimate mismatch: {metric}")
    pvalues = []
    for test in summary["primary_tests"]:
        p, successes = exact_sign_p(test["seed_values"], test["null_margin"])
        if p != test["raw_p"] or successes != test["successes"]:
            raise RuntimeError("reference exact sign test mismatch")
        primary = [r for r in summary["per_seed"] if r["h"] == .9 and r["rho"] == .4]
        primary.sort(key=lambda r: r["seed"])
        if [r[test["metric"]] for r in primary] != test["seed_values"]:
            raise RuntimeError("primary seed values mismatch")
        pvalues.append(p)
    if holm_adjust(pvalues) != [t["holm_p"] for t in summary["primary_tests"]]:
        raise RuntimeError("Holm mismatch")
    print(f"PASS: {len(index)} compact hashes; {len(historical)} preserved sources; 96 paired cells/seeds; four sign/Holm tests.")
    print("No checkpoint archive audit, training, or test-target evaluation was performed.")
    return len(index)


if __name__ == "__main__":
    verify()
