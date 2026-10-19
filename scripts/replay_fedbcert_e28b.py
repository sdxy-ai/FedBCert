#!/usr/bin/env python
"""Portable engineering smoke / historical E28b replay, never new confirmation.

The frozen numerical engine and protocols are not modified. Full replay keeps
the original all-checkpoint test gate, but protects the included package files
instead of requiring private historical directories. Original phase strings in
raw records are legacy schema values; replay_manifest.json is authoritative.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch
from fedbcert_e28b_common import (
    MODEL_NAMES, evaluate_validation, generate_graph, sha256, train_model,
    write_json,
)
from fedbcert_e28b_confirmation_common import (
    PROTOCOL, estimate, exact_sign_p, holm_adjust, load_protocol, matrix,
    paired_metrics, resample_indices, validate_specification,
)


def package_index(root=ROOT):
    path = Path(root) / "PACKAGE_SHA256.json"
    if not path.is_file():
        raise RuntimeError("run inside the generated anonymous code package")
    index = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in index.items():
        candidate = (Path(root) / relative).resolve()
        if not candidate.is_relative_to(Path(root).resolve()):
            raise RuntimeError("unsafe manifest path")
        if sha256(candidate) != expected:
            raise RuntimeError(f"package changed: {relative}")
    return index


def reserve_output(path):
    output = Path(path).resolve()
    # Exclusive creation: even an empty preexisting directory is not reused.
    output.mkdir(parents=True, exist_ok=False)
    return output


def engineering_smoke(base, output):
    """Tiny CPU run: validation only, no published test target evaluation."""
    # Engineering-only size/budget overrides are prominently recorded.
    data = generate_graph(base, 28000, .9, .4, block_size=20)
    initial_p, reports = None, []
    for name in MODEL_NAMES:
        state, report = train_model(base, data, name, torch.device("cpu"),
                                   output / name, rounds=2)
        if name == "observed_p":
            initial_p = report["initial_tensor_sha256"]
        elif name == "identity" and report["initial_tensor_sha256"] != initial_p:
            raise RuntimeError("P/I initial state mismatch")
        rows, evidence, audit = evaluate_validation(base, data, name, state)
        expected = data["masks"]["validation"]
        if not bool(expected[evidence["nodes"]].all()):
            raise RuntimeError("smoke evidence not validation-only")
        if audit["reliability"]["certified_flip_nodes"] or audit["reliability"]["bound_violations"]:
            raise RuntimeError("smoke reliability failure")
        reports.append(dict(model=name, training=report, rows=rows, audit=audit))
    write_json(output / "smoke_report.json", dict(
        phase="engineering_smoke", scientific_evidence=False,
        independent_confirmation=False, split="validation", seed=28000,
        nodes=120, rounds=2, models=reports, audit_targets_used=False,
        published_test_targets_used=False))
    print("PASS: 120-node/2-round offline smoke; validation only; not scientific evidence.")


def historical_replay(spec, base, output, device, source_index):
    from run_fedbcert_e28b_confirmation import train_all, test_all
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(output / "replay_manifest.json", dict(
        phase="historical_reproduction", independent_confirmation=False,
        fresh_seeds=False, published_seeds_reused=True,
        output_scope="exclusive replay directory, never historical result roots",
        raw_phase_strings="legacy original-engine schema, NOT independent evidence",
        protocol_sha256=sha256(PROTOCOL), created_at_utc=now))
    # The original engine checks all included immutable sources before/after
    # each model and requires the same full 288-checkpoint commit before tests.
    write_json(output / "freeze_manifest.json", dict(
        experiment="E28b-historical-reproduction", frozen_at_utc=now,
        source_sha256=source_index, protected_sha256={}, matrix=matrix(spec),
        confirmation_authorized=True, independent_confirmation=False,
        authorization="explicit --full-replay; package-local historical reproduction",
        test_unlock_requires_all_checkpoints=True, device=str(device)))
    train_all(spec, base, output, device)
    test_all(spec, base, output)
    records = {}
    for unit in matrix(spec):
        path = output / "test_evaluation" / unit["graph_id"] / unit["model"] / "test_metrics.json"
        records[(unit["seed"], unit["h"], unit["rho"], unit["model"])] = json.loads(path.read_text())
    paired = []
    for seed in spec["confirmation"]["seeds"]:
        for h in spec["confirmation"]["homophily"]:
            for rho in spec["confirmation"]["cross_client_mass"]:
                values = paired_metrics(*(records[(seed, h, rho, name)] for name in MODEL_NAMES))
                paired.append(dict(seed=seed, h=h, rho=rho, **values))
    indices = resample_indices(spec)
    cells = []
    for h in spec["confirmation"]["homophily"]:
        for rho in spec["confirmation"]["cross_client_mass"]:
            rows = [r for r in paired if r["h"] == h and r["rho"] == rho]
            keys = [k for k in rows[0] if k not in ("seed", "h", "rho")]
            cells.append(dict(h=h, rho=rho, estimates={k: estimate([r[k] for r in rows], indices) for k in keys}))
    primary = spec["confirmation"]["primary_cell"]
    rows = [r for r in paired if r["h"] == primary["h"] and r["rho"] == primary["rho"]]
    tests = []
    for hypothesis in spec["statistics"]["primary_family"]:
        key, margin = hypothesis["metric"], hypothesis["null_margin"]
        p, successes = exact_sign_p([r[key] for r in rows], margin)
        tests.append(dict(metric=key, null_margin=margin, raw_p=p, successes=successes,
                          **estimate([r[key] for r in rows], indices)))
    for test, adjusted in zip(tests, holm_adjust([t["raw_p"] for t in tests])):
        test["holm_p"] = adjusted
    write_json(output / "replay_summary.json", dict(
        phase="historical_reproduction", independent_confirmation=False,
        training_runs=288, graph_units=96, primary_cell=primary,
        cells=cells, primary_tests=tests, per_seed=paired,
        decision="replay_completed_not_new_confirmatory_evidence"))
    package_index()
    print("Historical replay completed. This is NOT new independent confirmation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full-replay", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_index = package_index()
    spec, base = load_protocol()
    validate_specification(spec, base)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = reserve_output(args.output)
    try:
        if args.smoke:
            engineering_smoke(base, output)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            historical_replay(spec, base, output, device, source_index)
        package_index()
    except Exception:
        import traceback
        write_json(output / "REPLAY_FAILURE.json", dict(
            phase="historical_reproduction" if args.full_replay else "engineering_smoke",
            independent_confirmation=False, traceback=traceback.format_exc(),
            action="preserve failure; no automatic retry/overwrite"))
        raise


if __name__ == "__main__":
    main()
