#!/usr/bin/env python
"""Preflight, freeze, then execute exactly the E28b development matrix."""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from fedbcert_e28b_common import (
    MODEL_NAMES, PROTOCOL, evaluate_validation, generate_graph, run_matrix,
    sha256, train_model, write_json,
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def frozen_sources():
    return [PROTOCOL, ROOT / "scripts/fedbcert_e28b_common.py",
            ROOT / "scripts/run_fedbcert_e28b_controlled.py",
            ROOT / "scripts/analyze_fedbcert_e28b_controlled.py",
            ROOT / "tests/test_fedbcert_e28b_controlled.py",
            ROOT / "docs/FEDBCERT_E28B_CONTROLLED_PROTOCOL_ZH.md"]


def protected_sources():
    paths = [ROOT / "openfgl/flcore/fedbcert" / name
             for name in ("certificate.py", "federated.py", "deployment.py", "__init__.py")]
    paths += list(ROOT.glob("FedBCert*投稿主稿*.md"))
    paths += [ROOT / "configs/fedbcert_e28a_exposure_protocol.json",
              ROOT / "results_boundary_certified/boundary_certified_spectral_e28a_exposure_v2/freeze_manifest.json",
              ROOT / "submissions/knowledge_based_systems_fedbcert_20260831/fedbcert_kbs.tex",
              ROOT / "submissions/knowledge_based_systems_fedbcert_20260831/fedbcert_anonymous_code_ready_20260907.zip"]
    paths += [path for path in (ROOT / "configs").glob("*e*.json")
              if re.search(r"_e(?:19|20|21|22|23|24)(?:_|\.)", path.name)]
    # The first development attempt failed in record serialization, not science.
    # Preserve its protocol/freeze/failure and exact original source snapshots.
    old = ROOT / "results_boundary_certified/boundary_certified_spectral_e28b_controlled"
    paths += [ROOT / "configs/fedbcert_e28b_controlled_protocol.json",
              old / "freeze_manifest.json", old / "development/failure.json"]
    paths += list(old.rglob("*"))
    paths += list((ROOT / "docs/snapshots/e28b_pre_recording_fix_20261004").rglob("*"))
    for folder in (ROOT / "results_boundary_certified").glob("*e*"):
        if re.search(r"_e(?:19|20|21|22|23|24)(?:_|$)", folder.name):
            paths += list(folder.rglob("*decision*.json"))
    return sorted(set(path for path in paths if path.is_file()))


def hash_index(paths):
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def seed_audit(protocol, output_root):
    """Audit explicit historical seed fields and seed-number artifact paths.

    Not a mathematical proof of unused randomness; prospective controls remain
    the real protection. The new protocol and proposal are not historical runs.
    """
    wanted = set(protocol["seeds"]["engineering"] + protocol["seeds"]["development"]
                 + protocol["seeds"]["confirmation_reserved_only"])
    hits, malformed, scanned = [], [], []
    path_pattern = re.compile(r"seed_?(\d+)(?!\d)")

    def inspect(value, path, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                inspect(child, path, str(name))
        elif isinstance(value, list):
            for child in value:
                inspect(child, path, key)
        elif "seed" in key.lower() and isinstance(value, (int, float)) and value in wanted:
            hits.append(dict(path=path, seed=int(value), field=key))

    for relative in ("configs", "results_boundary_certified", "logs"):
        base = ROOT / relative
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path == PROTOCOL or output_root == path or output_root in path.parents:
                continue
            if path.name == "fedbcert_e28b_controlled_protocol.json" or (
                ROOT / "results_boundary_certified/boundary_certified_spectral_e28b_controlled"
            ) in path.parents:
                # Disclosed replay of the same frozen matrix, not seed replacement.
                continue
            for match in path_pattern.finditer(str(path.relative_to(ROOT))):
                if int(match.group(1)) in wanted:
                    hits.append(dict(path=str(path.relative_to(ROOT)), seed=int(match.group(1)), field="artifact_path"))
            if path.is_file() and path.suffix == ".json":
                relative_path = str(path.relative_to(ROOT))
                scanned.append(relative_path)
                try:
                    inspect(json.loads(path.read_text(encoding="utf-8-sig")), relative_path)
                except (ValueError, UnicodeError) as exc:
                    malformed.append(dict(path=relative_path, error=str(exc)))
    if hits:
        raise RuntimeError(f"historical seed reuse: {hits}")
    import hashlib
    return dict(candidate_seeds=sorted(wanted), historical_seed_hits=hits,
                scope="explicit seed fields in JSON under configs/results_boundary_certified/logs and seed-number artifact paths; docs/proposals/new E28b protocol and disclosed failed E28b v1 replay excluded",
                scanned_json_count=len(scanned),
                scanned_path_list_sha256=hashlib.sha256("\n".join(scanned).encode()).hexdigest(),
                malformed_json_files=malformed, assurance="no reuse found in this audit scope; independent seed namespaces reserved prospectively")


def verify_manifest(manifest):
    for index in ("source_sha256", "protected_sha256"):
        for relative, expected in manifest[index].items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f"frozen/protected hash mismatch: {relative}")


def preflight(protocol, output_root, device):
    folder = output_root / "engineering_preflight"
    folder.mkdir(parents=True, exist_ok=False)
    seed = protocol["seeds"]["engineering"][0]
    started = time.monotonic()
    try:
        data = generate_graph(protocol, seed, .9, .4)
        reports = []
        for name in MODEL_NAMES:
            state, resource_report = train_model(protocol, data, name, device, None, rounds=2)
            _, evidence, audit = evaluate_validation(protocol, data, name, state)
            reliability = audit["reliability"]
            if reliability["certified_flip_nodes"] or reliability["bound_violations"]:
                torch.save(evidence, folder / f"{name}_failure_evidence.pt")
                raise RuntimeError("preflight reliability failure")
            # No engineering efficacy is published or used to choose settings.
            report = {key: value for key, value in resource_report.items()
                      if key not in {"selected_validation_macro_f1", "selected_round"}}
            reports.append({**report, **audit})
        write_json(folder / "resource_reliability.json", dict(
            phase="engineering_only", seed=seed, h=.9, rho=.4, nodes=len(data["features"]),
            rounds_per_model=2, efficacy_not_reported=True, models=reports,
            elapsed_seconds=time.monotonic() - started, created_at_utc=utc_now()))
        print("Engineering preflight passed: 3 models, 2 rounds each; no efficacy reported.", flush=True)
    except Exception:
        write_json(folder / "failure.json", dict(created_at_utc=utc_now(), traceback=traceback.format_exc()))
        raise


def freeze(protocol, output_root, device):
    preflight_path = output_root / "engineering_preflight/resource_reliability.json"
    if not preflight_path.is_file():
        raise RuntimeError("successful engineering preflight required before freeze")
    if (output_root / "development").exists() or (output_root / "freeze_manifest.json").exists():
        raise RuntimeError("freeze cannot overwrite prior artifacts")
    matrix = run_matrix(protocol)
    if len(matrix) != protocol["development"]["training_runs"]:
        raise RuntimeError("run matrix size mismatch")
    audit = seed_audit(protocol, output_root)
    old_manifest = json.loads((ROOT / "results_boundary_certified/boundary_certified_spectral_e28b_controlled/freeze_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in old_manifest["source_sha256"].items():
        archive = ROOT / "docs/snapshots/e28b_pre_recording_fix_20261004" / relative
        if sha256(archive) != expected:
            raise RuntimeError(f"original source snapshot mismatch: {relative}")
    manifest = dict(experiment="E28b", phase="development", frozen_at_utc=utc_now(),
                    source_sha256=hash_index(frozen_sources()), protected_sha256=hash_index(protected_sources()),
                    preflight_sha256=sha256(preflight_path), seed_audit=audit,
                    matrix=matrix, device=str(device), environment=dict(
                        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                    confirmation_authorized=False, test_evaluation_authorized=False,
                    recording_fix_disclosure=dict(
                        original_freeze="results_boundary_certified/boundary_certified_spectral_e28b_controlled/freeze_manifest.json",
                        original_failure="results_boundary_certified/boundary_certified_spectral_e28b_controlled/development/failure.json",
                        original_source_snapshot="docs/snapshots/e28b_pre_recording_fix_20261004/",
                        change="dict merge removes duplicated model keyword; output namespace only; no efficacy inspected, generator/models/seeds/budgets/gates unchanged",
                        replay="whole original matrix from beginning, same seeds; not a fresh independent replicate"))
    write_json(output_root / "freeze_manifest.json", manifest)
    print(f"Frozen {len(matrix)} development training runs; confirmation disabled.", flush=True)


def run_development(protocol, output_root, device):
    manifest_path = output_root / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    if manifest["matrix"] != run_matrix(protocol) or manifest["device"] != str(device):
        raise RuntimeError("matrix/device freeze mismatch")
    if sha256(output_root / "engineering_preflight/resource_reliability.json") != manifest["preflight_sha256"]:
        raise RuntimeError("preflight hash mismatch")
    folder = output_root / "development"
    folder.mkdir(exist_ok=False)  # No silent resume, overwrite, or seed replacement.
    started = time.monotonic()
    previous_id, data, initial_p = None, None, None
    try:
        for index, unit in enumerate(manifest["matrix"], 1):
            verify_manifest(manifest)
            if time.monotonic() - started > protocol["stopping"]["resource_caps"]["seconds_total_development"]:
                raise RuntimeError("total development wall-time cap")
            graph_id, name = unit["graph_id"], unit["model"]
            graph_folder = folder / graph_id
            if graph_id != previous_id:
                graph_folder.mkdir()
                data = generate_graph(protocol, unit["seed"], unit["h"], unit["rho"])
                torch.save(data, graph_folder / "sealed_graph_bundle.pt")
                write_json(graph_folder / "graph_manifest.json", dict(
                    **data["metadata"], bundle_sha256=sha256(graph_folder / "sealed_graph_bundle.pt"),
                    freeze_manifest_sha256=sha256(manifest_path), test_targets_sealed=True,
                    generator_contract_sha256=manifest["source_sha256"][str(PROTOCOL.relative_to(ROOT))]))
                previous_id, initial_p = graph_id, None
            print(f"[{index:02d}/{len(manifest['matrix'])}] start {graph_id}/{name}", flush=True)
            model_folder = graph_folder / name
            state, training = train_model(protocol, data, name, device, model_folder)
            if name == "observed_p":
                initial_p = training["initial_tensor_sha256"]
            elif name == "identity" and training["initial_tensor_sha256"] != initial_p:
                raise RuntimeError("P/I initial state mismatch")
            rows, evidence, audit = evaluate_validation(protocol, data, name, state)
            torch.save(evidence, model_folder / "validation_evidence.pt")
            record = {**unit, "phase": "development", "split": "validation", **training, **audit,
                      "protocol_sha256": sha256(PROTOCOL), "checkpoint_sha256": sha256(model_folder / "selected_state.pt"),
                      "evidence_sha256": sha256(model_folder / "validation_evidence.pt"), "rows": rows}
            write_json(model_folder / "validation_metrics.json", record)
            reliability = audit["reliability"]
            if reliability["certified_flip_nodes"] or reliability["bound_violations"]:
                raise RuntimeError("certified flip or analytic bound violation; saved counterexample")
            verify_manifest(manifest)
            print(f"[{index:02d}/{len(manifest['matrix'])}] done, training {training['training_seconds']:.2f}s, audit {audit['certification_diagnostic_seconds']:.2f}s; reliability passed", flush=True)
        write_json(folder / "completion.json", dict(
            phase="development", completed_training_runs=len(manifest["matrix"]),
            test_evaluated=False, confirmation_started=False, elapsed_seconds=time.monotonic() - started,
            freeze_manifest_sha256=sha256(manifest_path), finished_at_utc=utc_now()))
    except Exception:
        write_json(folder / "failure.json", dict(unit=unit, elapsed_seconds=time.monotonic() - started,
                                                  traceback=traceback.format_exc(), created_at_utc=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "freeze", "development"))
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    torch.set_num_threads(protocol["training"]["torch_cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_root = ROOT / protocol["artifacts"]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    {"preflight": preflight, "freeze": freeze, "development": run_development}[args.action](protocol, output_root, device)


if __name__ == "__main__":
    main()
