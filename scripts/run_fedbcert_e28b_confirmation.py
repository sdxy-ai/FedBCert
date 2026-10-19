#!/usr/bin/env python
"""Freeze -> train all -> commit all -> unlock confirmation tests."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import numpy as np
import torch
from fedbcert_e28b_common import generate_graph, resource_usage, sha256, train_model, write_json
from fedbcert_e28b_confirmation_common import (
    PROTOCOL, check_index, evaluate_test, load_protocol, matrix, require_test_unlock, validate_specification,
)
from run_fedbcert_e28b_controlled import verify_manifest


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def historical_seed_audit(seeds, output):
    wanted, hits, malformed, files = set(seeds), [], [], []
    pattern = re.compile(r"seed_?(\d+)(?!\d)")

    def inspect(value, relative, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                inspect(child, relative, str(name))
        elif isinstance(value, list):
            for child in value:
                inspect(child, relative, key)
        elif key.lower() in {"seed", "whole_graph_seed", "graph_seed", "data_seed", "model_seed"} and isinstance(value, int) and value in wanted:
            hits.append(dict(path=relative, field=key, seed=value))

    for base in (ROOT / "results_boundary_certified", ROOT / "logs"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path == output or output in path.parents:
                continue
            relative = str(path.relative_to(ROOT))
            for match in pattern.finditer(relative):
                if int(match[1]) in wanted:
                    hits.append(dict(path=relative, field="artifact_path", seed=int(match[1])))
            if path.is_file() and path.suffix == ".json":
                files.append(relative)
                try:
                    inspect(json.loads(path.read_text(encoding="utf-8-sig")), relative)
                except (ValueError, UnicodeError) as error:
                    malformed.append(dict(path=relative, error=str(error)))
    if hits or malformed:
        raise RuntimeError(f"seed audit incomplete/reused: hits={hits}, malformed={malformed}")
    return dict(seeds=sorted(wanted), hits=hits, malformed=malformed, scanned_json_count=len(files),
                scanned_path_list_sha256=hashlib.sha256("\n".join(files).encode()).hexdigest(),
                scope="actual seed fields and seed-number artifact paths under results_boundary_certified/logs; planned/reserved seed lists are not evidence of consumption")


def freeze(spec, base, output, device):
    validate_specification(spec, base)
    if (output / "freeze_manifest.json").exists() or (output / "training").exists():
        raise RuntimeError("refuse to overwrite existing confirmation")
    dev = ROOT / base["artifacts"]["output_root"]
    old = json.loads((dev / "freeze_manifest.json").read_text(encoding="utf-8"))
    verify_manifest(old)
    development = json.loads((dev / "analysis/summary.json").read_text(encoding="utf-8"))
    if development["decision"] != "eligible_for_separately_authorized_confirmation":
        raise RuntimeError("development gate did not permit confirmation")
    sources = dict(old["source_sha256"])
    for relative in spec["artifacts"]["sources"]:
        sources[relative] = sha256(ROOT / relative)
    protected = dict(old["protected_sha256"])
    for path in dev.rglob("*"):
        if path.is_file():
            protected[str(path.relative_to(ROOT))] = sha256(path)
    for relative in ("docs/FEDBCERT_E28B_CONTROLLED_RESULTS_ZH.md", "scripts/audit_fedbcert_e28b_artifacts.py"):
        protected[relative] = sha256(ROOT / relative)
    manifest = dict(experiment=spec["experiment"], frozen_at_utc=utc_now(),
                    source_sha256=sources, protected_sha256=protected,
                    matrix=matrix(spec), confirmation_authorized=True,
                    test_unlock_requires_all_checkpoints=True, device=str(device),
                    development_summary_sha256=sha256(dev / "analysis/summary.json"),
                    seed_audit=historical_seed_audit(spec["confirmation"]["seeds"], output),
                    environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    write_json(output / "freeze_manifest.json", manifest)
    print("Frozen independent confirmation: 96 graphs / 288 trainings; all-checkpoint test gate enabled.", flush=True)


def guard_resources(spec, device, start, phase):
    caps = spec["resources"]
    usage = resource_usage(device)
    key = "seconds_total_training" if phase == "train" else "seconds_total_test_audit"
    if time.monotonic() - start > caps[key] or any(usage[k] > caps[k] for k in usage):
        raise RuntimeError("confirmation resource hard stop")
    return usage


def verified_manifest(spec, output, device=None):
    manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    verify_manifest(manifest)
    if manifest["matrix"] != matrix(spec) or not manifest["confirmation_authorized"]:
        raise RuntimeError("frozen confirmation matrix/authorization mismatch")
    if device is not None and manifest["device"] != str(device):
        raise RuntimeError("device changed after freeze")
    return manifest


def train_all(spec, base, output, device):
    manifest = verified_manifest(spec, output, device)
    folder = output / "training"
    folder.mkdir(exist_ok=False)
    start, data, previous, initial_p = time.monotonic(), None, None, None
    unit, artifact_index = None, {}
    try:
        for number, unit in enumerate(manifest["matrix"], 1):
            verify_manifest(manifest)
            guard_resources(spec, device, start, "train")
            graph_id, name = unit["graph_id"], unit["model"]
            graph_folder = folder / graph_id
            if graph_id != previous:
                graph_folder.mkdir()
                data = generate_graph(base, unit["seed"], unit["h"], unit["rho"])
                torch.save(data, graph_folder / "sealed_graph_bundle.pt")
                write_json(graph_folder / "graph_manifest.json", dict(
                    **data["metadata"], bundle_sha256=sha256(graph_folder / "sealed_graph_bundle.pt"),
                    test_targets_used=False, confirmation_protocol_sha256=sha256(PROTOCOL)))
                previous, initial_p = graph_id, None
            print(f"train [{number:03d}/288] {graph_id}/{name}", flush=True)
            model_folder = graph_folder / name
            _, report = train_model(base, data, name, device, model_folder)
            if name == "observed_p":
                initial_p = report["initial_tensor_sha256"]
            if name == "identity" and report["initial_tensor_sha256"] != initial_p:
                raise RuntimeError("P/I initial state mismatch")
            write_json(model_folder / "training_record.json", {
                **unit, **report, "phase": "confirmation_training", "test_targets_used": False,
                "protocol_sha256": sha256(PROTOCOL), "checkpoint_sha256": sha256(model_folder / "selected_state.pt")})
            guard_resources(spec, device, start, "train")
            verify_manifest(manifest)
            print(f"done [{number:03d}/288] {report['training_seconds']:.2f}s, hashes/resources passed; efficacy withheld", flush=True)
        # This independent index is complete and irrevocable before first test use.
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                artifact_index[str(path.relative_to(output))] = sha256(path)
        write_json(output / "checkpoint_commit.json", dict(
            experiment=spec["experiment"], all_training_complete=True, training_runs=288, graph_units=96,
            matrix=manifest["matrix"], artifact_sha256=artifact_index,
            protocol_sha256=sha256(PROTOCOL), freeze_manifest_sha256=sha256(output / "freeze_manifest.json"),
            test_evaluated_before_commit=False, committed_at_utc=utc_now(),
            training_elapsed_seconds=time.monotonic() - start, **resource_usage(device)))
        print("All 288 checkpoints committed. Test targets have not been evaluated.", flush=True)
    except Exception:
        write_json(folder / "failure.json", dict(unit=unit, traceback=traceback.format_exc(),
                                                 elapsed_seconds=time.monotonic() - start, at_utc=utc_now()))
        raise


def test_all(spec, base, output):
    manifest = verified_manifest(spec, output)
    commit = require_test_unlock(output, spec)  # Checks all training files before any target access.
    if commit["freeze_manifest_sha256"] != sha256(output / "freeze_manifest.json") or commit["test_evaluated_before_commit"]:
        raise RuntimeError("checkpoint commit mismatch")
    folder = output / "test_evaluation"
    folder.mkdir(exist_ok=False)
    start, previous, data, reference = time.monotonic(), None, None, None
    cpu, unit, records_index = torch.device("cpu"), None, {}
    try:
        for number, unit in enumerate(manifest["matrix"], 1):
            verify_manifest(manifest)
            guard_resources(spec, cpu, start, "test")
            graph_id, name = unit["graph_id"], unit["model"]
            train_folder = output / "training" / graph_id
            if graph_id != previous:
                data_path = train_folder / "sealed_graph_bundle.pt"
                if sha256(data_path) != commit["artifact_sha256"][str(data_path.relative_to(output))]:
                    raise RuntimeError("graph bundle changed after commit")
                data = torch.load(data_path, map_location="cpu", weights_only=False)
                (folder / graph_id).mkdir()
                previous, reference = graph_id, None
            model_folder = folder / graph_id / name
            model_folder.mkdir()
            state_path = train_folder / name / "selected_state.pt"
            if sha256(state_path) != commit["artifact_sha256"][str(state_path.relative_to(output))]:
                raise RuntimeError("checkpoint changed after commit")
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            rows, evidence, audit = evaluate_test(base, data, name, state, commit_verified=True)
            if name == "observed_p":
                reference = evidence
            elif not torch.equal(evidence["nodes"], reference["nodes"]) or any(
                not torch.equal(value, reference["groups"][key]) for key, value in evidence["groups"].items()
            ):
                raise RuntimeError("controls do not share observed-P exposure groups")
            torch.save(evidence, model_folder / "test_evidence.pt")
            write_json(model_folder / "test_metrics.json", {
                **unit, **audit, "rows": rows, "phase": "independent_confirmation", "split": "test",
                "checkpoint_sha256": sha256(state_path), "evidence_sha256": sha256(model_folder / "test_evidence.pt"),
                "protocol_sha256": sha256(PROTOCOL), "commit_sha256": sha256(output / "checkpoint_commit.json"),
                **resource_usage(cpu)})
            if audit["reliability"]["certified_flip_nodes"] or audit["reliability"]["bound_violations"]:
                raise RuntimeError("certified flip/bound violation; counterexample preserved")
            guard_resources(spec, cpu, start, "test")
            records_index[str((model_folder / "test_metrics.json").relative_to(output))] = sha256(model_folder / "test_metrics.json")
            if number % 12 == 0 or number == 1:
                print(f"test [{number:03d}/288] original test masks only; reliability/resources passed; efficacy withheld", flush=True)
        check_index(output, commit["artifact_sha256"])
        verify_manifest(manifest)
        write_json(folder / "completion.json", dict(
            experiment=spec["experiment"], completed_test_runs=288, graph_units=96,
            split="test", checkpoints_mutated=False, records_sha256=records_index,
            checkpoint_commit_sha256=sha256(output / "checkpoint_commit.json"),
            test_elapsed_seconds=time.monotonic() - start, finished_at_utc=utc_now(), **resource_usage(cpu)))
        print("All 288 confirmation test audits complete; final efficacy analysis can now run.", flush=True)
    except Exception:
        write_json(folder / "failure.json", dict(unit=unit, traceback=traceback.format_exc(),
                                                 elapsed_seconds=time.monotonic() - start, at_utc=utc_now()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "train", "test"))
    args = parser.parse_args()
    spec, base = load_protocol()
    validate_specification(spec, base)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = ROOT / spec["artifacts"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "freeze":
        freeze(spec, base, output, device)
    elif args.action == "train":
        train_all(spec, base, output, device)
    else:
        test_all(spec, base, output)


if __name__ == "__main__":
    main()
