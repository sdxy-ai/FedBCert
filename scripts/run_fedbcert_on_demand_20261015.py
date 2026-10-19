"""Frozen, label-free application replay; never trains or selects models."""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from audit_fedbcert_on_demand_inputs_20261015 import sha256, write_exclusive_json
from fedbcert_e28b_common import make_model
from openfgl.flcore.fedbcert.certificate import (
    boundary_uncertainty, retained_mass_uncertainty_basis, certificate_statistics,
    cone_pairwise_certificate_statistics, row_normalized_shift,
)
from openfgl.flcore.fedbcert.geometry_baselines import evaluate_geometry_baselines
from openfgl.flcore.fedbcert.on_demand import AggregateReplayService, potential_exposure, state_fingerprint

PROTOCOL = ROOT / "configs/fedbcert_on_demand_20261015.json"
SOURCES = (
    "configs/fedbcert_on_demand_20261015.json",
    "scripts/run_fedbcert_on_demand_20261015.py",
    "scripts/audit_fedbcert_on_demand_inputs_20261015.py",
    "scripts/analyze_fedbcert_on_demand_20261015.py",
    "openfgl/flcore/fedbcert/on_demand.py",
    "openfgl/flcore/fedbcert/geometry_baselines.py",
    "openfgl/flcore/fedbcert/certificate.py",
    "openfgl/flcore/fedbcert/federated.py",
    "scripts/fedbcert_e28b_common.py",
    "tests/test_fedbcert_on_demand.py",
    "tests/test_fedbcert_geometry_baselines.py",
    "tests/test_fedbcert_on_demand_runner_20261015.py",
    "tests/test_fedbcert_on_demand_analysis_20261015.py",
    "tests/test_fedbcert_on_demand_inputs_20261015.py",
    "docs/FEDBCERT_ON_DEMAND_PROTOCOL_20261015_ZH.md",
    "docs/FEDBCERT_CERTIFICATE_DOMINANCE_20261015_ZH.md",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_hashes(mapping):
    for relative, expected in mapping.items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"frozen file changed: {relative}")


def workload_targets(nodes, count, seed):
    if count == len(nodes):
        return nodes.clone()
    if not 0 < count < len(nodes):
        raise ValueError("invalid target count")
    positions = np.random.default_rng(28615 + seed).choice(len(nodes), count, replace=False)
    return nodes[torch.from_numpy(np.sort(positions))]


def prepare():
    spec = read(PROTOCOL)
    out = ROOT / spec["output_root"]
    if out.exists():
        raise FileExistsError(f"exclusive pilot directory already exists: {out}")
    audit_path = ROOT / spec["input_audit"]
    audit = read(audit_path)
    if not audit["passed"] or [case["seed"] for case in audit["cases"]] != spec["seeds"]:
        raise RuntimeError("complete fixed-cell input audit required")
    verify_hashes(audit["verified_file_sha256"])
    sources = {name: sha256(ROOT / name) for name in SOURCES}
    manifest = dict(schema_version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha256(PROTOCOL), input_audit_sha256=sha256(audit_path),
        frozen_input_sha256=audit["verified_file_sha256"], source_sha256=sources,
        environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                         numpy=np.__version__, cpu_threads=1, dtype="float64"),
        interpretation=spec["scope"], outcomes_observed_before_freeze=False)
    out.mkdir(parents=True)
    write_exclusive_json(out / "freeze_manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "cases": len(audit["cases"]),
                      "protocol_sha256": manifest["protocol_sha256"]}), flush=True)


class CachedLocalHidden:
    def __init__(self, model, hidden):
        self._model, self._hidden = model, hidden

    def __getattr__(self, name):
        return getattr(self._model, name)

    def bounded_encoding(self, features):
        return self._hidden


@torch.no_grad()
def certificate_for_strategy(model, view, nominal, row_budget, strategy, guard):
    basis = retained_mass_uncertainty_basis(view.shift, row_budget, model.order_count)
    if strategy == "cone":
        stats = cone_pairwise_certificate_statistics(
            CachedLocalHidden(model, nominal.hidden), view.features, view.shift,
            row_budget, nominal.logits, basis, count_semantics="upper_bound", combine_symmetric=False)
    elif strategy in {"ball_pairwise", "nonnegative_diameter"}:
        if strategy == "nonnegative_diameter":
            basis = [value / math.sqrt(2.0) for value in basis]
        stats = certificate_statistics(model, nominal.logits, basis)
    else:
        raise ValueError("not a certificate strategy")
    if not torch.isfinite(stats.worst_slack).all():
        raise FloatingPointError("nonfinite certificate")
    return stats.worst_slack > guard, stats.worst_slack


@torch.no_grad()
def run_strategy(model, allowed, targets, strategy, guard):
    """Fixture initializes the private service; routing reads only local views.

    ``allowed`` is a service fixture, not verifier input. Its remote topology
    and features are passed only into the service constructor. Labels and
    central-reference predictions never enter this function.
    """
    started = time.perf_counter()
    service = AggregateReplayService(model, allowed["features"], allowed["full_edge_index"], allowed["membership"])
    setup_seconds = time.perf_counter() - started
    n, classes = len(allowed["features"]), model.classifier.out_features
    nominal_all = torch.full((n, classes), float("nan"), dtype=torch.float64)
    certified = torch.zeros(n, dtype=torch.bool)
    slacks = torch.full((n,), float("nan"), dtype=torch.float64)
    metadata_bytes = metadata_packets = metadata_seconds = nominal_seconds = certificate_seconds = 0
    if strategy != "full_reference":
        target_owners = torch.unique(allowed["membership"][targets]).tolist()
        for owner in target_owners:
            view = service.local_view(owner)
            stage = time.perf_counter()
            nominal = service.nominal(owner)
            nominal_seconds += time.perf_counter() - stage
            nominal_all[view.nodes] = nominal.logits
            stage = time.perf_counter()
            receipt = service.metadata(owner, "presence" if strategy == "exposure_only" else "upper_count")
            metadata_seconds += time.perf_counter() - stage
            metadata_bytes += receipt.request_bytes + receipt.response_bytes
            metadata_packets += 2
            stage = time.perf_counter()
            if strategy == "exposure_only":
                certified[view.nodes] = ~potential_exposure(view.shift, receipt.values.bool(), model.order_count)
            else:
                row_budget = boundary_uncertainty(view.retained_degree, receipt.values.to(view.features.dtype))
                accepted, slack = certificate_for_strategy(model, view, nominal, row_budget, strategy, guard)
                certified[view.nodes], slacks[view.nodes] = accepted, slack
            certificate_seconds += time.perf_counter() - stage
    unresolved = targets[~certified[targets]]
    stage = time.perf_counter()
    fallback = service.exact(unresolved)
    fallback_seconds = time.perf_counter() - stage
    final = nominal_all[targets].clone()
    inverse = torch.full((n,), -1, dtype=torch.long)
    inverse[targets] = torch.arange(len(targets))
    final[inverse[fallback.nodes]] = fallback.logits
    if not torch.isfinite(final).all():
        raise RuntimeError("nonfinite routed output")
    elapsed = time.perf_counter() - started
    account = fallback.accounting
    wire = account["request_bytes"] + account["response_bytes"]
    row = dict(target_n=len(targets), certified_n=int(certified[targets].sum()), fallback_n=len(unresolved),
        metadata_wire_bytes=int(metadata_bytes), aggregate_request_bytes=account["request_bytes"],
        aggregate_response_bytes=account["response_bytes"], aggregate_wire_bytes=wire,
        total_wire_bytes=int(metadata_bytes + wire), remote_vectors=account["remote_vectors"],
        request_response_packets=account["request_response_packets"] + metadata_packets,
        aggregate_request_response_packets=account["request_response_packets"],
        metadata_request_response_packets=metadata_packets, cpu_seconds=elapsed,
        service_setup_seconds=setup_seconds, nominal_seconds=nominal_seconds,
        metadata_seconds=metadata_seconds, certificate_seconds=certificate_seconds,
        fallback_seconds=fallback_seconds,
        fallback_accounting={key: value for key, value in account.items() if key != "stages"})
    evidence = dict(targets=targets, final_logits=final, nominal_logits=nominal_all[targets],
                    certified=certified[targets], slacks=slacks[targets],
                    fallback_nodes=fallback.nodes, fallback_logits=fallback.logits,
                    fallback_accounting=account)
    return row, evidence


@torch.no_grad()
def validate_case(base, bundle, state, historical, spec):
    """Offline checker; none of its reference outputs enters the policy."""
    model = make_model(base, "observed_p", dtype=torch.float64).eval()
    model.load_state_dict(state, strict=True)
    allowed = {key: bundle[key].detach().cpu().clone() for key in
               ("features", "membership", "full_edge_index")}
    allowed["features"] = allowed["features"].double()
    nodes = torch.where(bundle["masks"]["test_sealed"])[0]
    if not torch.equal(nodes, historical["nodes"]):
        raise RuntimeError("historical test node order changed")
    shift, _ = row_normalized_shift(allowed["full_edge_index"], len(allowed["features"]),
                                   device=torch.device("cpu"), dtype=torch.float64)
    reference = model(allowed["features"], shift)
    atol = spec["numerics"]["reference_logit_atol"]
    if not torch.allclose(reference[nodes], historical["completion_logits"], rtol=0, atol=atol):
        raise RuntimeError("central fullgraph reference differs from historical snapshot")
    service = AggregateReplayService(model, **dict(features=allowed["features"],
        edge_index=allowed["full_edge_index"], membership=allowed["membership"]))
    nominal = torch.empty_like(reference)
    rules = {name: torch.empty(len(reference), dtype=torch.bool) for name in
             ("ball_pairwise", "nonnegative_diameter", "cone", "combined")}
    slack_arrays = {name: torch.empty(len(reference), dtype=torch.float64) for name in rules}
    for owner in torch.unique(allowed["membership"]).tolist():
        view, current = service.local_view(owner), service.nominal(owner)
        nominal[view.nodes] = current.logits
        receipt = service.metadata(owner, "upper_count")
        delta = boundary_uncertainty(view.retained_degree, receipt.values.to(torch.float64))
        stats = evaluate_geometry_baselines(model, view.features, view.shift, delta,
                                           logits=current.logits, guard=spec["numerics"]["guard"])
        for name, stat in stats.items():
            rules[name][view.nodes], slack_arrays[name][view.nodes] = stat.certified, stat.worst_slack
        for name in ("ball_pairwise", "nonnegative_diameter", "cone"):
            accepted, slack = certificate_for_strategy(model, view, current, delta, name, spec["numerics"]["guard"])
            if not torch.equal(accepted, stats[name].certified) or not torch.allclose(
                    slack, stats[name].worst_slack, rtol=0, atol=atol):
                raise RuntimeError(f"timed certificate path mismatch: {name}")
    if not torch.equal(nominal[nodes], historical["nominal_logits"]):
        raise RuntimeError("nominal forward differs from historical logits")
    for new, old in (("cone", "cone_only"), ("ball_pairwise", "retained_pairwise")):
        if not torch.equal(rules[new][nodes], historical["certificates"][old]):
            raise RuntimeError(f"historical certificate mask changed: {new}")
    if not torch.equal(rules["cone"], rules["combined"]):
        raise RuntimeError("unexpected extra hybrid acceptance; inspect numerical boundary")
    for tighter, weaker in (("cone", "nonnegative_diameter"), ("nonnegative_diameter", "ball_pairwise")):
        if bool((slack_arrays[tighter] + spec["numerics"]["dominance_atol"] < slack_arrays[weaker]).any()):
            raise RuntimeError("geometry dominance check failed")
    return model, allowed, nodes, reference, nominal


def check_result(row, result, reference, nominal_reference, targets, spec):
    prediction = result["final_logits"].argmax(1)
    ref_classes = reference[targets].argmax(1)
    cert = result["certified"]
    row["class_mismatches"] = int((prediction != ref_classes).sum())
    row["certified_class_mismatches"] = int(((prediction != ref_classes) & cert).sum())
    row["nominal_vs_full_class_mismatches"] = int((nominal_reference[targets].argmax(1) != ref_classes).sum())
    ids = result["fallback_nodes"]
    error = float((result["fallback_logits"] - reference[ids]).abs().max()) if len(ids) else 0.0
    row["fallback_max_abs_error"] = error
    if row["class_mismatches"] or error > spec["numerics"]["reference_logit_atol"]:
        raise RuntimeError(f"class/numerical replay mismatch: {row}")


@torch.no_grad()
def run():
    spec = read(PROTOCOL)
    out = ROOT / spec["output_root"]
    manifest = read(out / "freeze_manifest.json")
    if sha256(PROTOCOL) != manifest["protocol_sha256"]:
        raise RuntimeError("protocol changed after freeze")
    if sha256(ROOT / spec["input_audit"]) != manifest["input_audit_sha256"]:
        raise RuntimeError("input audit changed after freeze")
    verify_hashes(manifest["frozen_input_sha256"])
    verify_hashes(manifest["source_sha256"])
    write_exclusive_json(out / "execution_started.json", dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        freeze_manifest_sha256=sha256(out / "freeze_manifest.json")))
    started = time.monotonic()
    records, evidence_index = [], {}
    (out / "cases").mkdir()
    audit = read(ROOT / spec["input_audit"])
    base = read(ROOT / "configs/fedbcert_e28b_controlled_protocol_v2.json")
    try:
        for case_index, case in enumerate(audit["cases"]):
            seed, graph_id = case["seed"], case["graph_id"]
            bundle = torch.load(ROOT / case["bundle"], map_location="cpu", weights_only=False)
            state = torch.load(ROOT / case["selected_state"], map_location="cpu", weights_only=True)
            historical = torch.load(ROOT / case["historical_test_evidence"], map_location="cpu", weights_only=False)
            model, allowed, nodes, reference, nominal = validate_case(base, bundle, state, historical, spec)
            before = state_fingerprint(model)
            case_records, saved = [], {}
            for workload_index, workload in enumerate(spec["workloads"]):
                targets = workload_targets(nodes, workload["count"], seed)
                for repetition in range(-spec["timing"]["warmup"], spec["timing"]["repetitions"]):
                    shift = (case_index + workload_index + repetition) % len(spec["strategies"])
                    order = spec["strategies"][shift:] + spec["strategies"][:shift]
                    for strategy in order:
                        if time.monotonic() - started > 1800:
                            raise TimeoutError("frozen total replay time cap exceeded")
                        row, result = run_strategy(model, allowed, targets, strategy, spec["numerics"]["guard"])
                        row.update(seed=seed, graph_id=graph_id, workload=workload["name"],
                                   strategy=strategy, repetition=repetition)
                        check_result(row, result, reference, nominal, targets, spec)
                        if repetition >= 0:
                            case_records.append(row)
                        if repetition == 0:
                            saved[f"{workload['name']}/{strategy}"] = result
            if state_fingerprint(model) != before:
                raise RuntimeError("replay mutated frozen model")
            case_path = out / "cases" / f"{graph_id}.json"
            write_exclusive_json(case_path, dict(seed=seed, graph_id=graph_id, records=case_records))
            evidence_path = out / "cases" / f"{graph_id}.pt"
            with evidence_path.open("xb") as stream:
                torch.save(saved, stream)
            evidence_index[str(evidence_path.relative_to(ROOT))] = sha256(evidence_path)
            records.extend(case_records)
            print(json.dumps(dict(completed_case=graph_id, records=len(case_records),
                                  elapsed_seconds=round(time.monotonic() - started, 2))), flush=True)
        verify_hashes(manifest["frozen_input_sha256"])
        verify_hashes(manifest["source_sha256"])
        write_exclusive_json(out / "runs.json", dict(schema_version=1,
            protocol_sha256=manifest["protocol_sha256"], input_audit_sha256=manifest["input_audit_sha256"],
            freeze_manifest_sha256=sha256(out / "freeze_manifest.json"),
            frozen_input_sha256=manifest["frozen_input_sha256"], inputs_unchanged_after=True,
            implementation_unchanged_after=True, records=records, evidence_sha256=evidence_index,
            elapsed_seconds=time.monotonic() - started))
    except Exception as error:
        write_exclusive_json(out / "failure.json", dict(type=type(error).__name__, reason=str(error),
            completed_records=len(records), elapsed_seconds=time.monotonic() - started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.phase == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
