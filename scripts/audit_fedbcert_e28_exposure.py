#!/usr/bin/env python3
"""Read-only E18 checkpoint audit; writes exclusively to a new E28a namespace."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from sklearn.metrics import f1_score
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    build_boundary_ledger_views,
    build_federated_clients_from_ledgers,
    canonical_undirected,
    certificate_statistics,
    cone_pairwise_certificate_statistics,
    induced_edges,
    missing_degrees,
    row_normalized_shift,
)
from openfgl.flcore.fedbcert.federated import (
    _completion_logits_with_aligned_nodewise_encoding,
    _logit_audit_tolerance,
)
from openfgl.flcore.fedsla_xn.data import (
    load_cross_network_dataset,
    stratified_four_way_masks,
)

CONFIG = ROOT / "configs/fedbcert_e28a_exposure_protocol.json"
GROUPS = ("all", "exposed", "direct", "indirect", "non_exposed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.dtype, array.shape)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_new_text(path: Path, text: str) -> None:
    """Never replace a pre-existing artifact with different content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite existing artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_json(path: Path, value: dict | list) -> None:
    write_new_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to emit an empty table without a schema")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_new_text(path, stream.getvalue())


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def exposure_masks(
    shift: torch.Tensor, exact_missing: torch.Tensor, maximum_power: int
) -> dict[str, torch.Tensor]:
    """A boundary *row* first affects its owner at power one, not zero."""
    if maximum_power < 0 or int(maximum_power) != maximum_power:
        raise ValueError("maximum power must be a nonnegative integer")
    if shift.layout != torch.sparse_coo or shift.ndim != 2 or shift.size(0) != shift.size(1):
        raise ValueError("exposure requires a square sparse COO local shift")
    if exact_missing.ndim != 1 or exact_missing.numel() != shift.size(0):
        raise ValueError("counts and shift must be node aligned")
    counts = exact_missing.to(device=shift.device, dtype=torch.float64)
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("exact counts must be finite and nonnegative")
    if bool((counts != counts.floor()).any()):
        raise ValueError("exact boundary counts must be integers")
    if bool((shift.coalesce().values() < 0).any()):
        raise ValueError("exposure propagation requires nonnegative weights")
    boundary = counts > 0
    exposed = torch.zeros_like(boundary)
    for _ in range(maximum_power):
        reachable = torch.sparse.mm(shift, exposed.to(shift.dtype)[:, None]).reshape(-1) > 0
        exposed = boundary | reachable
    return {
        "all": torch.ones_like(boundary),
        "exposed": exposed,
        "direct": boundary & exposed,
        "indirect": exposed & ~boundary,
        "non_exposed": ~exposed,
    }


def subgroup_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    certified: torch.Tensor,
    mask: torch.Tensor,
    *,
    completed_logits: torch.Tensor | None = None,
    softmax_change_threshold: float = 0.05,
) -> dict:
    prediction = logits.argmax(dim=1)
    correct = prediction == labels
    count = int(mask.sum().item())
    certified_count = int((mask & certified).sum().item())
    correct_certified = int((mask & certified & correct).sum().item())
    result = {
        "node_count": count,
        "correct_count": int((mask & correct).sum().item()),
        "certified_count": certified_count,
        "correct_certified_count": correct_certified,
        "wrong_certified_count": certified_count - correct_certified,
        "local_macro_f1": None,
        "local_accuracy": safe_div(int((mask & correct).sum().item()), count),
        "certified_ratio": safe_div(certified_count, count),
        "certified_accuracy": safe_div(correct_certified, count),
        "accuracy_given_certified": safe_div(correct_certified, certified_count),
        "label_counts": json.dumps(torch.bincount(labels[mask], minlength=logits.size(1)).tolist()),
        "completion_flip_count": 0,
        "certified_completion_flip_count": 0,
        "mean_absolute_completion_logit_change": None,
        "mean_maximum_softmax_change": None,
        "large_softmax_change_count": 0,
    }
    if count:
        result["local_macro_f1"] = float(f1_score(
            labels[mask].detach().cpu().numpy(),
            prediction[mask].detach().cpu().numpy(), average="macro", zero_division=0,
        ))
    if completed_logits is not None:
        flips = prediction != completed_logits.argmax(dim=1)
        result["completion_flip_count"] = int((mask & flips).sum().item())
        result["certified_completion_flip_count"] = int((mask & certified & flips).sum().item())
        if count:
            difference = (completed_logits - logits).abs()
            probability_change = (completed_logits.softmax(1) - logits.softmax(1)).abs().amax(1)
            result["mean_absolute_completion_logit_change"] = float(difference[mask].mean().item())
            result["mean_maximum_softmax_change"] = float(probability_change[mask].mean().item())
            result["large_softmax_change_count"] = int(
                (mask & (probability_change >= softmax_change_threshold)).sum().item()
            )
    return result


def load_saved_membership(path: Path, node_count: int, client_count: int) -> np.ndarray:
    rows = read_csv(path)
    membership = np.full(node_count, -1, dtype=np.int64)
    for row in rows:
        node, client = int(row["node_id"]), int(row["client_id"])
        if not 0 <= node < node_count or membership[node] != -1:
            raise ValueError("saved membership contains an out-of-range or duplicate node")
        if not 0 <= client < client_count:
            raise ValueError("saved membership contains an invalid client")
        membership[node] = client
    if len(rows) != node_count or np.any(membership < 0):
        raise ValueError("saved membership must cover every node exactly once")
    if set(membership.tolist()) != set(range(client_count)):
        raise ValueError("saved membership contains an empty client")
    return membership


def source_units(config: dict) -> list[dict]:
    units = []
    for dataset in config["datasets"]:
        for clients in config["num_clients"]:
            for partition in config["partitions"]:
                for seed in config["seeds"]:
                    identity = f"{dataset}_{partition}_c{clients}_seed{seed}"
                    run = ROOT / config["source_root"] / "confirmation" / config["selected_variant"] / identity
                    units.append({
                        "identity": identity, "dataset": dataset,
                        "num_clients": clients, "partition": partition, "seed": seed,
                        "run": run,
                    })
    if len(units) != config["expected_units"]:
        raise ValueError("configured matrix does not match expected unit count")
    return units


def dataset_files(config: dict, dataset: str) -> list[Path]:
    base = ROOT / config["data_root"] / dataset
    if dataset in {"Cora", "CiteSeer", "PubMed"}:
        return [base / "linqs_data.pt"]
    if dataset not in {"Computers", "Photo", "CS"}:
        raise ValueError("this audit only supports the registered six E18 datasets")
    base = base / dataset / "processed"
    # Requiring all cache metadata avoids invoking a dataset processing/download path.
    return [base / name for name in ("data.pt", "pre_filter.pt", "pre_transform.pt")]


def model_from_arguments(arguments: dict, input_dim: int, class_count: int) -> PolynomialSpectralClassifier:
    return PolynomialSpectralClassifier(
        input_dim=input_dim, hidden_dim=int(arguments["hidden_dim"]),
        num_classes=class_count, order_count=int(arguments["order_count"]),
        dropout=float(arguments["dropout"]), encoder_type=arguments["encoder_type"],
        encoder_depth=int(arguments["encoder_depth"]),
        encoder_residual_scale=float(arguments["encoder_residual_scale"]),
        utility_diffusion_hops=int(arguments.get("utility_diffusion_hops", 1)),
        utility_diffusion_scale=float(arguments.get("utility_diffusion_scale", 0.25)),
        spectral_fusion_scale=float(arguments["spectral_fusion_scale"]),
        certificate_margin_mode="pairwise_difference",
    )


def check_output_root(path: Path) -> Path:
    resolved = path.resolve()
    parent = (ROOT / "results_boundary_certified").resolve()
    if resolved.parent != parent or not resolved.name.startswith("boundary_certified_spectral_e28a_exposure"):
        raise ValueError("output must be a dedicated E28a directory immediately below results_boundary_certified")
    return resolved


def inventory(config: dict, config_path: Path) -> dict:
    units = source_units(config)
    policy = read_json(ROOT / config["source_policy"])
    source_config = read_json(ROOT / config["source_protocol"])
    confirmation = read_json(ROOT / config["source_confirmation"])
    if policy.get("config_sha256") != sha256(ROOT / config["source_protocol"]):
        raise RuntimeError("E18 protocol no longer matches the frozen policy hash")
    if policy.get("selected_variant") != config["selected_variant"] or not confirmation.get("complete"):
        raise RuntimeError("registered source predictor/confirmation is not the frozen E18 result")
    matrix = source_config["confirmation_matrix"]
    if any(config[key] != matrix[key] for key in ("num_clients", "partitions")):
        raise RuntimeError("E28a must include the complete E18 client/partition matrix")
    if config["datasets"] != source_config["datasets"] or config["seeds"] != matrix["seeds"]:
        raise RuntimeError("E28a cannot selectively omit source datasets or seeds")
    source_paths = [ROOT / config[key] for key in (
        "source_protocol", "source_policy", "source_confirmation", "source_overall_table",
    )]
    protected = [ROOT / path for path in config["protected_files"]]
    data_paths = [path for dataset in config["datasets"] for path in dataset_files(config, dataset)]
    units_manifest = []
    missing = []
    for unit in units:
        files = {name: unit["run"] / name for name in (
            "fedavg_selected_state.pt", "partition_membership.csv", "protocol.json",
            "federated_metrics.csv", "run_complete.json",
        )}
        record = {key: value for key, value in unit.items() if key != "run"}
        record["files"] = {}
        for name, path in files.items():
            relative = path.relative_to(ROOT).as_posix()
            record["files"][name] = {"path": relative, "sha256": sha256(path) if path.is_file() else None}
            if not path.is_file():
                missing.append(relative)
        units_manifest.append(record)
    paths = source_paths + protected + data_paths
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        hashes[relative] = sha256(path) if path.is_file() else None
        if not path.is_file():
            missing.append(relative)
    return {
        "protocol_sha256": sha256(config_path),
        "source_files": hashes, "units": units_manifest,
        "expected_units": config["expected_units"], "missing_files": missing,
        "audit_code_sha256": sha256(Path(__file__)),
        "analysis_code_sha256": sha256(ROOT / "scripts/analyze_fedbcert_e28_exposure.py"),
        "no_training": True, "retrospective": True,
        "data_provenance": config["claims"]["source_data_provenance"],
    }


def verify_hashes(files: dict) -> None:
    for relative, expected in files.items():
        if expected is None or sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed or is missing: {relative}")


def audit_unit(unit: dict, config: dict, data, device: torch.device, output: Path, manifest: dict) -> None:
    unit_manifest = next(row for row in manifest["units"] if row["identity"] == unit["identity"])
    file_hashes = {record["path"]: record["sha256"] for record in unit_manifest["files"].values()}
    verify_hashes(file_hashes)
    signature = {
        "protocol_sha256": manifest["protocol_sha256"],
        "audit_code_sha256": manifest["audit_code_sha256"],
        "source_file_hashes": file_hashes, "device": str(device),
    }
    target = output / "units" / unit["identity"]
    completed_path = target / "run_complete.json"
    if completed_path.exists():
        previous = read_json(completed_path)
        if previous["signature"] != signature:
            raise RuntimeError("existing result has a different source/code/config/device; use a new E28a output namespace")
        verify_hashes(previous["output_hashes"])
        print(f"[SKIP] {unit['identity']}", flush=True)
        return
    started = time.perf_counter()
    source_protocol = read_json(unit["run"] / "protocol.json")
    arguments = source_protocol["arguments"]
    source_rows = read_csv(unit["run"] / "federated_metrics.csv")
    if len(source_rows) != 1 or source_rows[0].get("method") != "fedavg":
        raise RuntimeError("E18 source must contain exactly one FedAvg row")
    source_row = source_rows[0]
    if read_json(unit["run"] / "run_complete.json").get("status") != "complete":
        raise RuntimeError("source runner was not complete")
    for key in ("dataset", "partition", "num_clients", "seed"):
        if str(arguments[key]) != str(unit[key]) or str(source_row[key]) != str(unit[key]):
            raise RuntimeError(f"source identity mismatch: {key}")
    checks = {
        "no_test_selection": source_protocol.get("test_labels_used_for_selection") is False,
        "frozen_encoder": arguments["encoder_type"] == "dual_feature_spectral",
        "frozen_alpha": float(arguments["spectral_fusion_scale"]) == 0.25,
        "fixed_order": int(arguments["order_count"]) - 1 == config["maximum_propagation_power"],
        "retained_mass": arguments["uncertainty_recurrence"] == "retained_mass",
        "pairwise_margin": arguments["certificate_margin_mode"] == "pairwise_difference",
        "standard_training": float(source_row["robust_lambda"]) == 0.0,
        "cache_root": Path(arguments["data_root"]).as_posix().removeprefix("./") == config["data_root"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"source contract mismatch: {checks}")

    features = data.x.float()
    labels = data.y.reshape(-1).long()
    node_count = int(features.size(0))
    class_count = int(labels.max().item()) + 1
    membership = load_saved_membership(unit["run"] / "partition_membership.csv", node_count, unit["num_clients"])
    full_edges = canonical_undirected(data.edge_index, node_count)
    local_edges = induced_edges(full_edges, membership)
    missing = missing_degrees(full_edges, local_edges, node_count)
    if int(full_edges.size(1) // 2) != int(source_protocol["undirected_edges"]):
        raise RuntimeError("cached graph edge count differs from the frozen run")
    if int(missing.sum().item()) != 2 * int(source_protocol["cut_edges"]):
        raise RuntimeError("saved partition/cached graph cut count differs from the frozen run")
    masks = stratified_four_way_masks(
        labels, train_ratio=arguments["train_ratio"], calibration_ratio=arguments["calibration_ratio"],
        val_ratio=arguments["val_ratio"], test_ratio=arguments["test_ratio"], seed=unit["seed"],
    )
    ledgers = build_boundary_ledger_views(full_edges, membership, missing)
    clients = build_federated_clients_from_ledgers(
        ledgers, features, labels, masks[0], masks[2], masks[3],
        int(arguments["order_count"]), device, uncertainty_recurrence="retained_mass",
    )
    full_shift, _ = row_normalized_shift(full_edges, node_count, device=device)
    state = torch.load(unit["run"] / "fedavg_selected_state.pt", map_location="cpu", weights_only=True)
    model = model_from_arguments(arguments, features.size(1), class_count).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    frozen_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    logits = torch.empty((node_count, class_count), device=device)
    groups = {name: torch.zeros(node_count, dtype=torch.bool, device=device) for name in GROUPS}
    certificates = {name: torch.zeros(node_count, dtype=torch.bool, device=device) for name in config["certificates"]}
    slacks = {name: torch.empty(node_count, device=device) for name in config["certificates"]}
    radius = torch.empty(node_count, device=device)
    bounds = torch.empty_like(logits)
    with torch.inference_mode():
        for client in clients:
            local_logits = model(client.features, client.shift)
            nominal_snapshot = local_logits.clone()
            local_groups = exposure_masks(client.shift, client.missing_degree, config["maximum_propagation_power"])
            symmetric = certificate_statistics(model, local_logits, client.uncertainty_basis)
            raw = cone_pairwise_certificate_statistics(
                model, client.features, client.shift, client.uncertainty_basis[1],
                local_logits, client.uncertainty_basis, count_semantics="upper_bound", combine_symmetric=False,
            )
            combined = replace(raw, certified=raw.certified | symmetric.certified,
                               worst_slack=torch.maximum(raw.worst_slack, symmetric.worst_slack))
            # Check mutation of the shared nominal tensor, not a second GPU
            # forward pass whose reduction order can differ by a few ULPs.
            if not torch.equal(local_logits, nominal_snapshot):
                raise RuntimeError("certificate evaluation changed nominal logits")
            for name, statistics in (("symmetric", symmetric), ("cone_raw", raw), ("combined", combined)):
                certificates[name][client.global_nodes] = statistics.certified
                slacks[name][client.global_nodes] = statistics.worst_slack
            for name in GROUPS:
                groups[name][client.global_nodes] = local_groups[name]
            logits[client.global_nodes] = local_logits
            radius[client.global_nodes] = symmetric.representation_radius
            bounds[client.global_nodes] = symmetric.class_logit_bounds
        completed_logits = _completion_logits_with_aligned_nodewise_encoding(model, clients, full_shift, logits)
    for key, value in model.state_dict().items():
        if not torch.equal(frozen_state[key], value.detach().cpu()):
            raise RuntimeError("audit mutated a frozen model tensor")
    if bool((radius[groups["non_exposed"]] != 0).any()):
        raise RuntimeError("non-exposed nodes have nonzero structural radius: exposure off-by-one or count contract error")
    test = masks[3].to(device)
    truth = labels.to(device)
    prediction = logits.argmax(1)
    complete_margin = completed_logits.gather(1, prediction[:, None]) - completed_logits
    complete_margin.scatter_(1, prediction[:, None], float("inf"))
    actual_minimum_margin = complete_margin.amin(1)
    tolerance = _logit_audit_tolerance(logits, completed_logits)
    logit_violations = test & ((completed_logits - logits).abs() > bounds + tolerance).any(1)
    pair_tolerance = 2 * tolerance.amax(1)
    counterexamples = []
    for name in config["certificates"]:
        flips = test & certificates[name] & (prediction != completed_logits.argmax(1))
        lower_violations = test & (slacks[name] > actual_minimum_margin + pair_tolerance)
        for node in torch.where(flips | lower_violations | logit_violations)[0].tolist():
            counterexamples.append({
                "node_id": node, "certificate": name, "nominal_class": int(prediction[node]),
                "completion_class": int(completed_logits[node].argmax()),
                "certified_flip": bool(flips[node]), "pairwise_bound_violation": bool(lower_violations[node]),
                "logit_bound_violation": bool(logit_violations[node]),
                "certified_gap_lower": float(slacks[name][node]),
                "completion_gap": float(actual_minimum_margin[node]),
            })
    if counterexamples:
        write_json(target / "counterexamples.json", counterexamples)
        raise RuntimeError(f"reliability counterexample saved for {unit['identity']}")

    identity_fields = {key: value for key, value in unit.items() if key != "run"}
    rows = []
    for group in GROUPS:
        mask = test & groups[group]
        for name in config["certificates"]:
            statistics = subgroup_metrics(
                logits, truth, certificates[name], mask, completed_logits=completed_logits,
                softmax_change_threshold=config["diagnostic_softmax_change_threshold"],
            )
            rows.append({
                **identity_fields, "group": group, "certificate": name, **statistics,
                "mean_representation_radius": float(radius[mask].mean()) if bool(mask.any()) else None,
                "zero_radius_count": int((mask & (radius == 0)).sum()),
                "minimum_gap_lower": float(slacks[name][mask].min()) if bool(mask.any()) else None,
            })
    keyed = {(row["group"], row["certificate"]): row for row in rows}
    source_checks = []
    for name, prefix in (("symmetric", "tight_pairwise"), ("cone_raw", "cone_raw"), ("combined", "cone_hybrid")):
        row = keyed[("all", name)]
        for metric in ("certified_ratio", "certified_accuracy"):
            field = f"test_bound_{prefix}_{metric}"
            source_checks.append({"field": field, "reconstructed": row[metric], "source": float(source_row[field])})
    all_row = keyed[("all", "symmetric")]
    for metric in ("local_macro_f1", "local_accuracy", "completion_flip_fraction"):
        reconstructed = (safe_div(all_row["completion_flip_count"], all_row["node_count"])
                         if metric == "completion_flip_fraction" else all_row[metric])
        source_checks.append({"field": f"test_{metric}", "reconstructed": reconstructed,
                              "source": float(source_row[f"test_{metric}"])})
    source_checks.append({"field": "test_boundary_node_count", "reconstructed": keyed[("direct", "symmetric")]["node_count"],
                          "source": int(source_row["test_boundary_node_count"])})
    source_checks.append({"field": "test_mean_absolute_completion_logit_change",
                          "reconstructed": all_row["mean_absolute_completion_logit_change"],
                          "source": float(source_row["test_mean_absolute_completion_logit_change"])})
    for check in source_checks:
        tolerance_value = config["source_metric_tolerance"][
            "completion_logit_change_absolute" if "logit_change" in check["field"] else "classification_and_coverage_absolute"
        ]
        check["absolute_difference"] = abs(check["reconstructed"] - check["source"])
        check["passed"] = check["absolute_difference"] <= tolerance_value
    if not all(check["passed"] for check in source_checks):
        write_json(target / "source_mismatch.json", source_checks)
        raise RuntimeError(f"frozen source metrics not reproduced: {unit['identity']}")
    verify_hashes(file_hashes)
    write_csv(target / "exposure_metrics.csv", rows)
    data_fingerprints = {
        "membership": tensor_sha256(torch.from_numpy(membership)),
        **{name: tensor_sha256(mask) for name, mask in zip(("train", "calibration", "validation", "test"), masks)},
        "exact_missing": tensor_sha256(missing), "nominal_logits": tensor_sha256(logits),
        "completion_logits": tensor_sha256(completed_logits),
    }
    write_json(completed_path, {
        "status": "complete", "signature": signature,
        "output_hashes": {(target / "exposure_metrics.csv").relative_to(ROOT).as_posix(): sha256(target / "exposure_metrics.csv")},
        "source_metric_checks": source_checks, "reconstructed_tensor_sha256": data_fingerprints,
        "no_training": True, "model_unchanged": True, "logits_unchanged_by_certificates": True,
        "source_metrics_reproduced": True, "certificate_violations": 0,
        "maximum_power": config["maximum_propagation_power"],
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"[DONE] {unit['identity']} exposed={keyed[('exposed', 'combined')]['node_count']}/{all_row['node_count']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--limit", type=int, help="engineering smoke only; does not change the registered full matrix")
    parser.add_argument("--output-root", type=Path, help="new E28a namespace; never overwrites old results")
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads and smoke limit must be positive")
    config = read_json(args.config)
    output = check_output_root(args.output_root or ROOT / config["result_root"])
    manifest = inventory(config, args.config)
    freeze_path = output / "freeze_manifest.json"
    if freeze_path.exists():
        if read_json(freeze_path)["inventory"] != manifest:
            raise RuntimeError("sources or audit code changed since inventory; use a new E28a output namespace")
    else:
        write_json(freeze_path, {"created_utc": datetime.now(timezone.utc).isoformat(), "inventory": manifest})
    print(json.dumps({"expected_units": manifest["expected_units"], "missing_files": manifest["missing_files"], "no_training": True}), flush=True)
    if manifest["missing_files"]:
        raise SystemExit("source artifacts missing; recorded in freeze_manifest.json; no retraining performed")
    if args.inventory_only:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent device fallback")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    completed = 0
    cached_dataset = None
    data = None
    for unit in source_units(config):
        if args.limit is not None and completed >= args.limit:
            break
        if unit["dataset"] != cached_dataset:
            data = load_cross_network_dataset(unit["dataset"], str(ROOT / config["data_root"]))
            cached_dataset = unit["dataset"]
        audit_unit(unit, config, data, device, output, manifest)
        completed += 1
    verify_hashes(manifest["source_files"])
    for unit in manifest["units"]:
        verify_hashes({record["path"]: record["sha256"] for record in unit["files"].values()})
    write_json(output / ("smoke_summary.json" if args.limit is not None else "execution_summary.json"), {
        "processed_units": completed, "expected_units": config["expected_units"],
        "full_matrix_complete": completed == config["expected_units"],
        "all_registered_sources_unchanged": True, "no_training": True,
        "device": str(device), "torch_version": torch.__version__,
    })


if __name__ == "__main__":
    main()
