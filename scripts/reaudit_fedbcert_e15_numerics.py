#!/usr/bin/env python3
"""Read-only numerical re-audit of E15 logit-bound diagnostics.

E15 evaluates one frozen model once on concatenated client batches and once
on the full graph.  Float32 GEMM/normalization can associate reductions
differently across those batch shapes, including for the graph-invariant
feature expert whose analytic boundary radius is exactly zero.  This script
reconstructs every raw flagged state, reproduces the historical count, and
applies the dtype-aware audit tolerance without retraining or rewriting runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    build_boundary_ledger_views,
    build_federated_clients_from_ledgers,
    canonical_undirected,
    induced_edges,
    missing_degrees,
    prediction_certificate_statistics,
    retained_mass_uncertainty_basis,
    row_normalized_shift,
)
from openfgl.flcore.fedbcert.federated import (
    _completion_logits_with_aligned_nodewise_encoding,
    _logit_audit_tolerance,
)
from openfgl.flcore.fedbcert.data import (
    load_cross_network_dataset,
    stratified_four_way_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="results_boundary_certified/boundary_certified_spectral_e15",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--only", default="", help="optional run-directory suffix for debugging"
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def membership_from_csv(path: Path, node_count: int) -> np.ndarray:
    membership = np.full(node_count, -1, dtype=np.int64)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            membership[int(row["node_id"])] = int(row["client_id"])
    if bool((membership < 0).any()):
        raise RuntimeError(f"incomplete membership: {path}")
    return membership


def masks_for_data(data, labels: torch.Tensor, arguments: dict):
    if all(
        hasattr(data, attribute)
        for attribute in (
            "official_train_mask",
            "official_validation_mask",
            "official_test_mask",
        )
    ):
        official_train = data.official_train_mask.bool()
        indices = torch.where(official_train)[0].cpu().numpy()
        train_indices, calibration_indices = train_test_split(
            indices,
            test_size=1.0 / 6.0,
            random_state=int(arguments["seed"]),
            stratify=labels[indices].cpu().numpy(),
        )
        train = torch.zeros_like(official_train, dtype=torch.bool)
        calibration = torch.zeros_like(train)
        train[torch.as_tensor(train_indices, dtype=torch.long)] = True
        calibration[torch.as_tensor(calibration_indices, dtype=torch.long)] = True
        return (
            train,
            calibration,
            data.official_validation_mask.bool().clone(),
            data.official_test_mask.bool().clone(),
        )
    return stratified_four_way_masks(
        labels,
        train_ratio=float(arguments["train_ratio"]),
        calibration_ratio=float(arguments["calibration_ratio"]),
        val_ratio=float(arguments["val_ratio"]),
        test_ratio=float(arguments["test_ratio"]),
        seed=int(arguments["seed"]),
    )


def high_precision_spectral_audit(
    model: PolynomialSpectralClassifier,
    clients,
    full_shift: torch.Tensor,
    test_mask: torch.Tensor,
    node_count: int,
    num_classes: int,
) -> dict:
    """Re-evaluate only the graph-sensitive difference in float64.

    The graph-invariant expert cancels algebraically and is never recomputed.
    Local and completion propagation share the exact same locally batched
    bounded encodings.  This audit therefore checks the certificate theorem's
    linear spectral path independently of the float32 end-to-end diagnostic.
    """
    alpha = float(model.spectral_fusion_scale)
    if alpha == 0.0:
        return {
            "violations": 0,
            "max_deviation": 0.0,
            "max_excess_over_bound": 0.0,
            "tolerance": 0.0,
        }
    device = next(model.parameters()).device
    hidden_dim = int(model.classifier.in_features)
    theta = model.theta.detach().to(dtype=torch.float64)
    weight = model.classifier.weight.detach().to(dtype=torch.float64)
    global_hidden = torch.empty(
        (node_count, hidden_dim), device=device, dtype=torch.float64
    )
    local_spectral = torch.empty(
        (node_count, num_classes), device=device, dtype=torch.float64
    )
    class_bounds = torch.empty_like(local_spectral)

    def filtered(hidden: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        current = hidden
        output = theta[0] * current
        for order in range(1, int(model.order_count)):
            current = torch.sparse.mm(shift, current)
            output = output + theta[order] * current
        return output

    with torch.no_grad():
        for client in clients:
            nodes = client.global_nodes
            hidden = model.bounded_encoding(client.features).to(torch.float64)
            global_hidden[nodes] = hidden
            local_shift = client.shift.to(dtype=torch.float64)
            local_spectral[nodes] = filtered(hidden, local_shift) @ weight.T
            row_budget = client.uncertainty_basis[1].to(torch.float64)
            basis = retained_mass_uncertainty_basis(
                local_shift, row_budget, int(model.order_count)
            )
            radius = sum(
                theta[order].abs() * basis[order]
                for order in range(1, int(model.order_count))
            )
            class_bounds[nodes] = (
                alpha * radius[:, None] * torch.linalg.vector_norm(weight, dim=1)
            )
        completion_spectral = filtered(
            global_hidden, full_shift.to(dtype=torch.float64)
        ) @ weight.T

    deviation = alpha * (completion_spectral - local_spectral).abs()
    scale = torch.maximum(completion_spectral.abs(), local_spectral.abs())
    # Fixed before inspecting float64 residuals: 128 double-precision ULPs.
    tolerance = 128.0 * torch.finfo(torch.float64).eps * (1.0 + alpha * scale)
    active = test_mask.to(device)[:, None].expand_as(deviation)
    violations = active & (deviation > class_bounds + tolerance)
    excess = deviation - class_bounds
    return {
        "violations": int(violations.sum().item()),
        "max_deviation": float(deviation[active].max().item()),
        "max_excess_over_bound": float(excess[active].max().item()),
        "tolerance": float(tolerance[active].max().item()),
    }


def reaudit_run(run: Path, raw_reported: int, device: torch.device) -> dict:
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    arguments = protocol["arguments"]
    data = load_cross_network_dataset(arguments["dataset"], arguments["data_root"])
    features = data.x.float()
    labels = data.y.reshape(-1).long()
    node_count = int(data.num_nodes)
    full_edges = canonical_undirected(data.edge_index, node_count)
    membership = membership_from_csv(run / "partition_membership.csv", node_count)
    local_union = induced_edges(full_edges, membership)
    missing = missing_degrees(full_edges, local_union, node_count)
    ledgers = build_boundary_ledger_views(full_edges, membership, missing)
    train, _, validation, test = masks_for_data(data, labels, arguments)
    clients = build_federated_clients_from_ledgers(
        ledgers,
        features,
        labels,
        train,
        validation,
        test,
        int(arguments["order_count"]),
        device,
        str(arguments["uncertainty_recurrence"]),
    )
    full_shift, _ = row_normalized_shift(full_edges, node_count, device=device)
    state = torch.load(
        run / "fedavg_selected_state.pt", map_location="cpu", weights_only=True
    )
    model = PolynomialSpectralClassifier(
        input_dim=int(features.size(1)),
        hidden_dim=int(arguments["hidden_dim"]),
        num_classes=int(labels.max().item()) + 1,
        order_count=int(arguments["order_count"]),
        dropout=float(arguments["dropout"]),
        encoder_type=str(arguments["encoder_type"]),
        encoder_depth=int(arguments["encoder_depth"]),
        encoder_residual_scale=float(arguments["encoder_residual_scale"]),
        utility_diffusion_hops=int(arguments.get("utility_diffusion_hops", 1)),
        utility_diffusion_scale=float(arguments.get("utility_diffusion_scale", 0.25)),
        spectral_fusion_scale=float(arguments["spectral_fusion_scale"]),
        certificate_margin_mode=str(arguments["certificate_margin_mode"]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    num_classes = int(labels.max().item()) + 1
    local_logits = torch.empty((node_count, num_classes), device=device)
    class_bounds = torch.empty_like(local_logits)
    certified = torch.zeros(node_count, dtype=torch.bool, device=device)
    with torch.no_grad():
        for client in clients:
            prediction = prediction_certificate_statistics(
                model, client.features, client.shift, client.uncertainty_basis
            )
            local_logits[client.global_nodes] = prediction.logits
            class_bounds[client.global_nodes] = (
                prediction.certificate.class_logit_bounds
            )
            certified[client.global_nodes] = prediction.certificate.certified
        completion_logits = model(features.to(device), full_shift)
        aligned_completion_logits = (
            _completion_logits_with_aligned_nodewise_encoding(
                model, clients, full_shift, local_logits
            )
        )
        high_precision = high_precision_spectral_audit(
            model,
            clients,
            full_shift,
            test,
            node_count,
            num_classes,
        )

    test = test.to(device)
    deviation = (completion_logits - local_logits).abs()
    scale = torch.maximum(completion_logits.abs(), local_logits.abs())
    historical_tolerance = 1e-6 + 1e-5 * scale
    revised_tolerance = _logit_audit_tolerance(local_logits, completion_logits)
    active = test[:, None].expand_as(deviation)
    historical_bad = active & (deviation > class_bounds + historical_tolerance)
    revised_bad = active & (deviation > class_bounds + revised_tolerance)
    aligned_deviation = (aligned_completion_logits - local_logits).abs()
    aligned_bad_without_tolerance = active & (aligned_deviation > class_bounds)
    prediction = local_logits.argmax(dim=1)
    completion_prediction = completion_logits.argmax(dim=1)
    certified_flips = int(
        (certified & test & (prediction != completion_prediction)).sum().item()
    )
    old_count = int(historical_bad.sum().item())
    if old_count:
        old_deviation = deviation[historical_bad]
        old_bounds = class_bounds[historical_bad]
        old_excess = old_deviation - old_bounds
        old_tolerance_excess = old_excess - historical_tolerance[historical_bad]
        max_deviation = float(old_deviation.max().item())
        max_bound_excess = float(old_excess.max().item())
        max_historical_tolerance_excess = float(old_tolerance_excess.max().item())
        zero_bound_count = int((old_bounds == 0).sum().item())
    else:
        max_deviation = 0.0
        max_bound_excess = 0.0
        max_historical_tolerance_excess = 0.0
        zero_bound_count = 0
    return {
        "variant": run.parent.name,
        "dataset": arguments["dataset"],
        "partition": arguments["partition"],
        "num_clients": int(arguments["num_clients"]),
        "seed": int(arguments["seed"]),
        "raw_reported_violations": int(raw_reported),
        "reproduced_historical_violations": old_count,
        "revised_dtype_aware_violations": int(revised_bad.sum().item()),
        "aligned_no_tolerance_violations": int(
            aligned_bad_without_tolerance.sum().item()
        ),
        "float64_spectral_violations": high_precision["violations"],
        "historical_zero_bound_violations": zero_bound_count,
        "certified_prediction_flips": certified_flips,
        "max_flagged_raw_deviation": max_deviation,
        "max_flagged_excess_over_bound": max_bound_excess,
        "max_excess_over_historical_tolerance": max_historical_tolerance_excess,
        "max_aligned_deviation": float(aligned_deviation[active].max().item()),
        "max_aligned_vs_original_completion": float(
            (aligned_completion_logits - completion_logits).abs().max().item()
        ),
        "float64_max_deviation": high_precision["max_deviation"],
        "float64_max_excess_over_bound": high_precision[
            "max_excess_over_bound"
        ],
        "float64_max_audit_tolerance": high_precision["tolerance"],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    analysis = root / "analysis"
    raw_path = analysis / "paper_data_e15_all_metrics.csv"
    if not raw_path.exists():
        raise SystemExit(f"missing E15 analysis data: {raw_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda:0" if use_cuda else "cpu")
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    flagged = [
        row for row in raw_rows if int(row["test_logit_bound_violations"]) > 0
    ]
    if args.only:
        flagged = [
            row
            for row in flagged
            if (
                f"{row['variant']}/{row['dataset']}_{row['partition']}_"
                f"c{row['num_clients']}_seed{row['seed']}"
            ) == args.only
        ]
    rows: list[dict] = []
    for row in flagged:
        run = (
            root
            / "confirmation"
            / row["variant"]
            / (
                f"{row['dataset']}_{row['partition']}_"
                f"c{row['num_clients']}_seed{row['seed']}"
            )
        )
        result = reaudit_run(
            run, int(row["test_logit_bound_violations"]), device
        )
        rows.append(result)
        print(
            f"[AUDIT] {run.name}: raw={result['raw_reported_violations']} "
            f"reproduced={result['reproduced_historical_violations']} "
            f"revised={result['revised_dtype_aware_violations']} "
            f"float64={result['float64_spectral_violations']}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_total = sum(int(row["test_logit_bound_violations"]) for row in raw_rows)
    certified_completion_total = sum(
        int(row["test_certified_completion_violations"]) for row in raw_rows
    )
    reproduced_total = sum(
        row["reproduced_historical_violations"] for row in rows
    )
    revised_total = sum(row["revised_dtype_aware_violations"] for row in rows)
    aligned_total = sum(row["aligned_no_tolerance_violations"] for row in rows)
    float64_total = sum(row["float64_spectral_violations"] for row in rows)
    flips_total = sum(row["certified_prediction_flips"] for row in rows)
    complete = (
        raw_total == reproduced_total
        and revised_total == 0
        and float64_total == 0
        and certified_completion_total == 0
        and flips_total == 0
    )
    summary = {
        "protocol": "FedBCert E15 read-only dtype-aware numerical re-audit",
        "complete": complete,
        "device": str(device),
        "flagged_frozen_states": len(rows),
        "raw_reported_logit_bound_violations": raw_total,
        "reproduced_historical_logit_bound_violations": reproduced_total,
        "revised_dtype_aware_logit_bound_violations": revised_total,
        "aligned_no_tolerance_logit_bound_violations": aligned_total,
        "float64_spectral_logit_bound_violations": float64_total,
        "certified_completion_violations": certified_completion_total,
        "certified_prediction_flips": flips_total,
        "all_historical_flags_had_zero_analytic_bound": all(
            row["historical_zero_bound_violations"]
            == row["reproduced_historical_violations"]
            for row in rows
        ),
        "max_flagged_raw_deviation": max(
            (row["max_flagged_raw_deviation"] for row in rows), default=0.0
        ),
        "max_aligned_deviation": max(
            (row["max_aligned_deviation"] for row in rows), default=0.0
        ),
        "max_aligned_vs_original_completion": max(
            (
                row["max_aligned_vs_original_completion"]
                for row in rows
            ),
            default=0.0,
        ),
        "float64_max_excess_over_bound": max(
            (row["float64_max_excess_over_bound"] for row in rows),
            default=0.0,
        ),
        "float64_max_audit_tolerance": max(
            (row["float64_max_audit_tolerance"] for row in rows),
            default=0.0,
        ),
        "tolerance_rule": (
            "max(1e-6 + 1e-5*scale, "
            "64*torch.finfo(dtype).eps*(1+scale))"
        ),
        "training_or_state_modified": False,
        "logits_predictions_radii_or_certified_nodes_modified": False,
        "primary_resolution": (
            "cancel the graph-invariant expert algebraically, reuse identical "
            "local-batch bounded encodings, and re-evaluate only the spectral "
            "path in float64 with a fixed 128-ULP double-precision audit"
        ),
    }
    if rows:
        write_csv(analysis / "paper_table_e15_numerical_reaudit.csv", rows)
    (analysis / "numerical_reaudit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit("E15 numerical re-audit did not pass")


if __name__ == "__main__":
    main()
