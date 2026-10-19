#!/usr/bin/env python3
"""Evaluate frozen FedBCert predictors under safe ledger granularities."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    build_boundary_ledger_views,
    build_federated_clients_from_ledgers,
    canonical_undirected,
    cone_pairwise_certificate_statistics,
    induced_edges,
    missing_degrees,
    retained_mass_uncertainty_basis,
    row_normalized_shift,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config", default="configs/fedbcert_e27_ledger_granularity_protocol.json"
    )
    value.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    value.add_argument("--force", action="store_true")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fixed_width_bits(maximum_code: int) -> int:
    return int(math.ceil(math.log2(max(int(maximum_code) + 1, 2))))


def model_from_arguments(arguments: dict, bundle: dict, device: torch.device):
    model = PolynomialSpectralClassifier(
        input_dim=int(bundle["x"].size(1)),
        hidden_dim=int(arguments["hidden_dim"]),
        num_classes=int(bundle["y"].max().item()) + 1,
        order_count=int(arguments["order_count"]),
        dropout=float(arguments["dropout"]),
        encoder_type=str(arguments["encoder_type"]),
        encoder_depth=int(arguments["encoder_depth"]),
        encoder_residual_scale=float(arguments["encoder_residual_scale"]),
        utility_diffusion_hops=int(arguments.get("utility_diffusion_hops", 1)),
        utility_diffusion_scale=float(arguments.get("utility_diffusion_scale", 0.25)),
        spectral_fusion_scale=arguments.get("spectral_fusion_scale"),
        certificate_margin_mode="pairwise_difference",
    ).to(device)
    return model


def represented_count(
    mode: str, exact: torch.Tensor, client_maximum: float
) -> tuple[torch.Tensor | None, torch.Tensor, int]:
    """Return represented counts, row budgets, and local ledger bits."""
    if mode == "universal":
        return None, torch.full_like(exact, 2.0), 0
    if mode == "presence":
        budget = 2.0 * (exact > 0).to(exact.dtype)
        return None, budget, int(exact.numel())
    if mode == "client_max":
        reported = torch.full_like(exact, float(client_maximum))
        bits = fixed_width_bits(int(client_maximum))
        return reported, None, bits
    if mode == "exact":
        reported = exact.clone()
        width = fixed_width_bits(int(exact.max().item()))
        return reported, None, width * int(exact.numel())
    if mode.startswith("bin"):
        width = int(mode.removeprefix("bin"))
        reported = torch.ceil(exact / float(width)) * float(width)
        code = reported / float(width)
        bits = fixed_width_bits(int(code.max().item())) * int(exact.numel())
        return reported, None, bits
    raise ValueError(f"unsupported ledger granularity: {mode}")


def completion_logits(
    model: PolynomialSpectralClassifier,
    clients,
    full_shift: torch.Tensor,
    local_logits: torch.Tensor,
) -> torch.Tensor:
    """Use identical nodewise encodings and change only graph propagation."""
    if model.encoder_type == "feature_only_mlp":
        return local_logits.clone()
    hidden = torch.empty(
        (local_logits.size(0), model.classifier.in_features),
        dtype=local_logits.dtype,
        device=local_logits.device,
    )
    feature_logits = (
        torch.empty_like(local_logits) if model.feature_classifier is not None else None
    )
    for client in clients:
        hidden[client.global_nodes] = model.bounded_encoding(client.features)
        if feature_logits is not None:
            feature_logits[client.global_nodes] = model.feature_classifier(
                model.feature_encoding(client.features)
            )
    spectral = model.classifier(model.filter(hidden, full_shift))
    if feature_logits is None:
        return spectral
    return feature_logits + model.spectral_fusion_scale * spectral


def safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def main() -> None:
    args = parser().parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_digest = sha256(config_path)
    device = torch.device(
        "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    result_root = Path(config["result_root"])
    units_root = result_root / "units"
    bundle_root = Path(config["bundle_root"])
    e10c_root = Path(config["e10c_root"])
    modes = tuple(config["information_modes"])
    completed = 0

    for dataset in config["datasets"]:
        variant = config["frozen_variant_by_dataset"][dataset]
        for partition in config["partitions"]:
            for seed in config["seeds"]:
                identity = f"{dataset}_{partition}_c{config['num_clients']}_seed{seed}"
                output = units_root / identity
                completion = output / "run_complete.json"
                if completion.exists() and not args.force:
                    print(f"[SKIP] complete: {identity}")
                    completed += 1
                    continue
                output.mkdir(parents=True, exist_ok=True)

                bundle_dir = bundle_root / identity
                bundle_path = bundle_dir / "split_bundle.pt"
                manifest_path = bundle_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                bundle_digest = sha256(bundle_path)
                expected_identity = {
                    "dataset": dataset,
                    "partition": partition,
                    "num_clients": int(config["num_clients"]),
                    "seed": int(seed),
                }
                if bundle_digest != manifest.get("bundle_sha256"):
                    raise RuntimeError(f"bundle SHA-256 mismatch: {identity}")
                if any(manifest.get(key) != value for key, value in expected_identity.items()):
                    raise RuntimeError(f"bundle identity mismatch: {identity}")
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)

                e10c_dir = e10c_root / variant / identity
                protocol = json.loads((e10c_dir / "protocol.json").read_text(encoding="utf-8"))
                if protocol["immutable_split_bundle_audit"]["bundle_sha256"] != bundle_digest:
                    raise RuntimeError(f"E10c/bundle hash mismatch: {identity}")
                arguments = protocol["arguments"]
                state_path = e10c_dir / "fedbcert_ba_selected_state.pt"
                state = torch.load(state_path, map_location="cpu", weights_only=True)

                features = bundle["x"].float()
                labels = bundle["y"].reshape(-1).long()
                node_count = int(features.size(0))
                full_edges = canonical_undirected(bundle["edge_index"], node_count)
                membership = bundle["membership"].long().cpu().numpy()
                local_union = induced_edges(full_edges, membership)
                exact_missing = missing_degrees(full_edges, local_union, node_count)
                ledgers = build_boundary_ledger_views(full_edges, membership, exact_missing)
                order_count = int(arguments["order_count"])
                clients = build_federated_clients_from_ledgers(
                    ledgers,
                    features,
                    labels,
                    bundle["train_mask"].bool(),
                    bundle["validation_mask"].bool(),
                    bundle["test_mask"].bool(),
                    order_count,
                    device,
                    uncertainty_recurrence="retained_mass",
                )
                full_shift, _ = row_normalized_shift(full_edges, node_count, device=device)
                retained_degrees = []
                for ledger in ledgers:
                    _, degree = row_normalized_shift(
                        ledger.local_edge_index,
                        int(ledger.global_nodes.numel()),
                        device=device,
                    )
                    retained_degrees.append(degree)

                model = model_from_arguments(arguments, bundle, device)
                model.load_state_dict(state, strict=True)
                model.eval()
                test_global = bundle["test_mask"].bool().to(device)
                labels_global = labels.to(device)
                boundary_global = exact_missing.to(device) > 0
                rows: list[dict] = []
                reference_logits: torch.Tensor | None = None

                for mode in modes:
                    started = time.perf_counter()
                    logits_global = torch.empty(
                        (node_count, int(labels.max().item()) + 1), device=device
                    )
                    certified_global = torch.zeros(node_count, dtype=torch.bool, device=device)
                    radius_global = torch.zeros(node_count, dtype=torch.float32, device=device)
                    mode_bits = 0
                    max_overstatement = 0.0

                    with torch.no_grad():
                        for client, ledger, retained_degree in zip(
                            clients, ledgers, retained_degrees
                        ):
                            exact = ledger.missing_degree.to(device)
                            reported, direct_budget, local_bits = represented_count(
                                mode, exact, float(exact.max().item())
                            )
                            mode_bits += int(local_bits)
                            if reported is not None:
                                if bool((reported + 1e-12 < exact).any()):
                                    raise RuntimeError(f"unsafe count representation: {identity}:{mode}")
                                row_budget = boundary_uncertainty(retained_degree, reported)
                                max_overstatement = max(
                                    max_overstatement,
                                    float((reported - exact).max().item()),
                                )
                                basis = retained_mass_uncertainty_basis(
                                    client.shift, row_budget, order_count
                                )
                            else:
                                row_budget = direct_budget
                                if mode == "universal":
                                    basis = [torch.zeros_like(row_budget)] + [
                                        torch.full_like(row_budget, 2.0)
                                        for _ in range(1, order_count)
                                    ]
                                else:
                                    basis = retained_mass_uncertainty_basis(
                                        client.shift, row_budget, order_count
                                    )
                            local_logits = model(client.features, client.shift)
                            certificate = cone_pairwise_certificate_statistics(
                                model,
                                client.features,
                                client.shift,
                                row_budget,
                                local_logits,
                                basis,
                                count_semantics="upper_bound",
                                combine_symmetric=True,
                            )
                            logits_global[client.global_nodes] = local_logits
                            certified_global[client.global_nodes] = certificate.certified
                            radius_global[client.global_nodes] = certificate.representation_radius

                    if reference_logits is None:
                        reference_logits = logits_global.detach().cpu().clone()
                    elif not torch.equal(reference_logits, logits_global.detach().cpu()):
                        raise RuntimeError(f"ledger granularity changed logits: {identity}:{mode}")

                    full_logits = completion_logits(model, clients, full_shift, logits_global)
                    prediction = logits_global.argmax(dim=1)
                    full_prediction = full_logits.argmax(dim=1)
                    correct = prediction == labels_global
                    active_count = int(test_global.sum().item())
                    boundary_test = test_global & boundary_global
                    interior_test = test_global & ~boundary_global
                    per_client_ratio = []
                    per_client_accuracy = []
                    for client in clients:
                        client_mask = torch.zeros_like(test_global)
                        client_mask[client.global_nodes] = True
                        active = test_global & client_mask
                        count = int(active.sum().item())
                        per_client_ratio.append(
                            safe_div(int((certified_global & active).sum().item()), count)
                        )
                        per_client_accuracy.append(
                            safe_div(
                                int((certified_global & correct & active).sum().item()), count
                            )
                        )
                    violations = int(
                        (
                            certified_global
                            & (prediction != full_prediction)
                            & test_global
                        ).sum().item()
                    )
                    elapsed = time.perf_counter() - started
                    test_prediction = prediction[test_global].detach().cpu().numpy()
                    test_truth = labels_global[test_global].detach().cpu().numpy()
                    rows.append(
                        {
                            **expected_identity,
                            "variant": variant,
                            "certificate_information": mode,
                            "local_ledger_total_bits": mode_bits,
                            "local_ledger_bits_per_node": mode_bits / max(node_count, 1),
                            "maximum_count_overstatement": max_overstatement,
                            "test_local_macro_f1": float(
                                f1_score(
                                    test_truth,
                                    test_prediction,
                                    average="macro",
                                    zero_division=0,
                                )
                            ),
                            "test_certified_ratio": safe_div(
                                int((certified_global & test_global).sum().item()), active_count
                            ),
                            "test_certified_accuracy": safe_div(
                                int((certified_global & correct & test_global).sum().item()),
                                active_count,
                            ),
                            "test_boundary_certified_ratio": safe_div(
                                int((certified_global & boundary_test).sum().item()),
                                int(boundary_test.sum().item()),
                            ),
                            "test_boundary_certified_accuracy": safe_div(
                                int((certified_global & correct & boundary_test).sum().item()),
                                int(boundary_test.sum().item()),
                            ),
                            "test_interior_certified_ratio": safe_div(
                                int((certified_global & interior_test).sum().item()),
                                int(interior_test.sum().item()),
                            ),
                            "test_worst_client_certified_ratio": min(per_client_ratio),
                            "test_worst_client_certified_accuracy": min(per_client_accuracy),
                            "test_mean_representation_radius": float(
                                radius_global[test_global].mean().item()
                            ),
                            "test_certified_completion_violations": violations,
                            "verifier_wall_seconds": elapsed,
                        }
                    )

                exact_row = next(row for row in rows if row["certificate_information"] == "exact")
                ordered = ["exact", "bin2", "bin4", "bin8", "presence"]
                ordered_rows = {
                    row["certificate_information"]: row for row in rows if row["certificate_information"] in ordered
                }
                for finer, coarser in zip(ordered, ordered[1:]):
                    if (
                        float(ordered_rows[finer]["test_certified_ratio"])
                        + 1e-12
                        < float(ordered_rows[coarser]["test_certified_ratio"])
                    ):
                        raise RuntimeError(
                            f"refinement monotonicity failed: {identity}:{finer}:{coarser}"
                        )
                if any(int(row["test_certified_completion_violations"]) for row in rows):
                    raise RuntimeError(f"certificate violation: {identity}")
                write_csv(output / "certificate_metrics.csv", rows)
                completion.write_text(
                    json.dumps(
                        {
                            "protocol": config["protocol"],
                            **expected_identity,
                            "variant": variant,
                            "config_sha256": config_digest,
                            "bundle_sha256": bundle_digest,
                            "state_sha256": sha256(state_path),
                            "information_modes": list(modes),
                            "certificate": config["fixed_certificate"],
                            "no_training": True,
                            "no_selection": True,
                            "ordinary_logits_identical": True,
                            "safe_upper_bounds_verified": True,
                            "refinement_monotonicity_verified": True,
                            "test_labels_used_for_selection": False,
                            "test_metrics_emitted": True,
                            "exact_certified_ratio": exact_row["test_certified_ratio"],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[DONE] {identity}")
                completed += 1

    print(json.dumps({"completed_units": completed, "expected_units": config["expected_units"]}, indent=2))


if __name__ == "__main__":
    main()
