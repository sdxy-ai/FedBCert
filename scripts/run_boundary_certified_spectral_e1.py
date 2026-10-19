#!/usr/bin/env python3
"""E1 true-FedAvg confirmation for boundary-certified spectral learning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, List, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    boundary_severity_from_aggregates,
    build_boundary_ledger_views,
    build_federated_clients_from_ledgers,
    canonical_undirected,
    cpu_state,
    evaluate_federated_state,
    evaluate_personalized_states,
    induced_edges,
    missing_degrees,
    pairwise_masked_sum,
    prediction_certificate_statistics,
    row_normalized_shift,
    train_client,
    train_client_identity_null_staged,
    train_client_projected,
    weighted_average_states,
)
from openfgl.flcore.fedbeso import (
    balanced_random_membership,
    metis_membership,
    mixed_membership,
)
from openfgl.flcore.fedbcert.data import (
    load_cross_network_dataset,
    stratified_four_way_masks,
)


TensorState = Dict[str, torch.Tensor]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity_shift(node_count: int, device: torch.device) -> torch.Tensor:
    """Return a sparse identity propagation operator.

    This is used only by the parameter-matched no-graph audit.  Keeping the
    spectral expert, polynomial coefficients, optimizer, and aggregation
    unchanged while replacing every propagation step by the identity isolates
    the contribution of observed graph structure.
    """
    indices = torch.arange(int(node_count), device=device, dtype=torch.long)
    return torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        torch.ones(int(node_count), device=device),
        (int(node_count), int(node_count)),
        device=device,
    ).coalesce()


def official_four_way_masks(
    labels: torch.Tensor,
    official_train_mask: torch.Tensor,
    official_validation_mask: torch.Tensor,
    official_test_mask: torch.Tensor,
    *,
    seed: int,
    calibration_fraction_of_official_train: float = 1.0 / 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split only the official training fold into train/calibration folds."""
    indices = torch.where(official_train_mask.bool())[0].cpu().numpy()
    train_indices, calibration_indices = train_test_split(
        indices,
        test_size=float(calibration_fraction_of_official_train),
        random_state=int(seed),
        stratify=labels[indices].cpu().numpy(),
    )
    train_mask = torch.zeros_like(official_train_mask, dtype=torch.bool)
    calibration_mask = torch.zeros_like(train_mask)
    train_mask[torch.as_tensor(train_indices, dtype=torch.long)] = True
    calibration_mask[torch.as_tensor(calibration_indices, dtype=torch.long)] = True
    validation_mask = official_validation_mask.bool().clone()
    test_mask = official_test_mask.bool().clone()
    if bool(
        (train_mask & calibration_mask).any()
        or (train_mask & validation_mask).any()
        or (train_mask & test_mask).any()
        or (calibration_mask & validation_mask).any()
        or (calibration_mask & test_mask).any()
        or (validation_mask & test_mask).any()
    ):
        raise RuntimeError("official four-way masks overlap")
    return train_mask, calibration_mask, validation_mask, test_mask


def local_validation_metrics(
    state: TensorState,
    clients: Sequence,
    global_labels: torch.Tensor,
    global_validation_mask: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model = PolynomialSpectralClassifier(
        int(clients[0].features.size(1)),
        args.hidden_dim,
        int(global_labels.max().item()) + 1,
        args.order_count,
        args.dropout,
        args.encoder_type,
        args.encoder_depth,
        args.encoder_residual_scale,
        args.utility_diffusion_hops,
        args.utility_diffusion_scale,
        args.spectral_fusion_scale,
        args.certificate_margin_mode,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logits = torch.empty(
        (global_labels.numel(), int(global_labels.max().item()) + 1), device=device
    )
    certified = torch.zeros(global_labels.numel(), dtype=torch.bool, device=device)
    with torch.no_grad():
        for client in clients:
            prediction_stats = prediction_certificate_statistics(
                model, client.features, client.shift, client.uncertainty_basis
            )
            local_logits = prediction_stats.logits
            stats = prediction_stats.certificate
            logits[client.global_nodes] = local_logits
            certified[client.global_nodes] = stats.certified
    mask = global_validation_mask
    prediction = logits[mask].argmax(dim=1).detach().cpu().numpy()
    truth = global_labels[mask].detach().cpu().numpy()
    validation_f1 = float(
        f1_score(truth, prediction, average="macro", zero_division=0)
    )
    validation_accuracy = float(np.mean(prediction == truth))
    certified_accuracy = float(
        (
            certified
            & (logits.argmax(dim=1) == global_labels)
            & mask
        ).sum().item()
        / max(int(mask.sum().item()), 1)
    )
    return {
        "validation_local_macro_f1": validation_f1,
        "validation_local_accuracy": validation_accuracy,
        "validation_certified_accuracy": certified_accuracy,
    }


def personalized_validation_metrics(
    states: Sequence[TensorState],
    clients: Sequence,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, List[tuple[float, float]]]:
    predictions = []
    truths = []
    certified_correct = 0
    validation_count = 0
    client_scores: List[tuple[float, float]] = []
    with torch.no_grad():
        for state, client in zip(states, clients):
            model = PolynomialSpectralClassifier(
                int(client.features.size(1)),
                args.hidden_dim,
                int(state["classifier.weight"].size(0)),
                args.order_count,
                args.dropout,
                args.encoder_type,
                args.encoder_depth,
                args.encoder_residual_scale,
                args.utility_diffusion_hops,
                args.utility_diffusion_scale,
                args.spectral_fusion_scale,
                args.certificate_margin_mode,
            ).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            prediction_stats = prediction_certificate_statistics(
                model, client.features, client.shift, client.uncertainty_basis
            )
            logits = prediction_stats.logits
            stats = prediction_stats.certificate
            mask = client.validation_mask
            prediction = logits[mask].argmax(dim=1).detach().cpu().numpy()
            truth = client.labels[mask].detach().cpu().numpy()
            score = (
                float(f1_score(truth, prediction, average="macro", zero_division=0)),
                float(np.mean(prediction == truth)),
            )
            client_scores.append(score)
            predictions.append(prediction)
            truths.append(truth)
            certified_correct += int(
                (
                    stats.certified
                    & (logits.argmax(dim=1) == client.labels)
                    & mask
                ).sum().item()
            )
            validation_count += int(mask.sum().item())
    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    return {
        "validation_local_macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "validation_local_accuracy": float(np.mean(prediction == truth)),
        "validation_certified_accuracy": float(
            certified_correct / max(validation_count, 1)
        ),
    }, client_scores


def run_local_method(
    initial_state: TensorState,
    clients: Sequence,
    global_features: torch.Tensor,
    global_labels: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    full_shift: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> tuple[dict, List[dict]]:
    method_start = time.perf_counter()
    states = [
        {key: value.clone() for key, value in initial_state.items()}
        for _ in clients
    ]
    best_states = [
        {key: value.clone() for key, value in initial_state.items()}
        for _ in clients
    ]
    best_scores = [(-float("inf"), -float("inf")) for _ in clients]
    best_rounds = [0 for _ in clients]
    round_rows: List[dict] = []
    for round_id in range(1, args.rounds + 1):
        states = [
            train_client(
                global_state=state,
                client=client,
                input_dim=int(global_features.size(1)),
                hidden_dim=args.hidden_dim,
                num_classes=int(global_labels.max().item()) + 1,
                order_count=args.order_count,
                dropout=args.dropout,
                encoder_type=args.encoder_type,
                encoder_depth=args.encoder_depth,
                encoder_residual_scale=args.encoder_residual_scale,
                utility_diffusion_hops=args.utility_diffusion_hops,
                utility_diffusion_scale=args.utility_diffusion_scale,
                spectral_fusion_scale=args.spectral_fusion_scale,
                certificate_margin_mode=args.certificate_margin_mode,
                robust_lambda=0.0,
                certificate_margin=args.certificate_margin,
                local_epochs=args.local_epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed * 1000003 + round_id * 1009 + client.client_id,
                device=device,
            )
            for state, client in zip(states, clients)
        ]
        validation, client_scores = personalized_validation_metrics(
            states, clients, args, device
        )
        for index, score in enumerate(client_scores):
            if score > best_scores[index]:
                best_scores[index] = score
                best_rounds[index] = round_id
                best_states[index] = {
                    key: value.clone() for key, value in states[index].items()
                }
        round_rows.append({
            "method": "local",
            "round": round_id,
            "selected": False,
            **validation,
        })
    final_metrics = evaluate_personalized_states(
        best_states,
        clients,
        global_features=global_features,
        global_labels=global_labels,
        global_validation_mask=validation_mask,
        global_test_mask=test_mask,
        full_shift=full_shift,
        hidden_dim=args.hidden_dim,
        num_classes=int(global_labels.max().item()) + 1,
        order_count=args.order_count,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        encoder_depth=args.encoder_depth,
        encoder_residual_scale=args.encoder_residual_scale,
        utility_diffusion_hops=args.utility_diffusion_hops,
        utility_diffusion_scale=args.utility_diffusion_scale,
        device=device,
        include_test=args.evaluation_scope == "full",
        spectral_fusion_scale=args.spectral_fusion_scale,
        certificate_margin_mode=args.certificate_margin_mode,
        include_cone_certificate_audit=args.include_cone_certificate_audit,
    )
    parameter_count = int(
        sum(value.numel() for value in initial_state.values())
        - (
            initial_state["theta"].numel()
            if args.encoder_type == "feature_only_mlp"
            else 0
        )
    )
    model_state_bytes = int(
        sum(value.numel() * value.element_size() for value in initial_state.values())
    )
    for client, state in zip(clients, best_states):
        torch.save(state, output / f"local_client{client.client_id}_selected_state.pt")
    (output / "local_selected_rounds.json").write_text(
        json.dumps(
            {str(client.client_id): best_rounds[index]
             for index, client in enumerate(clients)},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "method": "local",
        "robust_lambda": 0.0,
        "selected_round": -1,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "gradient_mode": "local",
        "projection_step_count": 0,
        "projection_conflict_fraction": 0.0,
        "mean_pre_projection_cosine": 0.0,
        "projection_capped_fraction": 0.0,
        "mean_auxiliary_scale": 0.0,
        "minimum_post_projection_dot": 0.0,
        "trainable_parameters": parameter_count,
        "encoder_type": args.encoder_type,
        "encoder_depth": args.encoder_depth,
        "encoder_residual_scale": args.encoder_residual_scale,
        "utility_diffusion_hops": args.utility_diffusion_hops,
        "utility_diffusion_scale": args.utility_diffusion_scale,
        "upload_scalars_per_round": 0,
        "download_scalars_per_round": 0,
        "model_state_bytes": model_state_bytes,
        "upload_bytes_per_round": 0,
        "download_bytes_per_round": 0,
        "total_model_communication_bytes": 0,
        "boundary_ledger_bytes_one_time": 0,
        "method_wall_clock_seconds": float(time.perf_counter() - method_start),
        **final_metrics,
        "effective_robust_lambda": 0.0,
        "boundary_active": False,
        "boundary_cut_ratio": float(args._resolved_cut_ratio),
        "boundary_activation_threshold": args.boundary_activation_threshold,
    }, round_rows


def run_federated_method(
    method: str,
    robust_lambda: float,
    gradient_mode: str,
    initial_state: TensorState,
    clients: Sequence,
    global_features: torch.Tensor,
    global_labels: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    full_shift: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> tuple[dict, List[dict]]:
    method_start = time.perf_counter()
    global_state = {key: value.clone() for key, value in initial_state.items()}
    best_state = {key: value.clone() for key, value in global_state.items()}
    best_score = (-float("inf"), -float("inf"))
    best_round = 0
    round_rows: List[dict] = []
    projection_steps = 0
    projection_conflicts = 0
    projection_cosine_weighted_sum = 0.0
    minimum_post_projection_dot = float("inf")
    projection_capped = 0
    projection_scale_weighted_sum = 0.0
    for round_id in range(1, args.rounds + 1):
        staged_residual = args.identity_null_training_mode == "staged_residual"
        if staged_residual and round_id == args.feature_pretrain_rounds + 1:
            # Start correction from the validation-selected feature checkpoint
            # and restrict final checkpoint selection to the correction stage.
            global_state = {
                key: value.clone() for key, value in best_state.items()
            }
            best_score = (-float("inf"), -float("inf"))
            best_round = 0
        local_states = []
        sample_counts = []
        for client in clients:
            common_arguments = dict(
                    global_state=global_state,
                    client=client,
                    input_dim=int(global_features.size(1)),
                    hidden_dim=args.hidden_dim,
                    num_classes=int(global_labels.max().item()) + 1,
                    order_count=args.order_count,
                    dropout=args.dropout,
                    encoder_type=args.encoder_type,
                    encoder_depth=args.encoder_depth,
                    encoder_residual_scale=args.encoder_residual_scale,
                    utility_diffusion_hops=args.utility_diffusion_hops,
                    utility_diffusion_scale=args.utility_diffusion_scale,
                    spectral_fusion_scale=args.spectral_fusion_scale,
                    certificate_margin_mode=args.certificate_margin_mode,
                    robust_lambda=robust_lambda,
                    projection_ratio_cap=args.projection_ratio_cap,
                    certificate_margin=args.certificate_margin,
                    local_epochs=args.local_epochs,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    seed=args.seed * 1000003 + round_id * 1009 + client.client_id,
                    device=device,
            )
            if staged_residual:
                if robust_lambda != 0.0 or gradient_mode != "weighted":
                    raise ValueError(
                        "staged residual is defined only for standard FedAvg"
                    )
                local_state = train_client_identity_null_staged(
                    global_state=global_state,
                    client=client,
                    input_dim=int(global_features.size(1)),
                    hidden_dim=args.hidden_dim,
                    num_classes=int(global_labels.max().item()) + 1,
                    order_count=args.order_count,
                    dropout=args.dropout,
                    encoder_type=args.encoder_type,
                    encoder_depth=args.encoder_depth,
                    encoder_residual_scale=args.encoder_residual_scale,
                    utility_diffusion_hops=args.utility_diffusion_hops,
                    utility_diffusion_scale=args.utility_diffusion_scale,
                    local_epochs=args.local_epochs,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    seed=args.seed * 1000003 + round_id * 1009 + client.client_id,
                    device=device,
                    spectral_fusion_scale=float(args.spectral_fusion_scale),
                    certificate_margin_mode=args.certificate_margin_mode,
                    phase=(
                        "feature_pretrain"
                        if round_id <= args.feature_pretrain_rounds
                        else "residual_correction"
                    ),
                )
            elif gradient_mode == "projected":
                local_state, projection = train_client_projected(
                    **common_arguments
                )
                projection_steps += int(projection["projection_step_count"])
                projection_conflicts += int(
                    projection["projection_conflict_count"]
                )
                projection_cosine_weighted_sum += float(
                    projection["mean_pre_projection_cosine"]
                ) * int(projection["projection_step_count"])
                projection_capped += int(projection["projection_capped_count"])
                projection_scale_weighted_sum += float(
                    projection["mean_auxiliary_scale"]
                ) * int(projection["projection_step_count"])
                minimum_post_projection_dot = min(
                    minimum_post_projection_dot,
                    float(projection["minimum_post_projection_dot"]),
                )
            elif gradient_mode == "weighted":
                common_arguments.pop("projection_ratio_cap")
                local_state = train_client(**common_arguments)
            else:
                raise ValueError(f"unsupported gradient mode: {gradient_mode}")
            local_states.append(local_state)
            sample_counts.append(client.train_count)
        global_state = weighted_average_states(local_states, sample_counts)
        validation = local_validation_metrics(
            global_state,
            clients,
            global_labels,
            validation_mask,
            args,
            device,
        )
        score = (
            validation["validation_local_macro_f1"],
            validation["validation_local_accuracy"],
        )
        if score > best_score:
            best_score = score
            best_round = round_id
            best_state = {key: value.clone() for key, value in global_state.items()}
        round_rows.append(
            {
                "method": method,
                "round": round_id,
                "selected": False,
                **validation,
            }
        )
    for row in round_rows:
        row["selected"] = int(row["round"] == best_round)
    final_metrics = evaluate_federated_state(
        best_state,
        clients,
        global_features=global_features,
        global_labels=global_labels,
        global_validation_mask=validation_mask,
        global_test_mask=test_mask,
        full_shift=full_shift,
        hidden_dim=args.hidden_dim,
        num_classes=int(global_labels.max().item()) + 1,
        order_count=args.order_count,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        encoder_depth=args.encoder_depth,
        encoder_residual_scale=args.encoder_residual_scale,
        utility_diffusion_hops=args.utility_diffusion_hops,
        utility_diffusion_scale=args.utility_diffusion_scale,
        device=device,
        include_test=args.evaluation_scope == "full",
        spectral_fusion_scale=args.spectral_fusion_scale,
        certificate_margin_mode=args.certificate_margin_mode,
        include_cone_certificate_audit=args.include_cone_certificate_audit,
    )
    parameter_count = int(
        sum(value.numel() for value in best_state.values())
        - (
            best_state["theta"].numel()
            if args.encoder_type == "feature_only_mlp"
            else 0
        )
    )
    model_state_bytes = int(
        sum(value.numel() * value.element_size() for value in best_state.values())
    )
    upload_bytes_per_round = model_state_bytes * len(clients)
    download_bytes_per_round = model_state_bytes * len(clients)
    boundary_ledger_bytes = (
        2 * 8 * len(clients)
        if args.boundary_aggregation_mode in {"plain", "pairwise_masked"}
        and method == "fedbcert_ba"
        else 0
    )
    final = {
        "method": method,
        "robust_lambda": robust_lambda,
        "selected_round": best_round,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "gradient_mode": gradient_mode,
        "projection_step_count": projection_steps,
        "projection_conflict_fraction": (
            projection_conflicts / max(projection_steps, 1)
        ),
        "mean_pre_projection_cosine": (
            projection_cosine_weighted_sum / max(projection_steps, 1)
        ),
        "projection_capped_fraction": (
            projection_capped / max(projection_steps, 1)
        ),
        "mean_auxiliary_scale": (
            projection_scale_weighted_sum / max(projection_steps, 1)
        ),
        "minimum_post_projection_dot": (
            minimum_post_projection_dot if projection_steps else 0.0
        ),
        "trainable_parameters": parameter_count,
        "encoder_type": args.encoder_type,
        "encoder_depth": args.encoder_depth,
        "encoder_residual_scale": args.encoder_residual_scale,
        "utility_diffusion_hops": args.utility_diffusion_hops,
        "utility_diffusion_scale": args.utility_diffusion_scale,
        "upload_scalars_per_round": parameter_count * len(clients),
        "download_scalars_per_round": parameter_count * len(clients),
        "model_state_bytes": model_state_bytes,
        "upload_bytes_per_round": upload_bytes_per_round,
        "download_bytes_per_round": download_bytes_per_round,
        "total_model_communication_bytes": (
            (upload_bytes_per_round + download_bytes_per_round) * int(args.rounds)
        ),
        "boundary_ledger_bytes_one_time": boundary_ledger_bytes,
        "method_wall_clock_seconds": float(time.perf_counter() - method_start),
        **final_metrics,
    }
    torch.save(best_state, output / f"{method}_selected_state.pt")
    return final, round_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--protocol-name",
        default="",
        help="Optional protocol label overriding the historical E-stage label.",
    )
    parser.add_argument("--result-root", default="./results_boundary_certified")
    parser.add_argument("--data-root", default="./data/fedbcert")
    parser.add_argument(
        "--immutable-split-bundle",
        default="",
        help="Optional frozen split_bundle.pt replacing dataset/split generation.",
    )
    parser.add_argument(
        "--immutable-split-manifest",
        default="",
        help="Manifest containing the required bundle SHA-256 and identity.",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "Cora", "CiteSeer", "PubMed", "Computers", "Photo", "CS",
            "ogbn-arxiv",
        ],
        required=True,
    )
    parser.add_argument(
        "--partition",
        choices=[
            "metis", "mix05", "mix10", "mix20",
            "mixed25", "mixed50", "mixed75", "random",
        ],
        required=True,
    )
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--robust-lambda", type=float, required=True)
    parser.add_argument(
        "--robust-mode",
        choices=["weighted", "projected", "adaptive_projected"],
        default="weighted",
    )
    parser.add_argument(
        "--suite",
        choices=[
            "fedavg_only", "pair", "pair_with_local", "paper_internal", "scale_core",
        ],
        default="pair",
        help="Run the historical two-method protocol or the frozen E5 suite.",
    )
    parser.add_argument(
        "--projection-ratio-cap",
        type=float,
        default=1.0,
        help="Cap on ||lambda*g_certificate|| / ||g_classification||.",
    )
    parser.add_argument("--boundary-activation-threshold", type=float, default=0.20)
    parser.add_argument(
        "--boundary-budget-source",
        choices=["benchmark_oracle", "local_stub_ledger"],
        default="benchmark_oracle",
    )
    parser.add_argument(
        "--boundary-aggregation-mode",
        choices=["plain", "pairwise_masked"],
        default="plain",
    )
    parser.add_argument(
        "--lambda-source",
        default="explicit command-line value",
        help="Audit description of how robust-lambda was fixed.",
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--encoder-type",
        choices=[
            "linear_relu",
            "bounded_residual_mlp",
            "feature_only_mlp",
            "dual_feature_spectral",
            "dual_feature_identity_null_spectral",
            "shielded_sage_spectral",
            "shielded_diffusion_spectral",
        ],
        default="linear_relu",
    )
    parser.add_argument("--encoder-depth", type=int, default=1)
    parser.add_argument("--encoder-residual-scale", type=float, default=0.5)
    parser.add_argument("--utility-diffusion-hops", type=int, default=1)
    parser.add_argument("--utility-diffusion-scale", type=float, default=0.25)
    parser.add_argument(
        "--spectral-fusion-scale",
        type=float,
        default=None,
        help=(
            "Scale on dual-expert spectral logits. Omitted preserves the E10 "
            "default 0.25; zero is the pure-feature degeneration baseline."
        ),
    )
    parser.add_argument(
        "--spectral-shift-mode",
        choices=["observed", "identity"],
        default="observed",
        help=(
            "Use the observed induced-graph propagation operator or replace "
            "every polynomial propagation step by I for the parameter-matched "
            "no-graph control."
        ),
    )
    parser.add_argument(
        "--identity-null-training-mode",
        choices=["separate_ce", "staged_residual"],
        default="separate_ce",
        help=(
            "Historical separate expert CE, or feature pretraining followed "
            "by frozen-feature fused-loss identity-null residual correction."
        ),
    )
    parser.add_argument(
        "--feature-pretrain-rounds",
        type=int,
        default=0,
        help="Communication rounds completed before residual correction.",
    )
    parser.add_argument(
        "--certificate-margin-mode",
        choices=["classwise_sum", "pairwise_difference"],
        default="classwise_sum",
        help="Historical classwise bound or the tighter class-difference bound.",
    )
    parser.add_argument(
        "--uncertainty-recurrence",
        choices=["legacy", "retained_mass"],
        default="legacy",
        help="Historical recurrence or the retained-observed-mass contraction.",
    )
    parser.add_argument(
        "--include-cone-certificate-audit",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Also evaluate the E18 ReLU-positive-cone one-sided and hybrid "
            "certificates on the identical selected predictor."
        ),
    )
    parser.add_argument("--order-count", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--certificate-margin", type=float, default=0.05)
    parser.add_argument("--train-ratio", type=float, default=0.30)
    parser.add_argument("--calibration-ratio", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.40)
    parser.add_argument(
        "--evaluation-scope",
        choices=["full", "validation_only"],
        default="full",
        help="validation_only suppresses every test metric during development.",
    )
    parser.add_argument("--use-cuda", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wall_start = time.perf_counter()
    if args.robust_lambda <= 0:
        raise ValueError("robust-lambda must be positive; FedAvg is run automatically")
    if args.projection_ratio_cap <= 0:
        raise ValueError("projection-ratio-cap must be positive")
    if args.encoder_depth < 1:
        raise ValueError("encoder-depth must be positive")
    if args.encoder_type == "linear_relu" and args.encoder_depth != 1:
        raise ValueError("linear_relu requires --encoder-depth 1")
    if args.encoder_type == "bounded_residual_mlp" and args.encoder_depth < 2:
        raise ValueError("bounded_residual_mlp requires --encoder-depth >= 2")
    if args.encoder_type == "dual_feature_spectral" and args.encoder_depth < 2:
        raise ValueError("dual_feature_spectral requires --encoder-depth >= 2")
    if (
        args.encoder_type == "dual_feature_identity_null_spectral"
        and args.encoder_depth < 2
    ):
        raise ValueError(
            "dual_feature_identity_null_spectral requires --encoder-depth >= 2"
        )
    if args.encoder_type == "shielded_sage_spectral" and args.encoder_depth < 2:
        raise ValueError("shielded_sage_spectral requires --encoder-depth >= 2")
    if args.encoder_type == "shielded_diffusion_spectral" and args.encoder_depth < 2:
        raise ValueError("shielded_diffusion_spectral requires --encoder-depth >= 2")
    if args.encoder_residual_scale <= 0:
        raise ValueError("encoder-residual-scale must be positive")
    if args.utility_diffusion_hops < 1:
        raise ValueError("utility-diffusion-hops must be positive")
    if args.utility_diffusion_scale <= 0:
        raise ValueError("utility-diffusion-scale must be positive")
    if args.spectral_fusion_scale is not None and args.spectral_fusion_scale < 0:
        raise ValueError("spectral-fusion-scale must be nonnegative")
    if args.identity_null_training_mode == "staged_residual":
        if args.encoder_type != "dual_feature_identity_null_spectral":
            raise ValueError("staged_residual requires identity-null encoder")
        if not 0 < args.feature_pretrain_rounds < args.rounds:
            raise ValueError(
                "staged_residual requires 0 < feature-pretrain-rounds < rounds"
            )
        if args.suite != "fedavg_only":
            raise ValueError("staged_residual is frozen only for fedavg_only")
    elif args.feature_pretrain_rounds != 0:
        raise ValueError("feature-pretrain-rounds is only valid for staged_residual")
    if not 0.0 <= args.boundary_activation_threshold <= 1.0:
        raise ValueError("boundary-activation-threshold must lie in [0, 1]")
    run_name = f"{args.dataset}_{args.partition}_c{args.num_clients}_seed{args.seed}"
    output = Path(args.result_root) / args.exp_name / run_name
    completion = output / "run_complete.json"
    if completion.exists() and not args.force:
        print(f"[SKIP] completed {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        # Some PyTorch/CUDA combinations reject an otherwise valid torch.device
        # object here.  Memory telemetry must never abort a training run.
        try:
            torch.cuda.set_device(args.gpu)
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as error:
            print(f"[WARN] CUDA peak-memory reset unavailable: {error}")
    frozen_masks = None
    immutable_bundle_audit = {
        "enabled": False,
        "hash_verified": False,
    }
    if args.immutable_split_bundle:
        if not args.immutable_split_manifest:
            raise ValueError("immutable bundle requires --immutable-split-manifest")
        bundle_path = Path(args.immutable_split_bundle).resolve()
        manifest_path = Path(args.immutable_split_manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        if digest != manifest.get("bundle_sha256"):
            raise RuntimeError("immutable split bundle SHA-256 mismatch")
        identity = {
            "dataset": args.dataset,
            "partition": args.partition,
            "num_clients": args.num_clients,
            "seed": args.seed,
        }
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise RuntimeError("immutable split manifest identity mismatch")
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
        expected_keys = {
            "x", "y", "edge_index", "membership", "train_mask",
            "calibration_mask", "validation_mask", "test_mask",
        }
        if set(bundle) != expected_keys:
            raise RuntimeError(f"unexpected immutable bundle keys: {sorted(bundle)}")
        features = bundle["x"].float()
        labels = bundle["y"].reshape(-1).long()
        node_count = int(features.size(0))
        full_edges = canonical_undirected(bundle["edge_index"], node_count)
        membership = bundle["membership"].long().cpu().numpy()
        frozen_masks = (
            bundle["train_mask"].bool(),
            bundle["calibration_mask"].bool(),
            bundle["validation_mask"].bool(),
            bundle["test_mask"].bool(),
        )
        immutable_bundle_audit = {
            "enabled": True,
            "bundle": str(bundle_path),
            "manifest": str(manifest_path),
            "bundle_sha256": digest,
            "hash_verified": True,
            "identity_verified": True,
            "exact_keys_verified": True,
        }
    else:
        if args.immutable_split_manifest:
            raise ValueError("manifest supplied without --immutable-split-bundle")
        data = load_cross_network_dataset(args.dataset, args.data_root)
        features = data.x.float()
        labels = data.y.reshape(-1).long()
        node_count = int(data.num_nodes)
        full_edges = canonical_undirected(data.edge_index, node_count)
        if all(
            hasattr(data, attribute)
            for attribute in (
                "official_train_mask",
                "official_validation_mask",
                "official_test_mask",
            )
        ):
            frozen_masks = official_four_way_masks(
                labels,
                data.official_train_mask,
                data.official_validation_mask,
                data.official_test_mask,
                seed=args.seed,
            )
            immutable_bundle_audit["official_dataset_split"] = True

    if immutable_bundle_audit.get("enabled", False):
        if int(np.min(membership)) != 0 or int(np.max(membership)) + 1 != args.num_clients:
            raise RuntimeError("immutable bundle membership/client-count mismatch")
    elif args.partition == "metis":
        membership = metis_membership(
            full_edges, node_count, args.num_clients, args.seed
        )
    elif args.partition == "random":
        membership = balanced_random_membership(
            node_count, args.num_clients, args.seed
        )
    else:
        mixing = {
            "mix05": 0.05,
            "mix10": 0.10,
            "mix20": 0.20,
            "mixed25": 0.25,
            "mixed50": 0.50,
            "mixed75": 0.75,
        }[args.partition]
        membership = mixed_membership(
            full_edges,
            node_count,
            args.num_clients,
            args.seed,
            mixing,
        )
    local_union = induced_edges(full_edges, membership)
    missing = missing_degrees(full_edges, local_union, node_count)
    ledgers = build_boundary_ledger_views(full_edges, membership, missing)
    client_boundary_counts = [
        [ledger.retained_undirected_edges, ledger.missing_degree_sum]
        for ledger in ledgers
    ]
    if args.boundary_aggregation_mode == "pairwise_masked":
        aggregated_counts, aggregation_audit = pairwise_masked_sum(
            client_boundary_counts,
            seed=args.seed * 1000033 + 811,
        )
    else:
        aggregated_counts = [
            sum(row[column] for row in client_boundary_counts)
            for column in range(2)
        ]
        aggregation_audit = {
            "aggregation": "plain_sum",
            "client_count": len(client_boundary_counts),
            "payload_scalars_per_client": 2,
            "total_payload_scalars": 2 * len(client_boundary_counts),
            "masks_cancel_exactly": None,
            "individual_unmasked_values_visible_to_server": True,
            "differential_privacy_claimed": False,
        }
    retained_edges, missing_degree_sum = aggregated_counts
    if retained_edges != int(local_union.size(1) // 2):
        raise RuntimeError("aggregated retained-edge count is inconsistent")
    if missing_degree_sum != int(missing.sum().item()):
        raise RuntimeError("aggregated missing-degree count is inconsistent")
    cut_edges = int(missing_degree_sum // 2)
    cut_ratio = boundary_severity_from_aggregates(
        retained_edges, missing_degree_sum
    )
    undirected_edges = retained_edges + cut_edges
    adaptive_active = (
        args.robust_mode != "adaptive_projected"
        or cut_ratio >= args.boundary_activation_threshold
    )
    args._resolved_cut_ratio = cut_ratio
    masks = frozen_masks or stratified_four_way_masks(
        labels,
        train_ratio=args.train_ratio,
        calibration_ratio=args.calibration_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_mask, _, validation_mask, test_mask = masks
    clients = build_federated_clients_from_ledgers(
        ledgers,
        features,
        labels,
        train_mask,
        validation_mask,
        test_mask,
        args.order_count,
        device,
        args.uncertainty_recurrence,
    )
    full_shift, _ = row_normalized_shift(full_edges, node_count, device=device)
    if args.spectral_shift_mode == "identity":
        for client in clients:
            client.shift = identity_shift(int(client.features.size(0)), device)
            client.uncertainty_basis = [
                torch.zeros(
                    int(client.features.size(0)),
                    device=device,
                    dtype=client.features.dtype,
                )
                for _ in range(int(args.order_count))
            ]
        full_shift = identity_shift(node_count, device)
    features = features.to(device)
    labels = labels.to(device)
    validation_mask = validation_mask.to(device)
    test_mask = test_mask.to(device)

    seed_everything(args.seed * 100003 + 97)
    template = PolynomialSpectralClassifier(
        int(features.size(1)),
        args.hidden_dim,
        int(labels.max().item()) + 1,
        args.order_count,
        args.dropout,
        args.encoder_type,
        args.encoder_depth,
        args.encoder_residual_scale,
        args.utility_diffusion_hops,
        args.utility_diffusion_scale,
        args.spectral_fusion_scale,
        args.certificate_margin_mode,
    )
    if args.identity_null_training_mode == "staged_residual":
        # The correction stage starts from the exact feature predictor.
        with torch.no_grad():
            template.classifier.weight.zero_()
    initial_state = cpu_state(template)
    final_rows = []
    all_round_rows = []
    baseline, baseline_rounds = run_federated_method(
        "fedavg", 0.0, "weighted", initial_state, clients, features, labels,
        validation_mask, test_mask, full_shift, args, device, output,
    )
    baseline.update({
        "effective_robust_lambda": 0.0,
        "boundary_active": False,
        "boundary_cut_ratio": cut_ratio,
        "boundary_activation_threshold": args.boundary_activation_threshold,
    })
    final_rows.append({
        "dataset": args.dataset, "partition": args.partition,
        "num_clients": args.num_clients, "seed": args.seed, **baseline,
    })
    all_round_rows.extend(baseline_rounds)

    if args.suite == "pair_with_local":
        local, local_rounds = run_local_method(
            initial_state,
            clients,
            features,
            labels,
            validation_mask,
            test_mask,
            full_shift,
            args,
            device,
            output,
        )
        final_rows.insert(0, {
            "dataset": args.dataset, "partition": args.partition,
            "num_clients": args.num_clients, "seed": args.seed, **local,
        })
        all_round_rows.extend(local_rounds)

    if args.suite == "paper_internal":
        local, local_rounds = run_local_method(
            initial_state,
            clients,
            features,
            labels,
            validation_mask,
            test_mask,
            full_shift,
            args,
            device,
            output,
        )
        final_rows.insert(0, {
            "dataset": args.dataset, "partition": args.partition,
            "num_clients": args.num_clients, "seed": args.seed, **local,
        })
        all_round_rows.extend(local_rounds)
        weighted, weighted_rounds = run_federated_method(
            "fedbcert_weighted",
            args.robust_lambda,
            "weighted",
            initial_state,
            clients,
            features,
            labels,
            validation_mask,
            test_mask,
            full_shift,
            args,
            device,
            output,
        )
        weighted.update({
            "effective_robust_lambda": args.robust_lambda,
            "boundary_active": True,
            "boundary_cut_ratio": cut_ratio,
            "boundary_activation_threshold": args.boundary_activation_threshold,
        })
        projected, projected_rounds = run_federated_method(
            "fedbcert_pg",
            args.robust_lambda,
            "projected",
            initial_state,
            clients,
            features,
            labels,
            validation_mask,
            test_mask,
            full_shift,
            args,
            device,
            output,
        )
        projected.update({
            "effective_robust_lambda": args.robust_lambda,
            "boundary_active": True,
            "boundary_cut_ratio": cut_ratio,
            "boundary_activation_threshold": args.boundary_activation_threshold,
        })
        if adaptive_active:
            adaptive = projected.copy()
            adaptive.update({
                "method": "fedbcert_ba",
                "boundary_active": True,
            })
            adaptive_rounds = [
                {**row, "method": "fedbcert_ba"}
                for row in projected_rounds
            ]
            shutil.copyfile(
                output / "fedbcert_pg_selected_state.pt",
                output / "fedbcert_ba_selected_state.pt",
            )
        else:
            adaptive = baseline.copy()
            adaptive.update({
                "method": "fedbcert_ba",
                "robust_lambda": args.robust_lambda,
                "effective_robust_lambda": 0.0,
                "gradient_mode": "inactive_fedavg",
                "boundary_active": False,
            })
            adaptive_rounds = [
                {**row, "method": "fedbcert_ba"}
                for row in baseline_rounds
            ]
            shutil.copyfile(
                output / "fedavg_selected_state.pt",
                output / "fedbcert_ba_selected_state.pt",
            )
        for method_result in (weighted, projected, adaptive):
            final_rows.append({
                "dataset": args.dataset, "partition": args.partition,
                "num_clients": args.num_clients, "seed": args.seed,
                **method_result,
            })
        all_round_rows.extend(weighted_rounds)
        all_round_rows.extend(projected_rounds)
        all_round_rows.extend(adaptive_rounds)
    elif args.suite == "scale_core":
        projected, projected_rounds = run_federated_method(
            "fedbcert_pg",
            args.robust_lambda,
            "projected",
            initial_state,
            clients,
            features,
            labels,
            validation_mask,
            test_mask,
            full_shift,
            args,
            device,
            output,
        )
        projected.update({
            "effective_robust_lambda": args.robust_lambda,
            "boundary_active": True,
            "boundary_cut_ratio": cut_ratio,
            "boundary_activation_threshold": args.boundary_activation_threshold,
        })
        if adaptive_active:
            adaptive = projected.copy()
            adaptive.update({
                "method": "fedbcert_ba",
                "boundary_active": True,
            })
            adaptive_rounds = [
                {**row, "method": "fedbcert_ba"}
                for row in projected_rounds
            ]
            shutil.copyfile(
                output / "fedbcert_pg_selected_state.pt",
                output / "fedbcert_ba_selected_state.pt",
            )
        else:
            adaptive = baseline.copy()
            adaptive.update({
                "method": "fedbcert_ba",
                "robust_lambda": args.robust_lambda,
                "effective_robust_lambda": 0.0,
                "gradient_mode": "inactive_fedavg",
                "boundary_active": False,
            })
            adaptive_rounds = [
                {**row, "method": "fedbcert_ba"}
                for row in baseline_rounds
            ]
            shutil.copyfile(
                output / "fedavg_selected_state.pt",
                output / "fedbcert_ba_selected_state.pt",
            )
        for method_result in (projected, adaptive):
            final_rows.append({
                "dataset": args.dataset, "partition": args.partition,
                "num_clients": args.num_clients, "seed": args.seed,
                **method_result,
            })
        all_round_rows.extend(projected_rounds)
        all_round_rows.extend(adaptive_rounds)
    elif args.suite == "fedavg_only":
        # Post-hoc FedBCert evaluation of an otherwise standard FedAvg predictor.
        # The baseline row and selected state were already produced above.
        pass
    else:
        if args.robust_mode == "adaptive_projected" and not adaptive_active:
            robust = baseline.copy()
            robust.update({
                "method": "fedbcert_ba",
                "robust_lambda": args.robust_lambda,
                "effective_robust_lambda": 0.0,
                "gradient_mode": "inactive_fedavg",
                "boundary_active": False,
            })
            robust_rounds = [
                {**row, "method": "fedbcert_ba"} for row in baseline_rounds
            ]
            shutil.copyfile(
                output / "fedavg_selected_state.pt",
                output / "fedbcert_ba_selected_state.pt",
            )
        else:
            if args.robust_mode in {"projected", "adaptive_projected"}:
                robust_name = (
                    "fedbcert_ba"
                    if args.robust_mode == "adaptive_projected"
                    else "fedbcert_pg"
                )
                gradient_mode = "projected"
            else:
                robust_name = "fedbcert"
                gradient_mode = "weighted"
            robust, robust_rounds = run_federated_method(
                robust_name, args.robust_lambda, gradient_mode, initial_state,
                clients, features, labels, validation_mask, test_mask, full_shift,
                args, device, output,
            )
            robust.update({
                "effective_robust_lambda": args.robust_lambda,
                "boundary_active": bool(
                    args.robust_mode == "adaptive_projected" and adaptive_active
                ),
                "boundary_cut_ratio": cut_ratio,
                "boundary_activation_threshold": args.boundary_activation_threshold,
            })
        final_rows.append({
            "dataset": args.dataset, "partition": args.partition,
            "num_clients": args.num_clients, "seed": args.seed, **robust,
        })
        all_round_rows.extend(robust_rounds)
    write_csv(output / "federated_metrics.csv", final_rows)
    write_csv(output / "round_validation_metrics.csv", all_round_rows)
    protocol = {
        "protocol": args.protocol_name or (
            "E6 frozen dataset and client-scale confirmation"
            if args.suite == "scale_core"
            else "E5 frozen internal baselines and boundary-severity validation"
            if args.suite == "paper_internal"
            else "E3 boundary-adaptive projected-gradient spectral certification"
            if args.robust_mode == "adaptive_projected"
            else "E2 projected-gradient boundary-certified spectral learning"
            if args.robust_mode == "projected"
            else "E1 true-FedAvg boundary-certified spectral learning"
        ),
        "robust_gradient_mode": args.robust_mode,
        "development_seed_excluded": 0,
        "seed": args.seed,
        "robust_lambda_source": args.lambda_source,
        "robust_lambda": args.robust_lambda,
        "client_data": "remapped induced graph, local node tensors, masks, and missing-degree budgets",
        "server_observes": (
            "model states, labelled-training counts, and the final masked-sum boundary totals"
            if args.boundary_aggregation_mode == "pairwise_masked"
            else "model states, labelled-training counts, and plain boundary-count contributions"
            if args.robust_mode == "adaptive_projected"
            else "model states and labelled-training counts only"
        ),
        "remote_endpoint_identity_used_by_client_training": False,
        "remote_features_used_by_client_training": False,
        "missing_degree_budget_source": (
            "client-local boundary-stub ledger; exact counts; no remote endpoints"
            if args.boundary_budget_source == "local_stub_ledger"
            else "benchmark-oracle exact counts; no endpoints"
        ),
        "boundary_budget_privacy_scope": (
            "remote-endpoint blindness and hidden per-client scalar contributions; no differential-privacy claim"
            if args.boundary_aggregation_mode == "pairwise_masked"
            else "no differential-privacy claim"
        ),
        "training_client_builder_accepts_full_edge_list": False,
        "secure_aggregation_audit": aggregation_audit,
        "full_graph_usage": (
            "validation-only completion diagnostic; no test metric emitted"
            if args.evaluation_scope == "validation_only"
            else "final frozen-model diagnostic only"
        ),
        "round_selection_metric": "validation local Macro-F1; accuracy tie-break",
        "test_labels_used_for_selection": False,
        "test_metrics_emitted": args.evaluation_scope == "full",
        "evaluation_scope": args.evaluation_scope,
        "immutable_split_bundle_audit": immutable_bundle_audit,
        "bounded_encoder_contract": (
            "agreement-shielded spectral path has evaluated hidden-row l2 norm at most one; "
            "the E10 feature plus label-free diffusion utility expert is never certified"
            if args.encoder_type == "shielded_diffusion_spectral"
            else
            "agreement-shielded spectral path has evaluated hidden-row l2 norm at most one; "
            "the local GraphSAGE utility expert is never certified; only utility/safe class-agreement nodes may enter the certificate set"
            if args.encoder_type == "shielded_sage_spectral"
            else
            "spectral-path evaluated hidden rows have l2 norm at most one; "
            "graph-invariant feature expert has exactly zero boundary sensitivity"
            if args.encoder_type in {
                "dual_feature_spectral",
                "dual_feature_identity_null_spectral",
            }
            else
            "two-layer feature-only MLP; logits have exactly zero boundary sensitivity"
            if args.encoder_type == "feature_only_mlp"
            else "evaluated hidden rows have l2 norm at most one"
        ),
        "graph_invariant_feature_expert": (
            args.encoder_type in {
                "feature_only_mlp",
                "dual_feature_spectral",
                "dual_feature_identity_null_spectral",
            }
        ),
        "spectral_fusion_scale": float(
            args.spectral_fusion_scale
            if args.spectral_fusion_scale is not None
            else (
                0.0
                if args.encoder_type == "feature_only_mlp"
                else 0.25
                if args.encoder_type in {
                    "dual_feature_spectral",
                    "dual_feature_identity_null_spectral",
                }
                else 1.0
            )
        ),
        "spectral_shift_mode": args.spectral_shift_mode,
        "identity_null_spectral_residual": (
            args.encoder_type == "dual_feature_identity_null_spectral"
        ),
        "identity_null_formula": (
            "sum_{k>=1} theta_k (P^k-I)H with bias-free spectral head"
            if args.encoder_type == "dual_feature_identity_null_spectral"
            else None
        ),
        "identity_null_training_mode": args.identity_null_training_mode,
        "feature_pretrain_rounds": int(args.feature_pretrain_rounds),
        "zero_initialized_spectral_classifier": (
            args.identity_null_training_mode == "staged_residual"
        ),
        "parameter_matched_no_graph_control": (
            args.spectral_shift_mode == "identity"
        ),
        "certificate_margin_mode": args.certificate_margin_mode,
        "uncertainty_recurrence": args.uncertainty_recurrence,
        "cone_certificate_audit": {
            "enabled": bool(args.include_cone_certificate_audit),
            "encoder_range": "nonnegative unit l2 ball",
            "count_semantics": "actual boundary count no greater than ledger value",
            "primary_predictor_changed": False,
            "hybrid_rule": "union of retained-mass pairwise and cone one-sided certificates",
        },
        "utility_spectral_logit_residual_scale": (
            float(args.spectral_fusion_scale)
            if args.encoder_type in {
                "dual_feature_spectral",
                "dual_feature_identity_null_spectral",
                "shielded_diffusion_spectral",
            }
            else None
        ),
        "separate_feature_and_spectral_classification_losses": (
            args.encoder_type in {
                "dual_feature_spectral",
                "dual_feature_identity_null_spectral",
                "shielded_diffusion_spectral",
            }
            and args.identity_null_training_mode == "separate_ce"
        ),
        "feature_expert_detached_from_certificate_auxiliary_gradient": (
            args.encoder_type in {
                "dual_feature_spectral",
                "dual_feature_identity_null_spectral",
                "shielded_diffusion_spectral",
            }
        ),
        "agreement_shield_enabled": (
            args.encoder_type in {
                "shielded_sage_spectral",
                "shielded_diffusion_spectral",
            }
        ),
        "utility_expert": (
            "E10 feature expert plus label-free local diffusion residual; ordinary prediction source; not certified"
            if args.encoder_type == "shielded_diffusion_spectral"
            else
            "local mean-GraphSAGE; ordinary prediction source; not certified"
            if args.encoder_type == "shielded_sage_spectral"
            else None
        ),
        "shield_rule": (
            "fixed local safe-certificate AND safe/utility nominal class agreement"
            if args.encoder_type in {
                "shielded_sage_spectral",
                "shielded_diffusion_spectral",
            }
            else None
        ),
        "shield_recomputed_on_completion": False,
        "undirected_edges": undirected_edges,
        "retained_undirected_edges": retained_edges,
        "cut_edges": cut_edges,
        "cut_ratio": cut_ratio,
        "boundary_activation_threshold": args.boundary_activation_threshold,
        "boundary_active": bool(
            args.robust_mode == "adaptive_projected" and adaptive_active
        ),
        "boundary_activation_uses_labels": False,
        "arguments": vars(args),
    }
    protocol["wall_clock_seconds"] = float(time.perf_counter() - wall_start)
    try:
        protocol["peak_cuda_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        )
    except RuntimeError as error:
        print(f"[WARN] CUDA peak-memory query unavailable: {error}")
        protocol["peak_cuda_memory_bytes"] = None
    try:
        import resource

        protocol["peak_process_rss_bytes"] = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
    except (ImportError, AttributeError):
        protocol["peak_process_rss_bytes"] = None
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savetxt(
        output / "partition_membership.csv",
        np.column_stack([np.arange(node_count), membership]),
        delimiter=",",
        header="node_id,client_id",
        comments="",
        fmt="%d",
    )
    completion.write_text(
        json.dumps(
            {"status": "complete", "method_count": len(final_rows)},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {output}")


if __name__ == "__main__":
    main()
