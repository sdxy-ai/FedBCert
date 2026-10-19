#!/usr/bin/env python3
"""Train one endpoint-blind safe expert around a frozen FedLap+ utility expert.

E22 preserves the validation-selected FedLap+ prediction class exactly and
learns only an endpoint-blind safe expert.  Completion-certified agreement
distillation (CAD) asks that the safe expert certify the frozen teacher class,
after which a fixed agreement shield exposes certificates for the final
teacher-equivalent predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    build_boundary_ledger_views,
    build_federated_clients_from_ledgers,
    canonical_undirected,
    cpu_state,
    evaluate_external_teacher_state,
    induced_edges,
    missing_degrees,
    row_normalized_shift,
    train_client_external_teacher,
    weighted_average_states,
)


VARIANTS = {
    "spectral_ce": ("linear_relu", 1, "ce"),
    "spectral_kd": ("linear_relu", 1, "kd"),
    "spectral_cad": ("linear_relu", 1, "cad"),
    "spectral_identity_cad": ("linear_relu", 1, "cad"),
    "spectral_kd_cad": ("linear_relu", 1, "kd_cad"),
    "feature_ce": ("feature_only_mlp", 2, "ce"),
    "feature_kd_cad": ("feature_only_mlp", 2, "kd_cad"),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["Cora", "CiteSeer", "PubMed", "Photo"], required=True)
    p.add_argument("--partition", choices=["metis", "random"], required=True)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument(
        "--bundle-root",
        default="results_boundary_certified/boundary_certified_spectral_e22_teacher_shield/bundles",
    )
    p.add_argument(
        "--teacher-root",
        default="results_boundary_certified/boundary_certified_spectral_e22_teacher_shield/teachers",
    )
    p.add_argument(
        "--output-root",
        default="results_boundary_certified/boundary_certified_spectral_e22_teacher_shield/development",
    )
    p.add_argument("--protocol-name", default="FedBCert E22 strong-utility certified agreement distillation")
    p.add_argument("--evaluation-scope", choices=["validation_only", "test_once"], default="validation_only")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--order-count", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--encoder-residual-scale", type=float, default=0.5)
    p.add_argument("--distillation-weight", type=float, default=1.0)
    p.add_argument("--distillation-temperature", type=float, default=2.0)
    p.add_argument("--cad-weight", type=float, default=0.25)
    p.add_argument("--certificate-margin", type=float, default=0.05)
    p.add_argument("--certificate-kind", choices=["symmetric", "cone_hybrid"], default="cone_hybrid")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parser().parse_args()
    if args.num_clients < 2:
        raise ValueError("E22 requires at least two clients")
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("training horizons must be positive")
    encoder_type, encoder_depth, loss_mode = VARIANTS[args.variant]
    propagation_mode = (
        "identity" if args.variant == "spectral_identity_cad" else "observed"
    )
    certificate_kind = (
        "symmetric" if propagation_mode == "identity" else args.certificate_kind
    )
    run_name = f"{args.dataset}_{args.partition}_c{args.num_clients}_seed{args.seed}"
    bundle_dir = (REPOSITORY_ROOT / args.bundle_root / run_name).resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing immutable E22 bundle: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_path = bundle_dir / manifest["bundle"]
    if sha256(bundle_path) != manifest["bundle_sha256"]:
        raise RuntimeError("immutable E22 bundle SHA-256 mismatch")
    identity = {
        "dataset": args.dataset,
        "partition": args.partition,
        "num_clients": int(args.num_clients),
        "seed": int(args.seed),
    }
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise RuntimeError("E22 bundle identity does not match requested run")

    teacher_dir = (REPOSITORY_ROOT / args.teacher_root / run_name).resolve()
    teacher_path = teacher_dir / "utility_logits.pt"
    teacher_metrics_path = teacher_dir / "fedlap_plus_official_metrics.json"
    if not teacher_path.exists() or not teacher_metrics_path.exists():
        raise FileNotFoundError(
            f"missing frozen FedLap+ teacher artifacts under {teacher_dir}"
        )
    teacher_record = json.loads(teacher_metrics_path.read_text(encoding="utf-8"))
    if int(teacher_record.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError(
            "E22 teacher materialization must not evaluate test labels; rerun "
            "the FedLap+ adapter with --skip-test-evaluation"
        )
    teacher_artifact = torch.load(teacher_path, map_location="cpu", weights_only=True)
    if teacher_artifact.get("bundle_sha256") != manifest["bundle_sha256"]:
        raise RuntimeError("teacher logits were not generated from this immutable bundle")
    if teacher_artifact.get("source_commit") != "652fba0bb2e9a660a31ce7e8abcf728c225d188c":
        raise RuntimeError("teacher logits do not come from the pinned FedLap+ commit")
    teacher_logits = teacher_artifact["logits"].float()

    output_dir = (REPOSITORY_ROOT / args.output_root / args.variant / run_name).resolve()
    result_path = output_dir / "metrics.json"
    complete_path = output_dir / "run_complete.json"
    if result_path.exists() and complete_path.exists() and not args.force:
        print(f"[SKIP] complete: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    expected_keys = {
        "x", "y", "edge_index", "membership", "train_mask",
        "calibration_mask", "validation_mask", "test_mask",
    }
    if set(bundle) != expected_keys:
        raise RuntimeError(f"unexpected E22 bundle keys: {sorted(bundle)}")
    features = bundle["x"].float()
    labels = bundle["y"].reshape(-1).long()
    node_count = int(features.size(0))
    class_count = int(manifest["class_count"])
    if teacher_logits.shape != (node_count, class_count):
        raise RuntimeError("teacher artifact shape disagrees with the bundle")
    membership = bundle["membership"].long().numpy()
    full_edges = canonical_undirected(bundle["edge_index"], node_count)
    local_edges = induced_edges(full_edges, membership)
    missing = missing_degrees(full_edges, local_edges, node_count)
    ledgers = build_boundary_ledger_views(full_edges, membership, missing)
    device = torch.device(args.device)
    clients = build_federated_clients_from_ledgers(
        ledgers,
        features,
        labels,
        bundle["train_mask"].bool(),
        bundle["validation_mask"].bool(),
        bundle["test_mask"].bool(),
        args.order_count,
        device,
        uncertainty_recurrence="retained_mass",
    )
    full_shift, _ = row_normalized_shift(full_edges, node_count, device=device)

    torch.manual_seed(int(args.seed) * 1009 + 22)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) * 1009 + 22)
    initial_model = PolynomialSpectralClassifier(
        int(features.size(1)),
        args.hidden_dim,
        class_count,
        args.order_count,
        args.dropout,
        encoder_type=encoder_type,
        encoder_depth=encoder_depth,
        encoder_residual_scale=args.encoder_residual_scale,
        certificate_margin_mode="pairwise_difference",
    ).to(device)
    state = cpu_state(initial_model)
    sample_counts = [client.train_count for client in clients]
    best_state = None
    best_round = 0
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    history: list[dict] = []
    teacher_device = teacher_logits.to(device)

    for round_id in range(1, int(args.rounds) + 1):
        client_states = []
        for client in clients:
            client_states.append(train_client_external_teacher(
                state,
                client,
                teacher_device[client.global_nodes],
                input_dim=int(features.size(1)),
                hidden_dim=args.hidden_dim,
                num_classes=class_count,
                order_count=args.order_count,
                dropout=args.dropout,
                encoder_type=encoder_type,
                encoder_depth=encoder_depth,
                encoder_residual_scale=args.encoder_residual_scale,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=int(args.seed) * 100000 + round_id * 100 + client.client_id,
                device=device,
                loss_mode=loss_mode,
                distillation_weight=args.distillation_weight,
                distillation_temperature=args.distillation_temperature,
                cad_weight=args.cad_weight,
                certificate_margin=args.certificate_margin,
                propagation_mode=propagation_mode,
            ))
        state = weighted_average_states(client_states, sample_counts)
        validation = evaluate_external_teacher_state(
            state,
            clients,
            teacher_device,
            global_features=features.to(device),
            global_labels=labels.to(device),
            global_validation_mask=bundle["validation_mask"].bool().to(device),
            global_test_mask=bundle["test_mask"].bool().to(device),
            full_shift=full_shift,
            hidden_dim=args.hidden_dim,
            num_classes=class_count,
            order_count=args.order_count,
            dropout=args.dropout,
            encoder_type=encoder_type,
            encoder_depth=encoder_depth,
            encoder_residual_scale=args.encoder_residual_scale,
            device=device,
            include_test=False,
            certificate_kind=certificate_kind,
            propagation_mode=propagation_mode,
        )
        score = (
            validation["validation_certified_ratio"],
            validation["validation_safe_teacher_agreement_ratio"],
            validation["validation_safe_macro_f1"],
        )
        if score > best_score:
            best_score = score
            best_round = round_id
            best_state = {key: value.clone() for key, value in state.items()}
        history.append({
            "round": round_id,
            "validation_certified_ratio": validation["validation_certified_ratio"],
            "validation_safe_teacher_agreement_ratio": validation["validation_safe_teacher_agreement_ratio"],
            "validation_safe_macro_f1": validation["validation_safe_macro_f1"],
        })
        print(
            f"[{args.variant}/{run_name}] {round_id:03d}/{args.rounds} "
            f"cert={score[0]:.4f} agree={score[1]:.4f} "
            f"best={best_score[0]:.4f}@{best_round}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("E22 did not select a safe checkpoint")
    metrics = evaluate_external_teacher_state(
        best_state,
        clients,
        teacher_device,
        global_features=features.to(device),
        global_labels=labels.to(device),
        global_validation_mask=bundle["validation_mask"].bool().to(device),
        global_test_mask=bundle["test_mask"].bool().to(device),
        full_shift=full_shift,
        hidden_dim=args.hidden_dim,
        num_classes=class_count,
        order_count=args.order_count,
        dropout=args.dropout,
        encoder_type=encoder_type,
        encoder_depth=encoder_depth,
        encoder_residual_scale=args.encoder_residual_scale,
        device=device,
        include_test=args.evaluation_scope == "test_once",
        certificate_kind=certificate_kind,
        propagation_mode=propagation_mode,
    )
    state_path = output_dir / "safe_state.pt"
    torch.save(best_state, state_path)
    history_path = output_dir / "validation_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    record = {
        "protocol": args.protocol_name,
        "method": "FedBCert strong-utility teacher with completion-certified agreement distillation",
        "dataset": args.dataset,
        "partition": args.partition,
        "num_clients": int(args.num_clients),
        "seed": int(args.seed),
        "variant": args.variant,
        "encoder_type": encoder_type,
        "loss_mode": loss_mode,
        "propagation_mode": propagation_mode,
        "bundle_sha256": manifest["bundle_sha256"],
        "teacher_artifact_sha256": sha256(teacher_path),
        "teacher_source_commit": teacher_artifact["source_commit"],
        "teacher_selected_round": int(teacher_artifact["selected_round"]),
        "teacher_test_evaluation_count_before_wrapper": 0,
        "teacher_labels_in_artifact": bool(teacher_artifact["labels_included"]),
        "selection_metric": "validation certified ratio; safe-teacher agreement; safe validation Macro-F1 tie-break",
        "selected_round": int(best_round),
        "evaluation_scope": args.evaluation_scope,
        "test_labels_used_for_training_or_selection": False,
        "test_evaluation_count": int(args.evaluation_scope == "test_once"),
        "ordinary_prediction_contract": "exactly preserves frozen FedLap+ teacher argmax for every node",
        "certificate_scope": "fixed agreement shield only; utility-only nodes are explicitly uncertified",
        "information_interface": {
            "endpoint_blind_safe_expert": True,
            "strong_utility_teacher_requires_global_spectral_collaboration": True,
            "same_as_original_fedbcert_interface": False,
        },
        "hyperparameters": {
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "hidden_dim": int(args.hidden_dim),
            "order_count": int(args.order_count),
            "dropout": float(args.dropout),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "distillation_weight": float(args.distillation_weight),
            "distillation_temperature": float(args.distillation_temperature),
            "cad_weight": float(args.cad_weight),
            "certificate_margin": float(args.certificate_margin),
            "certificate_kind": certificate_kind,
        },
        "metrics": metrics,
        "safe_state_file": state_path.name,
        "safe_state_sha256": sha256(state_path),
        "validation_history_file": history_path.name,
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    complete_path.write_text(
        json.dumps({"complete": True, "metrics": result_path.name}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
