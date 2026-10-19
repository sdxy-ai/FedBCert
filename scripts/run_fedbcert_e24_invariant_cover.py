#!/usr/bin/env python3
"""Evaluate frozen graph-invariant and observed-graph safe-expert covers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openfgl.flcore.fedbcert import (  # noqa: E402
    PolynomialSpectralClassifier,
    build_boundary_ledger_views,
    build_federated_clients_from_ledgers,
    canonical_undirected,
    external_teacher_agreement_statistics,
    induced_edges,
    missing_degrees,
    row_normalized_shift,
)


COMPONENTS = {
    # The E22 wrapper records its frozen default cone-hybrid audit for feature
    # experts. Their completion radius is identically zero, so cone-hybrid and
    # symmetric acceptance are mathematically identical; retaining the recorded
    # kind keeps checkpoint selection and final auditing byte-for-byte aligned.
    "feature_ce": ("feature_only_mlp", 2, "cone_hybrid", "observed"),
    "feature_kd_cad": ("feature_only_mlp", 2, "cone_hybrid", "observed"),
    "spectral_identity_cad": ("linear_relu", 1, "symmetric", "identity"),
    "spectral_cad": ("linear_relu", 1, "cone_hybrid", "observed"),
}
LOSS_MODES = {
    "feature_ce": "ce",
    "feature_kd_cad": "kd_cad",
    "spectral_identity_cad": "cad",
    "spectral_cad": "cad",
}
COMMON_TRAINING_BUDGET = {
    "rounds": 100,
    "local_epochs": 1,
    "hidden_dim": 64,
    "order_count": 3,
    "dropout": 0.2,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "distillation_weight": 1.0,
    "distillation_temperature": 2.0,
    "cad_weight": 0.25,
    "certificate_margin": 0.05,
}
COVERS = {
    "feature_kd_cad": ("feature_kd_cad",),
    "feature_invariant_union": ("feature_kd_cad", "feature_ce"),
    "identity_invariant_cover": ("feature_kd_cad", "spectral_identity_cad"),
    "observed_graph_cover": ("feature_kd_cad", "spectral_cad"),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["Cora", "CiteSeer", "PubMed", "Photo"], required=True)
    p.add_argument("--partition", choices=["metis", "random"], required=True)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--bundle-root", required=True)
    p.add_argument("--teacher-root", required=True)
    p.add_argument("--component-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--order-count", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--encoder-residual-scale", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_shift(reference: torch.Tensor) -> torch.Tensor:
    count = int(reference.size(0))
    diagonal = torch.arange(count, device=reference.device)
    return torch.sparse_coo_tensor(
        torch.stack((diagonal, diagonal)),
        torch.ones(count, dtype=reference.dtype, device=reference.device),
        (count, count),
        device=reference.device,
    ).coalesce()


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return float(f1_score(
        labels[mask].detach().cpu().numpy(),
        logits[mask].argmax(dim=1).detach().cpu().numpy(),
        average="macro",
        zero_division=0,
    ))


def main() -> None:
    args = parser().parse_args()
    run_name = f"{args.dataset}_{args.partition}_c{args.num_clients}_seed{args.seed}"
    bundle_dir = (ROOT / args.bundle_root / run_name).resolve()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    bundle_path = bundle_dir / manifest["bundle"]
    if sha256(bundle_path) != manifest["bundle_sha256"]:
        raise RuntimeError("E24 immutable bundle SHA-256 mismatch")
    teacher_dir = (ROOT / args.teacher_root / run_name).resolve()
    teacher_path = teacher_dir / "utility_logits.pt"
    teacher_record = json.loads(
        (teacher_dir / "fedlap_plus_official_metrics.json").read_text(encoding="utf-8")
    )
    if int(teacher_record.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError("E24 teacher must be materialized without test evaluation")
    teacher_artifact = torch.load(teacher_path, map_location="cpu", weights_only=True)
    if teacher_artifact["bundle_sha256"] != manifest["bundle_sha256"]:
        raise RuntimeError("E24 teacher/bundle mismatch")

    output_dir = (ROOT / args.output_root / run_name).resolve()
    result_path = output_dir / "metrics.json"
    if result_path.exists() and not args.force:
        print(f"[SKIP] complete: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    features = bundle["x"].float()
    labels = bundle["y"].reshape(-1).long()
    membership = bundle["membership"].long().numpy()
    node_count = int(features.size(0))
    class_count = int(manifest["class_count"])
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
    teacher_logits = teacher_artifact["logits"].float().to(device)
    if teacher_logits.shape != (node_count, class_count):
        raise RuntimeError("E24 teacher shape mismatch")
    global_labels = labels.to(device)
    test_mask = bundle["test_mask"].bool().to(device)

    models: dict[str, PolynomialSpectralClassifier] = {}
    state_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    for name, (encoder_type, encoder_depth, _, propagation) in COMPONENTS.items():
        state_path = (ROOT / args.component_root / name / run_name / "safe_state.pt").resolve()
        record_path = state_path.parent / "metrics.json"
        if not state_path.exists() or not record_path.exists():
            raise FileNotFoundError(f"missing E24 component: {state_path.parent}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("protocol") != "FedBCert E24 frozen invariant-cover component":
            raise RuntimeError(f"E24 {name} protocol mismatch")
        identity = (
            record.get("dataset") == args.dataset
            and record.get("partition") == args.partition
            and int(record.get("num_clients", -1)) == int(args.num_clients)
            and int(record.get("seed", -1)) == int(args.seed)
            and record.get("variant") == name
        )
        if not identity:
            raise RuntimeError(f"E24 {name} run identity mismatch")
        if record.get("bundle_sha256") != manifest["bundle_sha256"]:
            raise RuntimeError(f"E24 {name} component/bundle mismatch")
        if record.get("teacher_artifact_sha256") != sha256(teacher_path):
            raise RuntimeError(f"E24 {name} component/teacher mismatch")
        if record.get("evaluation_scope") != "validation_only":
            raise RuntimeError(f"E24 {name} component touched test metrics")
        if int(record.get("test_evaluation_count", -1)) != 0:
            raise RuntimeError(f"E24 {name} component evaluated test labels")
        if record.get("test_labels_used_for_training_or_selection") is not False:
            raise RuntimeError(f"E24 {name} component leaked test labels")
        if record.get("loss_mode") != LOSS_MODES[name]:
            raise RuntimeError(f"E24 {name} loss mismatch")
        if record.get("propagation_mode", "observed") != propagation:
            raise RuntimeError(f"E24 {name} propagation mismatch")
        hyperparameters = record.get("hyperparameters", {})
        if any(hyperparameters.get(key) != value for key, value in COMMON_TRAINING_BUDGET.items()):
            raise RuntimeError(f"E24 {name} training-budget mismatch")
        expected_certificate = COMPONENTS[name][2]
        if hyperparameters.get("certificate_kind") != expected_certificate:
            raise RuntimeError(f"E24 {name} certificate mismatch")
        model = PolynomialSpectralClassifier(
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
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        models[name] = model
        state_hashes[name] = sha256(state_path)
        parameter_counts[name] = sum(value.numel() for value in model.parameters())

    safe_logits = {name: torch.empty_like(teacher_logits) for name in COMPONENTS}
    bounds = {name: torch.empty_like(teacher_logits) for name in COMPONENTS}
    shields = {
        name: torch.zeros(node_count, dtype=torch.bool, device=device)
        for name in COMPONENTS
    }
    with torch.no_grad():
        for client in clients:
            nodes = client.global_nodes
            for name, model in models.items():
                _, _, certificate_kind, propagation = COMPONENTS[name]
                if propagation == "identity":
                    shift = identity_shift(client.shift)
                    basis = [torch.zeros_like(value) for value in client.uncertainty_basis]
                else:
                    shift = client.shift
                    basis = client.uncertainty_basis
                stats = external_teacher_agreement_statistics(
                    model,
                    client.features,
                    shift,
                    basis,
                    teacher_logits[nodes],
                    certificate_kind=certificate_kind,
                )
                safe_logits[name][nodes] = stats.safe_logits
                bounds[name][nodes] = stats.certificate.class_logit_bounds
                shields[name][nodes] = stats.shield

        completion_safe = {
            "feature_ce": safe_logits["feature_ce"].clone(),
            "feature_kd_cad": safe_logits["feature_kd_cad"].clone(),
            "spectral_identity_cad": safe_logits["spectral_identity_cad"].clone(),
        }
        observed_model = models["spectral_cad"]
        hidden = torch.empty(
            (node_count, int(observed_model.classifier.in_features)),
            dtype=teacher_logits.dtype,
            device=device,
        )
        for client in clients:
            hidden[client.global_nodes] = observed_model.bounded_encoding(client.features)
        completion_safe["spectral_cad"] = observed_model.classifier(
            observed_model.filter(hidden, full_shift)
        )

    infinite = torch.full_like(teacher_logits, float("inf"))
    teacher_prediction = teacher_logits.argmax(dim=1)
    correct = teacher_prediction == global_labels
    metric_rows: dict[str, dict] = {}
    source_counts: dict[str, dict[str, int]] = {}
    union_checks: dict[str, bool] = {}
    for method, components in COVERS.items():
        first = components[0]
        first_route = shields[first]
        if len(components) == 1:
            second = None
            second_route = torch.zeros_like(first_route)
        else:
            second = components[1]
            second_route = (~first_route) & shields[second]
        certified = first_route | second_route
        local = torch.where(first_route[:, None], safe_logits[first], teacher_logits)
        completion = torch.where(first_route[:, None], completion_safe[first], teacher_logits)
        class_bounds = torch.where(first_route[:, None], bounds[first], infinite)
        if second is not None:
            local = torch.where(second_route[:, None], safe_logits[second], local)
            completion = torch.where(second_route[:, None], completion_safe[second], completion)
            class_bounds = torch.where(second_route[:, None], bounds[second], class_bounds)
        if not torch.equal(local.argmax(dim=1), teacher_prediction):
            raise RuntimeError(f"E24 {method} changed a teacher prediction")
        expected_union = shields[first] if second is None else shields[first] | shields[second]
        union_checks[method] = torch.equal(certified, expected_union)
        completion_prediction = completion.argmax(dim=1)
        deviation = (completion - local).abs()
        tolerance = 1e-6 + 1e-5 * torch.maximum(completion.abs(), local.abs())
        active_classes = (certified & test_mask)[:, None].expand_as(deviation)
        count = max(int(test_mask.sum().item()), 1)
        metric_rows[method] = {
            "test_utility_macro_f1": macro_f1(teacher_logits, global_labels, test_mask),
            "test_certified_ratio": float((certified & test_mask).sum().item() / count),
            "test_certified_accuracy": float((certified & correct & test_mask).sum().item() / count),
            "test_certified_completion_violations": int(
                (certified & test_mask & (teacher_prediction != completion_prediction)).sum().item()
            ),
            "test_logit_bound_violations": int(
                (active_classes & (deviation > class_bounds + tolerance)).sum().item()
            ),
        }
        source_counts[method] = {
            first: int(first_route.sum().item()),
            **({second: int(second_route.sum().item())} if second is not None else {}),
            "utility_teacher": int((~certified).sum().item()),
        }

    spectral_parameter_match = (
        parameter_counts["spectral_identity_cad"] == parameter_counts["spectral_cad"]
    )
    if not spectral_parameter_match:
        raise RuntimeError("E24 observed and identity spectral experts are not parameter matched")
    if not all(union_checks.values()):
        raise RuntimeError("E24 reported cover is not an exact component union")
    record = {
        "protocol": "FedBCert E24 invariant-cover independent control",
        "dataset": args.dataset,
        "partition": args.partition,
        "num_clients": int(args.num_clients),
        "seed": int(args.seed),
        "bundle_sha256": manifest["bundle_sha256"],
        "teacher_artifact_sha256": sha256(teacher_path),
        "teacher_test_evaluation_count_before_cover": 0,
        "test_evaluation_count": 1,
        "test_labels_used_for_training_or_selection": False,
        "ordinary_teacher_prediction_preserved_for_all_methods": True,
        "all_covers_are_exact_component_unions": True,
        "component_training_budget_and_teacher_matching_verified": True,
        "spectral_observed_identity_parameter_matching_verified": spectral_parameter_match,
        "selection_or_reselection_performed": False,
        "component_state_sha256": state_hashes,
        "component_parameter_counts": parameter_counts,
        "cover_source_counts_all_nodes": source_counts,
        "metrics": metric_rows,
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_complete.json").write_text(
        json.dumps({"complete": True, "metrics": result_path.name}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
