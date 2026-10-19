#!/usr/bin/env python3
"""Evaluate a frozen union cover of feature-safe and spectral-safe experts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openfgl.flcore.fedbcert import (
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
    "feature_ce": ("feature_only_mlp", 2),
    "feature_kd_cad": ("feature_only_mlp", 2),
    "spectral_cad": ("linear_relu", 1),
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


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    prediction = logits[mask].argmax(dim=1).detach().cpu().numpy()
    truth = labels[mask].detach().cpu().numpy()
    return float(f1_score(truth, prediction, average="macro", zero_division=0))


def main() -> None:
    args = parser().parse_args()
    run_name = f"{args.dataset}_{args.partition}_c{args.num_clients}_seed{args.seed}"
    bundle_dir = (REPOSITORY_ROOT / args.bundle_root / run_name).resolve()
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_path = bundle_dir / manifest["bundle"]
    if sha256(bundle_path) != manifest["bundle_sha256"]:
        raise RuntimeError("E23 immutable bundle SHA-256 mismatch")
    teacher_dir = (REPOSITORY_ROOT / args.teacher_root / run_name).resolve()
    teacher_path = teacher_dir / "utility_logits.pt"
    teacher_record = json.loads(
        (teacher_dir / "fedlap_plus_official_metrics.json").read_text(encoding="utf-8")
    )
    if int(teacher_record.get("test_evaluation_count", -1)) != 0:
        raise RuntimeError("E23 teacher must be materialized without test evaluation")
    teacher_artifact = torch.load(teacher_path, map_location="cpu", weights_only=True)
    if teacher_artifact["bundle_sha256"] != manifest["bundle_sha256"]:
        raise RuntimeError("E23 teacher/bundle mismatch")

    output_dir = (REPOSITORY_ROOT / args.output_root / run_name).resolve()
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
    teacher_logits = teacher_artifact["logits"].float()
    if teacher_logits.shape != (node_count, class_count):
        raise RuntimeError("E23 teacher shape mismatch")
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
    teacher_logits = teacher_logits.to(device)
    global_features = features.to(device)
    global_labels = labels.to(device)
    test_mask = bundle["test_mask"].bool().to(device)

    models: dict[str, PolynomialSpectralClassifier] = {}
    state_hashes: dict[str, str] = {}
    for name, (encoder_type, encoder_depth) in COMPONENTS.items():
        state_path = (
            (REPOSITORY_ROOT / args.component_root).resolve()
            / name / run_name / "safe_state.pt"
        )
        record_path = state_path.parent / "metrics.json"
        if not state_path.exists() or not record_path.exists():
            raise FileNotFoundError(f"missing E23 component: {state_path.parent}")
        component_record = json.loads(record_path.read_text(encoding="utf-8"))
        if component_record["bundle_sha256"] != manifest["bundle_sha256"]:
            raise RuntimeError(f"E23 {name} component/bundle mismatch")
        if component_record["evaluation_scope"] != "validation_only":
            raise RuntimeError(f"E23 {name} component touched test metrics")
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
        model.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=True), strict=True
        )
        model.eval()
        models[name] = model
        state_hashes[name] = sha256(state_path)

    safe_logits = {
        name: torch.empty((node_count, class_count), device=device)
        for name in COMPONENTS
    }
    bounds = {name: torch.empty_like(teacher_logits) for name in COMPONENTS}
    shields = {
        name: torch.zeros(node_count, dtype=torch.bool, device=device)
        for name in COMPONENTS
    }
    with torch.no_grad():
        for client in clients:
            nodes = client.global_nodes
            for name, model in models.items():
                stats = external_teacher_agreement_statistics(
                    model,
                    client.features,
                    client.shift,
                    client.uncertainty_basis,
                    teacher_logits[nodes],
                    certificate_kind=(
                        "cone_hybrid" if name == "spectral_cad" else "symmetric"
                    ),
                )
                safe_logits[name][nodes] = stats.safe_logits
                bounds[name][nodes] = stats.certificate.class_logit_bounds
                shields[name][nodes] = stats.shield

        completion_safe = {
            "feature_ce": safe_logits["feature_ce"].clone(),
            "feature_kd_cad": safe_logits["feature_kd_cad"].clone(),
        }
        spectral_model = models["spectral_cad"]
        hidden = torch.empty(
            (node_count, int(spectral_model.classifier.in_features)),
            dtype=teacher_logits.dtype,
            device=device,
        )
        for client in clients:
            hidden[client.global_nodes] = spectral_model.bounded_encoding(client.features)
        completion_safe["spectral_cad"] = spectral_model.classifier(
            spectral_model.filter(hidden, full_shift)
        )

    methods: dict[str, dict[str, torch.Tensor]] = {}
    infinite_bounds = torch.full_like(teacher_logits, float("inf"))
    for name in COMPONENTS:
        shield = shields[name]
        methods[name] = {
            "certified": shield,
            "local": torch.where(shield[:, None], safe_logits[name], teacher_logits),
            "completion": torch.where(
                shield[:, None], completion_safe[name], teacher_logits
            ),
            "bounds": torch.where(shield[:, None], bounds[name], infinite_bounds),
        }

    feature_route = shields["feature_kd_cad"]
    spectral_route = (~feature_route) & shields["spectral_cad"]
    cover = feature_route | spectral_route
    cover_local = torch.where(
        feature_route[:, None],
        safe_logits["feature_kd_cad"],
        torch.where(spectral_route[:, None], safe_logits["spectral_cad"], teacher_logits),
    )
    cover_completion = torch.where(
        feature_route[:, None],
        completion_safe["feature_kd_cad"],
        torch.where(
            spectral_route[:, None], completion_safe["spectral_cad"], teacher_logits
        ),
    )
    cover_bounds = torch.where(
        feature_route[:, None],
        bounds["feature_kd_cad"],
        torch.where(spectral_route[:, None], bounds["spectral_cad"], infinite_bounds),
    )
    methods["safe_cover"] = {
        "certified": cover,
        "local": cover_local,
        "completion": cover_completion,
        "bounds": cover_bounds,
    }
    teacher_prediction = teacher_logits.argmax(dim=1)
    correct = teacher_prediction == global_labels
    metric_rows: dict[str, dict] = {}
    for name, values in methods.items():
        prediction = values["local"].argmax(dim=1)
        if not torch.equal(prediction, teacher_prediction):
            raise RuntimeError(f"E23 {name} changed a teacher prediction")
        completion_prediction = values["completion"].argmax(dim=1)
        certified = values["certified"]
        deviation = (values["completion"] - values["local"]).abs()
        tolerance = 1e-6 + 1e-5 * torch.maximum(
            values["completion"].abs(), values["local"].abs()
        )
        active_classes = (certified & test_mask)[:, None].expand_as(deviation)
        count = max(int(test_mask.sum().item()), 1)
        metric_rows[name] = {
            "test_utility_macro_f1": macro_f1(teacher_logits, global_labels, test_mask),
            "test_certified_ratio": float((certified & test_mask).sum().item() / count),
            "test_certified_accuracy": float(
                (certified & correct & test_mask).sum().item() / count
            ),
            "test_certified_completion_violations": int(
                (certified & test_mask & (prediction != completion_prediction)).sum().item()
            ),
            "test_logit_bound_violations": int(
                (
                    active_classes
                    & (deviation > values["bounds"] + tolerance)
                ).sum().item()
            ),
        }
    exact_union = torch.equal(
        methods["safe_cover"]["certified"],
        shields["feature_kd_cad"] | shields["spectral_cad"],
    )
    record = {
        "protocol": "FedBCert E23 frozen multi-safe-expert certificate cover",
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
        "cover_is_exact_component_union": exact_union,
        "cover_priority": ["feature_kd_cad", "spectral_cad", "utility_teacher"],
        "component_state_sha256": state_hashes,
        "cover_source_counts_all_nodes": {
            "feature_safe": int(feature_route.sum().item()),
            "spectral_safe_additional": int(spectral_route.sum().item()),
            "utility_unshielded": int((~cover).sum().item()),
        },
        "metrics": metric_rows,
    }
    if not exact_union:
        raise RuntimeError("E23 safe cover is not the exact component union")
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_complete.json").write_text(
        json.dumps({"complete": True, "metrics": result_path.name}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
