#!/usr/bin/env python3
"""Audit E18a validation-only runs and freeze one global nonzero-alpha point."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONFIG = Path("configs/fedbcert_e18_cone_protocol.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="results_boundary_certified/boundary_certified_spectral_e18_cone",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def mean(values) -> float:
    data = np.asarray(list(values), dtype=float)
    return float(data.mean()) if data.size else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    root = Path(args.root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    datasets = tuple(config["datasets"])
    seeds = tuple(int(seed) for seed in config["development_seeds"])
    variants = config["variants"]
    rows: list[dict] = []
    errors: list[str] = []
    units = 0

    required_metrics = {
        "validation_local_macro_f1",
        "validation_bound_tight_pairwise_certified_ratio",
        "validation_bound_cone_raw_certified_ratio",
        "validation_bound_cone_hybrid_certified_ratio",
        "validation_bound_tight_pairwise_certified_accuracy",
        "validation_bound_cone_raw_certified_accuracy",
        "validation_bound_cone_hybrid_certified_accuracy",
        "validation_bound_cone_raw_certified_completion_violations",
        "validation_bound_cone_hybrid_certified_completion_violations",
        "validation_logit_bound_violations",
        "validation_mean_absolute_completion_logit_change",
    }
    for variant, specification in variants.items():
        for dataset in datasets:
            for seed in seeds:
                label = f"{variant}:{dataset}:seed{seed}"
                run = root / "development" / variant / f"{dataset}_random_c3_seed{seed}"
                metric_path = run / "federated_metrics.csv"
                protocol_path = run / "protocol.json"
                complete_path = run / "run_complete.json"
                if not all(path.exists() for path in (metric_path, protocol_path, complete_path)):
                    errors.append(f"missing:{label}")
                    continue
                units += 1
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                protocol_args = protocol.get("arguments", {})
                checks = {
                    "scope": protocol.get("evaluation_scope") != "validation_only",
                    "test_emission": protocol.get("test_metrics_emitted") is not False,
                    "test_selection": protocol.get("test_labels_used_for_selection") is not False,
                    "recurrence": protocol.get("uncertainty_recurrence") != "retained_mass",
                    "margin": protocol.get("certificate_margin_mode") != "pairwise_difference",
                    "cone": protocol.get("cone_certificate_audit", {}).get("enabled") is not True,
                    "suite": protocol_args.get("suite") != "fedavg_only",
                    "encoder": protocol_args.get("encoder_type") != specification["encoder_type"],
                    "depth": int(protocol_args.get("encoder_depth", -1)) != int(specification["encoder_depth"]),
                    "alpha": abs(float(protocol.get("spectral_fusion_scale", -1.0)) - float(specification["alpha"])) > 1e-12,
                }
                for name, failed in checks.items():
                    if failed:
                        errors.append(f"{name}:{label}")
                with metric_path.open(encoding="utf-8-sig", newline="") as handle:
                    current = list(csv.DictReader(handle))
                if len(current) != 1 or current[0].get("method") != "fedavg":
                    errors.append(f"method_set:{label}")
                    continue
                if any(key.startswith("test_") for key in current[0]):
                    errors.append(f"test_column:{label}")
                missing = required_metrics - set(current[0])
                if missing:
                    errors.append(f"metric_columns:{label}:{sorted(missing)}")
                    continue
                rows.append({"variant": variant, **current[0]})

    expected_units = int(config["expected_development_runner_units"])
    expected_rows = int(config["expected_development_metric_rows"])
    complete = units == expected_units and len(rows) == expected_rows and not errors
    if args.require_complete and not complete:
        raise SystemExit(
            f"E18a incomplete or unsafe: units={units}/{expected_units}; "
            f"rows={len(rows)}/{expected_rows}; errors={errors}"
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    feature = {
        (row["dataset"], int(row["seed"])): float(row["validation_local_macro_f1"])
        for row in grouped.get("feature_alpha0", [])
    }
    rule = config["selection_rule"]
    screen: list[dict] = []
    by_dataset: list[dict] = []
    for variant, specification in variants.items():
        current = grouped.get(variant, [])
        dataset_hybrid: dict[str, float] = {}
        for dataset in datasets:
            subset = [row for row in current if row["dataset"] == dataset]
            dataset_hybrid[dataset] = mean(
                float(row["validation_bound_cone_hybrid_certified_ratio"])
                for row in subset
            )
            by_dataset.append({
                "variant": variant,
                "alpha": specification["alpha"],
                "dataset": dataset,
                "runs": len(subset),
                "validation_macro_f1": mean(float(row["validation_local_macro_f1"]) for row in subset),
                "symmetric_certified_ratio": mean(float(row["validation_bound_tight_pairwise_certified_ratio"]) for row in subset),
                "cone_raw_certified_ratio": mean(float(row["validation_bound_cone_raw_certified_ratio"]) for row in subset),
                "cone_hybrid_certified_ratio": dataset_hybrid[dataset],
                "symmetric_certified_accuracy": mean(float(row["validation_bound_tight_pairwise_certified_accuracy"]) for row in subset),
                "cone_raw_certified_accuracy": mean(float(row["validation_bound_cone_raw_certified_accuracy"]) for row in subset),
                "cone_hybrid_certified_accuracy": mean(float(row["validation_bound_cone_hybrid_certified_accuracy"]) for row in subset),
            })
        f1 = mean(float(row["validation_local_macro_f1"]) for row in current)
        feature_difference = mean(
            float(row["validation_local_macro_f1"])
            - feature[(row["dataset"], int(row["seed"]))]
            for row in current
            if (row["dataset"], int(row["seed"])) in feature
        )
        symmetric = mean(float(row["validation_bound_tight_pairwise_certified_ratio"]) for row in current)
        cone_raw = mean(float(row["validation_bound_cone_raw_certified_ratio"]) for row in current)
        hybrid = mean(float(row["validation_bound_cone_hybrid_certified_ratio"]) for row in current)
        symmetric_accuracy = mean(float(row["validation_bound_tight_pairwise_certified_accuracy"]) for row in current)
        cone_raw_accuracy = mean(float(row["validation_bound_cone_raw_certified_accuracy"]) for row in current)
        hybrid_accuracy = mean(float(row["validation_bound_cone_hybrid_certified_accuracy"]) for row in current)
        violations = sum(
            int(row["validation_bound_cone_raw_certified_completion_violations"])
            + int(row["validation_bound_cone_hybrid_certified_completion_violations"])
            + int(row["validation_logit_bound_violations"])
            for row in current
        )
        completion_logit_change = mean(
            float(row["validation_mean_absolute_completion_logit_change"])
            for row in current
        )
        feasible = (
            float(specification["alpha"]) > 0.0
            and hybrid >= float(rule["mean_validation_hybrid_certified_ratio_min"])
            and min(dataset_hybrid.values(), default=float("-inf")) >= float(rule["minimum_dataset_mean_validation_hybrid_certified_ratio"])
            and hybrid - symmetric >= float(rule["mean_hybrid_minus_symmetric_certified_ratio_min"])
            and feature_difference >= float(rule["mean_validation_macro_f1_noninferiority_vs_feature"])
            and completion_logit_change >= float(rule["mean_absolute_completion_logit_change_min"])
            and violations <= int(rule["maximum_validation_certificate_violations"])
        )
        screen.append({
            "variant": variant,
            "alpha": specification["alpha"],
            "runs": len(current),
            "validation_macro_f1": f1,
            "macro_f1_difference_vs_feature": feature_difference,
            "symmetric_certified_ratio": symmetric,
            "cone_raw_certified_ratio": cone_raw,
            "cone_hybrid_certified_ratio": hybrid,
            "hybrid_minus_symmetric_certified_ratio": hybrid - symmetric,
            "symmetric_certified_accuracy": symmetric_accuracy,
            "cone_raw_certified_accuracy": cone_raw_accuracy,
            "cone_hybrid_certified_accuracy": hybrid_accuracy,
            "hybrid_minus_symmetric_certified_accuracy": hybrid_accuracy - symmetric_accuracy,
            "minimum_dataset_hybrid_certified_ratio": min(dataset_hybrid.values(), default=float("nan")),
            "certificate_violations": violations,
            "mean_absolute_completion_logit_change": completion_logit_change,
            "feasible": feasible,
        })

    feasible = [row for row in screen if row["feasible"]]
    feasible.sort(key=lambda row: (-float(row["validation_macro_f1"]), -float(row["cone_hybrid_certified_ratio"]), float(row["alpha"])))
    selected = feasible[0] if complete and feasible else None
    summary = {
        "protocol": config["protocol"],
        "complete": complete,
        "decision": "proceed_to_e18b" if selected else "stop_after_e18a" if complete else "incomplete",
        "selected_variant": selected["variant"] if selected else None,
        "selected_alpha": float(selected["alpha"]) if selected else None,
        "selected_certificate": "cone_hybrid" if selected else None,
        "selection_constraints": rule,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "expected_runner_units": expected_units,
        "found_runner_units": units,
        "expected_metric_rows": expected_rows,
        "found_metric_rows": len(rows),
        "test_metrics_used_or_emitted": any(error.startswith(("scope:", "test_emission:", "test_selection:", "test_column:")) for error in errors),
        "reserved_confirmation_seeds": config["reserved_confirmation_seeds"],
        "errors": errors,
    }
    write_csv(analysis / "paper_table_e18a_screen.csv", screen)
    write_csv(analysis / "paper_table_e18a_by_dataset.csv", by_dataset)
    write_csv(analysis / "paper_data_e18a_validation_metrics.csv", rows)
    (analysis / "frozen_cone_policy.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
