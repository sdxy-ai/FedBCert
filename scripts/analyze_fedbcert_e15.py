#!/usr/bin/env python3
"""Audit and summarize the fixed, selection-free E15 Pareto confirmation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONFIG = Path("configs/fedbcert_e15_protocol.json")
FEATURE = "feature_alpha0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="results_boundary_certified/boundary_certified_spectral_e15",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def mean(values) -> float:
    data = list(values)
    return float(np.mean(np.asarray(data, dtype=float))) if data else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hierarchical_interval(
    values: dict[str, dict[str, list[float]]], *, draws: int, seed: int
) -> tuple[float, float]:
    """Resample dataset, client/partition configuration, then seed."""
    datasets = sorted(values)
    if not datasets:
        return float("nan"), float("nan")
    generator = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        sampled_datasets = generator.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            configurations = sorted(values[str(dataset)])
            sampled_configurations = generator.choice(
                configurations, size=len(configurations), replace=True
            )
            configuration_means = []
            for configuration in sampled_configurations:
                current = np.asarray(
                    values[str(dataset)][str(configuration)], dtype=float
                )
                configuration_means.append(
                    float(generator.choice(current, size=current.size, replace=True).mean())
                )
            dataset_means.append(float(np.mean(configuration_means)))
        samples[draw] = float(np.mean(dataset_means))
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = Path(args.root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    datasets = tuple(config["datasets"])
    clients = tuple(int(value) for value in config["num_clients"])
    partitions = tuple(config["partitions"])
    seeds = tuple(int(value) for value in config["confirmation_seeds"])
    variants = config["variants"]
    rows: list[dict] = []
    errors: list[str] = []
    units = 0

    required_metrics = {
        "test_local_accuracy",
        "test_local_macro_f1",
        "test_worst_client_local_macro_f1",
        "test_certified_ratio",
        "test_certified_accuracy",
        "test_certified_accuracy_over_clean_accuracy",
        "test_worst_client_certified_ratio",
        "test_worst_client_certified_accuracy",
        "test_boundary_local_macro_f1",
        "test_boundary_certified_ratio",
        "test_boundary_certified_accuracy",
        "test_interior_local_macro_f1",
        "test_interior_certified_ratio",
        "test_interior_certified_accuracy",
        "test_bound_tight_pairwise_certified_ratio",
        "test_bound_tight_pairwise_certified_accuracy",
        "test_bound_tight_classwise_certified_ratio",
        "test_bound_tight_classwise_certified_accuracy",
        "test_bound_legacy_pairwise_certified_ratio",
        "test_bound_legacy_pairwise_certified_accuracy",
        "test_bound_legacy_classwise_certified_ratio",
        "test_bound_legacy_classwise_certified_accuracy",
        "model_state_bytes",
        "total_model_communication_bytes",
        "method_wall_clock_seconds",
    }

    for variant, specification in variants.items():
        for dataset in datasets:
            for client_count in clients:
                for partition in partitions:
                    for seed in seeds:
                        label = (
                            f"{variant}:{dataset}:{partition}:"
                            f"c{client_count}:seed{seed}"
                        )
                        run = (
                            root / "confirmation" / variant
                            / f"{dataset}_{partition}_c{client_count}_seed{seed}"
                        )
                        metric_path = run / "federated_metrics.csv"
                        protocol_path = run / "protocol.json"
                        complete_path = run / "run_complete.json"
                        if not all(
                            path.exists()
                            for path in (metric_path, protocol_path, complete_path)
                        ):
                            errors.append(f"missing:{label}")
                            continue
                        units += 1
                        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                        protocol_args = protocol.get("arguments", {})
                        checks = {
                            "scope": protocol.get("evaluation_scope") != "full",
                            "test_emission": protocol.get("test_metrics_emitted") is not True,
                            "test_selection": protocol.get("test_labels_used_for_selection") is not False,
                            "recurrence": protocol.get("uncertainty_recurrence") != "retained_mass",
                            "margin": protocol.get("certificate_margin_mode") != "pairwise_difference",
                            "encoder": protocol_args.get("encoder_type")
                            != specification["encoder_type"],
                            "depth": int(protocol_args.get("encoder_depth", -1))
                            != int(specification["encoder_depth"]),
                            "alpha": abs(
                                float(protocol.get("spectral_fusion_scale", -1.0))
                                - float(specification["alpha"])
                            ) > 1e-12,
                            "suite": protocol_args.get("suite") != "fedavg_only",
                        }
                        for name, failed in checks.items():
                            if failed:
                                errors.append(f"{name}:{label}")
                        with metric_path.open(encoding="utf-8-sig", newline="") as handle:
                            current = list(csv.DictReader(handle))
                        if len(current) != 1 or current[0].get("method") != "fedavg":
                            errors.append(f"method_set:{label}")
                            continue
                        if not required_metrics.issubset(current[0]):
                            errors.append(f"metric_schema:{label}")
                        rows.append({"variant": variant, **current[0]})

    expected_units = int(config["expected_runner_units"])
    expected_rows = int(config["expected_metric_rows"])
    complete = units == expected_units and len(rows) == expected_rows and not errors
    if args.require_complete and not complete:
        raise SystemExit(
            f"E15 incomplete or unsafe: units={units}/{expected_units}; "
            f"rows={len(rows)}/{expected_rows}; errors={errors}"
        )

    indexed = {
        (
            row["variant"], row["dataset"], int(row["num_clients"]),
            row["partition"], int(row["seed"]),
        ): row
        for row in rows
    }
    metric_names = (
        "test_local_accuracy",
        "test_local_macro_f1",
        "test_worst_client_local_macro_f1",
        "test_certified_ratio",
        "test_certified_accuracy",
        "test_certified_accuracy_over_clean_accuracy",
        "test_worst_client_certified_ratio",
        "test_worst_client_certified_accuracy",
        "test_boundary_local_macro_f1",
        "test_boundary_certified_ratio",
        "test_boundary_certified_accuracy",
        "test_interior_local_macro_f1",
        "test_interior_certified_ratio",
        "test_interior_certified_accuracy",
        "test_bound_tight_pairwise_certified_ratio",
        "test_bound_tight_pairwise_certified_accuracy",
        "test_bound_tight_classwise_certified_ratio",
        "test_bound_tight_classwise_certified_accuracy",
        "test_bound_legacy_pairwise_certified_ratio",
        "test_bound_legacy_pairwise_certified_accuracy",
        "test_bound_legacy_classwise_certified_ratio",
        "test_bound_legacy_classwise_certified_accuracy",
        "model_state_bytes",
        "total_model_communication_bytes",
        "method_wall_clock_seconds",
    )
    main_table: list[dict] = []
    for variant, specification in variants.items():
        current = [row for row in rows if row["variant"] == variant]
        main_table.append({
            "variant": variant,
            "alpha": specification["alpha"],
            "runs": len(current),
            **{
                metric: mean(float(row[metric]) for row in current)
                for metric in metric_names
            },
        })

    comparison_metrics = (
        "test_local_macro_f1",
        "test_worst_client_local_macro_f1",
        "test_certified_ratio",
        "test_certified_accuracy",
    )
    comparisons: list[dict] = []
    by_dataset: list[dict] = []
    draws = int(config["bootstrap"]["draws"])
    bootstrap_seed = int(config["bootstrap"]["seed"])
    comparison_index = 0
    for variant, specification in variants.items():
        if variant == FEATURE:
            continue
        for metric in comparison_metrics:
            nested: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            values: list[float] = []
            for dataset in datasets:
                dataset_values: list[float] = []
                for client_count in clients:
                    for partition in partitions:
                        configuration = f"c{client_count}_{partition}"
                        for seed in seeds:
                            left = indexed[(variant, dataset, client_count, partition, seed)]
                            right = indexed[(FEATURE, dataset, client_count, partition, seed)]
                            value = float(left[metric]) - float(right[metric])
                            values.append(value)
                            dataset_values.append(value)
                            nested[dataset][configuration].append(value)
                if metric == "test_local_macro_f1":
                    by_dataset.append({
                        "variant": variant,
                        "alpha": specification["alpha"],
                        "dataset": dataset,
                        "paired_units": len(dataset_values),
                        "mean_f1_difference_vs_feature": mean(dataset_values),
                    })
            interval = hierarchical_interval(
                nested,
                draws=draws,
                seed=bootstrap_seed + comparison_index,
            )
            comparison_index += 1
            comparisons.append({
                "variant": variant,
                "alpha": specification["alpha"],
                "metric": metric,
                "paired_units": len(values),
                "mean_difference_vs_feature": mean(values),
                "hierarchical_ci_low": interval[0],
                "hierarchical_ci_high": interval[1],
            })

    completion_violations = sum(
        int(row["test_certified_completion_violations"]) for row in rows
    )
    raw_logit_bound_violations = sum(
        int(row["test_logit_bound_violations"]) for row in rows
    )
    numerical_reaudit_path = analysis / "numerical_reaudit.json"
    numerical_reaudit = None
    if numerical_reaudit_path.exists():
        numerical_reaudit = json.loads(
            numerical_reaudit_path.read_text(encoding="utf-8")
        )
    numerical_reaudit_applied = bool(
        numerical_reaudit
        and numerical_reaudit.get("complete") is True
        and int(numerical_reaudit.get(
            "raw_reported_logit_bound_violations", -1
        )) == raw_logit_bound_violations
        and int(numerical_reaudit.get(
            "reproduced_historical_logit_bound_violations", -1
        )) == raw_logit_bound_violations
        and int(numerical_reaudit.get(
            "revised_dtype_aware_logit_bound_violations", -1
        )) == 0
        and int(numerical_reaudit.get(
            "float64_spectral_logit_bound_violations", -1
        )) == 0
        and numerical_reaudit.get(
            "all_historical_flags_had_zero_analytic_bound"
        ) is True
        and int(numerical_reaudit.get(
            "certified_completion_violations", -1
        )) == 0
        and int(numerical_reaudit.get("certified_prediction_flips", -1)) == 0
        and numerical_reaudit.get("training_or_state_modified") is False
        and numerical_reaudit.get(
            "logits_predictions_radii_or_certified_nodes_modified"
        ) is False
    )
    revised_logit_bound_violations = (
        int(numerical_reaudit["float64_spectral_logit_bound_violations"])
        if numerical_reaudit_applied
        else raw_logit_bound_violations
    )
    violations = completion_violations + revised_logit_bound_violations
    feature_min = min(
        float(row["test_certified_ratio"])
        for row in rows
        if row["variant"] == FEATURE
    ) if rows else float("nan")
    audit_passed = (
        complete
        and violations <= int(
            config["required_audits"]["maximum_certificate_violations"]
        )
        and feature_min >= float(
            config["required_audits"]["pure_feature_certified_ratio_min"]
        )
    )
    summary = {
        "protocol": config["protocol"],
        "complete": complete,
        "decision": (
            (
                "confirmation_complete_after_float64_structural_reaudit"
                if numerical_reaudit_applied
                else "confirmation_complete_no_model_selection"
            )
            if audit_passed
            else "confirmation_audit_failed"
        ),
        "expected_runner_units": expected_units,
        "found_runner_units": units,
        "expected_metric_rows": expected_rows,
        "found_metric_rows": len(rows),
        "certificate_violations": violations,
        "certified_completion_violations": completion_violations,
        "raw_reported_logit_bound_violations": raw_logit_bound_violations,
        "revised_dtype_aware_logit_bound_violations": (
            int(numerical_reaudit.get(
                "revised_dtype_aware_logit_bound_violations", -1
            ))
            if numerical_reaudit_applied
            else raw_logit_bound_violations
        ),
        "float64_spectral_logit_bound_violations": (
            revised_logit_bound_violations
        ),
        "numerical_reaudit_applied": numerical_reaudit_applied,
        "pure_feature_min_certified_ratio": feature_min,
        "selected_alpha": None,
        "test_labels_used_for_selection": False,
        "untouched_e14b_seeds": config["untouched_e14b_seeds"],
        "errors": errors,
    }
    write_csv(analysis / "paper_table_e15_frontier.csv", main_table)
    write_csv(analysis / "paper_table_e15_hierarchical_comparisons.csv", comparisons)
    write_csv(analysis / "paper_table_e15_by_dataset.csv", by_dataset)
    write_csv(analysis / "paper_data_e15_all_metrics.csv", rows)
    (analysis / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
