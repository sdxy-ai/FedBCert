#!/usr/bin/env python3
"""Analyze the frozen E24 graph-invariant cover controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fedbcert_e24_invariant_cover_protocol.json"
DEFAULT_ROOT = ROOT / "results_boundary_certified/boundary_certified_spectral_e24_invariant_cover"
METHODS = [
    "feature_kd_cad",
    "feature_invariant_union",
    "identity_invariant_cover",
    "observed_graph_cover",
]
LABELS = {
    "feature_kd_cad": "Feature KD+CAD",
    "feature_invariant_union": "Feature CE $\\cup$ Feature KD+CAD",
    "identity_invariant_cover": "$P=I$ invariant cover",
    "observed_graph_cover": "Observed-$P$ cover",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--require-complete", action="store_true")
    return p


def hierarchical_difference_ci(
    records: list[dict], method_a: str, method_b: str, *, seed: int, draws: int = 10000
) -> list[float]:
    datasets = sorted({record["dataset"] for record in records})
    lookup = {
        (record["dataset"], record["partition"], int(record["seed"])): record
        for record in records
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        values: list[float] = []
        for dataset_index in rng.integers(0, len(datasets), size=len(datasets)):
            dataset = datasets[int(dataset_index)]
            partitions = sorted({
                record["partition"] for record in records if record["dataset"] == dataset
            })
            for partition_index in rng.integers(0, len(partitions), size=len(partitions)):
                partition = partitions[int(partition_index)]
                seeds = sorted({
                    int(record["seed"])
                    for record in records
                    if record["dataset"] == dataset and record["partition"] == partition
                })
                for seed_index in rng.integers(0, len(seeds), size=len(seeds)):
                    record = lookup[(dataset, partition, seeds[int(seed_index)])]
                    values.append(
                        float(record["metrics"][method_a]["test_certified_ratio"])
                        - float(record["metrics"][method_b]["test_certified_ratio"])
                    )
        samples[draw] = float(np.mean(values))
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    datasets = sorted({row["dataset"] for row in rows})
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    x = np.arange(len(datasets), dtype=float)
    width = 0.19
    colors = ["#999999", "#E69F00", "#56B4E9", "#0072B2"]
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    for index, method in enumerate(METHODS):
        values = [100.0 * float(lookup[(dataset, method)]["test_certified_ratio"]) for dataset in datasets]
        ax.bar(x + (index - 1.5) * width, values, width, label=LABELS[method], color=colors[index])
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Certified ratio (%)")
    ax.set_title("Frozen invariant-cover controls")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"figure_e24_invariant_cover.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    records: list[dict] = []
    errors: list[str] = []
    for dataset in config["datasets"]:
        for partition in config["partitions"]:
            for seed in config["confirmation_seeds"]:
                run_name = f"{dataset}_{partition}_c3_seed{seed}"
                path = args.result_root / "evaluation" / run_name / "metrics.json"
                if not path.exists():
                    errors.append(f"missing: {path}")
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                checks = {
                    "test_evaluation_count": record.get("test_evaluation_count") == 1,
                    "teacher_test_evaluation_count_before_cover": record.get("teacher_test_evaluation_count_before_cover") == 0,
                    "test_labels_used_for_training_or_selection": record.get("test_labels_used_for_training_or_selection") is False,
                    "ordinary_teacher_prediction_preserved_for_all_methods": record.get("ordinary_teacher_prediction_preserved_for_all_methods") is True,
                    "all_covers_are_exact_component_unions": record.get("all_covers_are_exact_component_unions") is True,
                    "component_training_budget_and_teacher_matching": record.get("component_training_budget_and_teacher_matching_verified") is True,
                    "spectral_parameter_matching": record.get("spectral_observed_identity_parameter_matching_verified") is True,
                    "selection_or_reselection": record.get("selection_or_reselection_performed") is False,
                }
                errors.extend(f"{name}: {path}" for name, valid in checks.items() if not valid)
                records.append(record)

    expected = int(config["expected_cover_evaluation_units"])
    teacher_units = sum(
        (args.result_root / "teachers" / f"{dataset}_{partition}_c3_seed{seed}" / "run_complete.json").exists()
        for dataset in config["datasets"]
        for partition in config["partitions"]
        for seed in config["confirmation_seeds"]
    )
    component_units = sum(
        (args.result_root / "components" / component / f"{dataset}_{partition}_c3_seed{seed}" / "metrics.json").exists()
        for component in config["components"]
        for dataset in config["datasets"]
        for partition in config["partitions"]
        for seed in config["confirmation_seeds"]
    )
    if teacher_units != int(config["expected_teacher_units"]):
        errors.append(f"teacher units: {teacher_units}")
    if component_units != int(config["expected_component_training_units"]):
        errors.append(f"component units: {component_units}")
    complete = len(records) == expected and not errors

    rows: list[dict] = []
    for dataset in config["datasets"]:
        current = [record for record in records if record["dataset"] == dataset]
        for method in METHODS:
            if current:
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "units": len(current),
                    "test_utility_macro_f1": float(np.mean([
                        record["metrics"][method]["test_utility_macro_f1"] for record in current
                    ])),
                    "test_certified_ratio": float(np.mean([
                        record["metrics"][method]["test_certified_ratio"] for record in current
                    ])),
                    "test_certified_accuracy": float(np.mean([
                        record["metrics"][method]["test_certified_accuracy"] for record in current
                    ])),
                })
    overall: list[dict] = []
    for method in METHODS:
        if records:
            overall.append({
                "method": method,
                "units": len(records),
                "test_utility_macro_f1": float(np.mean([
                    record["metrics"][method]["test_utility_macro_f1"] for record in records
                ])),
                "test_certified_ratio": float(np.mean([
                    record["metrics"][method]["test_certified_ratio"] for record in records
                ])),
                "test_certified_accuracy": float(np.mean([
                    record["metrics"][method]["test_certified_accuracy"] for record in records
                ])),
            })

    def mean_difference(a: str, b: str) -> float | None:
        return (
            float(np.mean([
                record["metrics"][a]["test_certified_ratio"]
                - record["metrics"][b]["test_certified_ratio"]
                for record in records
            ]))
            if records else None
        )

    observed_feature = mean_difference("observed_graph_cover", "feature_invariant_union")
    observed_identity = mean_difference("observed_graph_cover", "identity_invariant_cover")
    ci_feature = hierarchical_difference_ci(
        records, "observed_graph_cover", "feature_invariant_union", seed=24128
    ) if records else [None, None]
    ci_identity = hierarchical_difference_ci(
        records, "observed_graph_cover", "identity_invariant_cover", seed=24129
    ) if records else [None, None]
    violations = sum(
        int(metrics["test_certified_completion_violations"])
        + int(metrics["test_logit_bound_violations"])
        for record in records
        for metrics in record["metrics"].values()
    )
    gate = config["claim_gate"]
    supported = (
        complete
        and violations <= int(gate["maximum_certificate_violations"])
        and observed_feature is not None
        and observed_identity is not None
        and observed_feature > float(gate["observed_minus_feature_invariant_cover_mean_min"])
        and observed_identity > float(gate["observed_minus_identity_invariant_cover_mean_min"])
        and ci_feature[0] > 0.0
        and ci_identity[0] > 0.0
    )
    decision = (
        "graph_dependent_cover_value_supported"
        if supported
        else config["decision_if_gate_fails"]
        if complete
        else "invariant_cover_confirmation_incomplete"
    )
    analysis = args.result_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / "paper_table_e24_by_dataset.csv", rows)
    write_csv(analysis / "paper_table_e24_overall.csv", overall)
    if rows:
        plot(rows, analysis / "figures")
    summary = {
        "protocol": config["protocol"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "complete": complete,
        "decision": decision,
        "model_or_hyperparameter_selection_performed": False,
        "expected_teacher_units": int(config["expected_teacher_units"]),
        "found_teacher_units": teacher_units,
        "expected_component_training_units": int(config["expected_component_training_units"]),
        "found_component_training_units": component_units,
        "expected_cover_evaluation_units": expected,
        "found_cover_evaluation_units": len(records),
        "certificate_violations": violations,
        "mean_observed_minus_feature_invariant_cover_certified_ratio": observed_feature,
        "observed_minus_feature_invariant_cover_certified_ratio_95ci": ci_feature,
        "mean_observed_minus_identity_invariant_cover_certified_ratio": observed_identity,
        "observed_minus_identity_invariant_cover_certified_ratio_95ci": ci_identity,
        "spectral_observed_identity_parameter_matching_verified": all(
            record.get("spectral_observed_identity_parameter_matching_verified") is True
            for record in records
        ),
        "component_training_budget_and_teacher_matching_verified": all(
            record.get("component_training_budget_and_teacher_matching_verified") is True
            for record in records
        ),
        "test_labels_used_for_selection": False,
        "errors": errors,
    }
    (analysis / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
