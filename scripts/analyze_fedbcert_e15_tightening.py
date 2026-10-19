#!/usr/bin/env python3
"""Fixed-predictor E15 ablation of recurrence and multiclass-bound tightening."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = (
    "tight_pairwise",
    "legacy_pairwise",
    "tight_classwise",
    "legacy_classwise",
)
METRICS = ("certified_ratio", "certified_accuracy")
ALPHA = {
    "feature_alpha0": 0.0,
    "dual_alpha010": 0.1,
    "dual_alpha025": 0.25,
    "dual_alpha050": 0.5,
    "dual_alpha100": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="results_boundary_certified/boundary_certified_spectral_e15",
    )
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=151500)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hierarchical_interval(
    values: dict[str, dict[str, list[float]]], *, draws: int, seed: int
) -> tuple[float, float]:
    datasets = sorted(values)
    generator = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        sampled_datasets = generator.choice(datasets, len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            configurations = sorted(values[str(dataset)])
            sampled_configurations = generator.choice(
                configurations, len(configurations), replace=True
            )
            configuration_means = []
            for configuration in sampled_configurations:
                current = np.asarray(
                    values[str(dataset)][str(configuration)], dtype=float
                )
                configuration_means.append(
                    float(generator.choice(current, current.size, replace=True).mean())
                )
            dataset_means.append(float(np.mean(configuration_means)))
        samples[draw] = float(np.mean(dataset_means))
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def nested_differences(rows: list[dict], left: str, right: str):
    nested: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    flat: list[float] = []
    for row in rows:
        value = float(row[left]) - float(row[right])
        configuration = f"c{row['num_clients']}_{row['partition']}"
        nested[row["dataset"]][configuration].append(value)
        flat.append(value)
    return nested, flat


def main() -> None:
    args = parse_args()
    analysis = Path(args.root) / "analysis"
    rows = read_csv(analysis / "paper_data_e15_all_metrics.csv")
    variants = sorted(
        {row["variant"] for row in rows},
        key=lambda value: ALPHA[value],
    )
    output: list[dict] = []
    by_dataset: list[dict] = []
    comparison_seed = int(args.seed)
    max_main_mismatch = 0.0

    for variant in variants:
        current = [row for row in rows if row["variant"] == variant]
        alpha = ALPHA[variant]
        for metric in METRICS:
            columns = {
                method: f"test_bound_{method}_{metric}" for method in METHODS
            }
            main_column = f"test_{metric}"
            max_main_mismatch = max(
                max_main_mismatch,
                max(
                    abs(float(row[main_column]) - float(row[columns["tight_pairwise"]]))
                    for row in current
                ),
            )
            method_means = {
                method: float(np.mean([float(row[column]) for row in current]))
                for method, column in columns.items()
            }
            comparisons = {
                "retained_mass_gain": (
                    columns["tight_pairwise"], columns["legacy_pairwise"]
                ),
                "pairwise_margin_gain": (
                    columns["tight_pairwise"], columns["tight_classwise"]
                ),
                "combined_gain": (
                    columns["tight_pairwise"], columns["legacy_classwise"]
                ),
            }
            result = {
                "variant": variant,
                "alpha": alpha,
                "metric": metric,
                "runs": len(current),
                **method_means,
            }
            for name, (left, right) in comparisons.items():
                nested, flat = nested_differences(current, left, right)
                low, high = hierarchical_interval(
                    nested, draws=args.draws, seed=comparison_seed
                )
                comparison_seed += 1
                result[name] = float(np.mean(flat))
                result[name + "_ci_low"] = low
                result[name + "_ci_high"] = high
                result[name + "_positive_fraction"] = float(
                    np.mean(np.asarray(flat) > 0.0)
                )
            output.append(result)

            if metric == "certified_ratio":
                for dataset in sorted({row["dataset"] for row in current}):
                    subset = [row for row in current if row["dataset"] == dataset]
                    by_dataset.append({
                        "variant": variant,
                        "alpha": alpha,
                        "dataset": dataset,
                        "runs": len(subset),
                        "retained_mass_gain": float(np.mean([
                            float(row[columns["tight_pairwise"]])
                            - float(row[columns["legacy_pairwise"]])
                            for row in subset
                        ])),
                        "pairwise_margin_gain": float(np.mean([
                            float(row[columns["tight_pairwise"]])
                            - float(row[columns["tight_classwise"]])
                            for row in subset
                        ])),
                        "combined_gain": float(np.mean([
                            float(row[columns["tight_pairwise"]])
                            - float(row[columns["legacy_classwise"]])
                            for row in subset
                        ])),
                    })

    if max_main_mismatch > 1e-12:
        raise RuntimeError(
            "reported certificate does not match tight-pairwise ablation: "
            f"{max_main_mismatch}"
        )
    write_csv(analysis / "paper_table_e15_certificate_tightening.csv", output)
    write_csv(
        analysis / "paper_table_e15_certificate_tightening_by_dataset.csv",
        by_dataset,
    )
    print(
        f"[DONE] {len(output)} fixed-predictor tightening rows; "
        f"main mismatch={max_main_mismatch:.3g}"
    )


if __name__ == "__main__":
    main()
