#!/usr/bin/env python3
"""Audit and summarize the frozen E27 ledger-granularity experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config", default="configs/fedbcert_e27_ledger_granularity_protocol.json"
    )
    value.add_argument("--require-complete", action="store_true")
    return value


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
    keyed_values: list[tuple[str, str, int, float]], seed: int = 270831
) -> tuple[float, float]:
    """Resample datasets, then partitions, then seeds within partitions."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for dataset, partition, _seed, value in keyed_values:
        grouped.setdefault(dataset, {}).setdefault(partition, []).append(float(value))
    rng = np.random.default_rng(seed)
    datasets = tuple(grouped)
    samples = []
    for _ in range(10000):
        draw: list[float] = []
        for dataset in rng.choice(datasets, size=len(datasets), replace=True):
            partitions = tuple(grouped[str(dataset)])
            for partition in rng.choice(
                partitions, size=len(partitions), replace=True
            ):
                values = np.asarray(
                    grouped[str(dataset)][str(partition)], dtype=float
                )
                draw.extend(
                    rng.choice(values, size=len(values), replace=True).tolist()
                )
        samples.append(float(np.mean(draw)))
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def main() -> None:
    args = parser().parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["result_root"])
    analysis = root / "analysis"
    modes = tuple(config["information_modes"])
    expected = {
        (dataset, partition, int(seed))
        for dataset in config["datasets"]
        for partition in config["partitions"]
        for seed in config["seeds"]
    }
    records: list[dict] = []
    found: set[tuple[str, str, int]] = set()
    errors: list[str] = []
    for path in (root / "units").glob("*/certificate_metrics.csv"):
        rows = read_csv(path)
        if len(rows) != len(modes) or {row["certificate_information"] for row in rows} != set(modes):
            errors.append(f"mode_set:{path}")
            continue
        key = (rows[0]["dataset"], rows[0]["partition"], int(rows[0]["seed"]))
        found.add(key)
        completion_path = path.with_name("run_complete.json")
        if not completion_path.exists():
            errors.append(f"missing_completion:{key}")
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        for name in (
            "no_training",
            "no_selection",
            "ordinary_logits_identical",
            "safe_upper_bounds_verified",
            "refinement_monotonicity_verified",
        ):
            if completion.get(name) is not True:
                errors.append(f"audit:{name}:{key}")
        if any(int(row["test_certified_completion_violations"]) for row in rows):
            errors.append(f"certificate_violation:{key}")
        records.extend(rows)
    errors.extend(f"missing:{key}" for key in sorted(expected - found))

    exact = {
        (row["dataset"], row["partition"], int(row["seed"])): row
        for row in records
        if row["certificate_information"] == "exact"
    }
    summary_rows: list[dict] = []
    for dataset in (*config["datasets"], "ALL"):
        subset = [row for row in records if dataset == "ALL" or row["dataset"] == dataset]
        for mode in modes:
            current = [row for row in subset if row["certificate_information"] == mode]
            if not current:
                continue
            deltas = []
            keyed_deltas: list[tuple[str, str, int, float]] = []
            for row in current:
                key = (row["dataset"], row["partition"], int(row["seed"]))
                delta = (
                    float(exact[key]["test_certified_ratio"])
                    - float(row["test_certified_ratio"])
                )
                deltas.append(delta)
                keyed_deltas.append((*key, delta))
            lower, upper = hierarchical_interval(keyed_deltas)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "certificate_information": mode,
                    "unit_count": len(current),
                    "mean_local_ledger_bits_per_node": float(
                        np.mean([float(row["local_ledger_bits_per_node"]) for row in current])
                    ),
                    "mean_certified_ratio": float(
                        np.mean([float(row["test_certified_ratio"]) for row in current])
                    ),
                    "mean_certified_accuracy": float(
                        np.mean([float(row["test_certified_accuracy"]) for row in current])
                    ),
                    "mean_boundary_certified_ratio": float(
                        np.mean([float(row["test_boundary_certified_ratio"]) for row in current])
                    ),
                    "mean_worst_client_certified_ratio": float(
                        np.mean([float(row["test_worst_client_certified_ratio"]) for row in current])
                    ),
                    "mean_representation_radius": float(
                        np.mean([float(row["test_mean_representation_radius"]) for row in current])
                    ),
                    "mean_verifier_wall_seconds": float(
                        np.mean([float(row["verifier_wall_seconds"]) for row in current])
                    ),
                    "exact_minus_mode_certified_ratio": float(np.mean(deltas)),
                    "delta_ci95_lower": lower,
                    "delta_ci95_upper": upper,
                    "delta_ci95_method": (
                        "10000-draw hierarchical percentile bootstrap: "
                        "dataset, partition within dataset, seed within partition"
                    ),
                }
            )

    overall = [row for row in summary_rows if row["dataset"] == "ALL"]
    by_dataset = [row for row in summary_rows if row["dataset"] != "ALL"]
    if records:
        write_csv(analysis / "paper_data_e27_ledger_granularity_long.csv", records)
        write_csv(analysis / "paper_table_e27_ledger_granularity_overall.csv", overall)
        write_csv(analysis / "paper_table_e27_ledger_granularity_by_dataset.csv", by_dataset)
    complete = not errors and found == expected
    summary = {
        "protocol": config["protocol"],
        "complete": complete,
        "decision": "frozen_information_audit_complete_no_selection" if complete else "audit_incomplete",
        "expected_units": len(expected),
        "found_units": len(found),
        "expected_metric_rows": len(expected) * len(modes),
        "found_metric_rows": len(records),
        "certificate_violations": sum(
            int(row["test_certified_completion_violations"]) for row in records
        ),
        "model_or_hyperparameter_selection_performed": False,
        "ordinary_logits_identical": not any(error.startswith("audit:ordinary_logits") for error in errors),
        "refinement_monotonicity_verified": not any(error.startswith("audit:refinement") for error in errors),
        "bootstrap": (
            "10000-draw hierarchical percentile bootstrap: dataset, partition "
            "within dataset, seed within partition"
        ),
        "errors": errors,
    }
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
