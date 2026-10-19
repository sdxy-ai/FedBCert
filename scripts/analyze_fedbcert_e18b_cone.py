#!/usr/bin/env python3
"""Audit E18b and report paired utility and certificate confidence intervals."""
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
    parser.add_argument("--root", default="results_boundary_certified/boundary_certified_spectral_e18_cone")
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


def hierarchical_interval(values: dict[str, dict[str, list[float]]], *, draws: int, seed: int) -> tuple[float, float]:
    datasets = sorted(values)
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_datasets = generator.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            configurations = sorted(values[str(dataset)])
            sampled_configurations = generator.choice(configurations, size=len(configurations), replace=True)
            configuration_means = []
            for configuration in sampled_configurations:
                current = np.asarray(values[str(dataset)][str(configuration)], dtype=float)
                configuration_means.append(float(generator.choice(current, size=current.size, replace=True).mean()))
            dataset_means.append(float(np.mean(configuration_means)))
        samples[draw] = float(np.mean(dataset_means))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    root = Path(args.root)
    analysis = root / "analysis"
    policy_path = analysis / "frozen_cone_policy.json"
    if not policy_path.exists():
        raise SystemExit("missing E18a frozen cone policy")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    selected = policy.get("selected_variant")
    if policy.get("decision") != "proceed_to_e18b" or not selected:
        raise SystemExit("E18a did not authorize E18b")
    if policy.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest():
        raise SystemExit("E18 protocol changed after development selection")
    selected_spec = config["variants"][selected]
    matrix = config["confirmation_matrix"]
    datasets = tuple(config["datasets"])
    clients = tuple(int(value) for value in matrix["num_clients"])
    partitions = tuple(matrix["partitions"])
    seeds = tuple(int(value) for value in matrix["seeds"])
    variants = ("feature_alpha0", selected)
    rows: list[dict] = []
    errors: list[str] = []
    units = 0
    for variant in variants:
        specification = config["variants"][variant]
        for dataset in datasets:
            for client_count in clients:
                for partition in partitions:
                    for seed in seeds:
                        label = f"{variant}:{dataset}:{partition}:c{client_count}:seed{seed}"
                        run = root / "confirmation" / variant / f"{dataset}_{partition}_c{client_count}_seed{seed}"
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
                            "scope": protocol.get("evaluation_scope") != "full",
                            "test_selection": protocol.get("test_labels_used_for_selection") is not False,
                            "recurrence": protocol.get("uncertainty_recurrence") != "retained_mass",
                            "margin": protocol.get("certificate_margin_mode") != "pairwise_difference",
                            "cone": protocol.get("cone_certificate_audit", {}).get("enabled") is not True,
                            "suite": protocol_args.get("suite") != "fedavg_only",
                            "encoder": protocol_args.get("encoder_type") != specification["encoder_type"],
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
                        rows.append({"variant": variant, **current[0]})

    expected_units = int(matrix["expected_runner_units"])
    expected_rows = int(matrix["expected_metric_rows"])
    complete = units == expected_units and len(rows) == expected_rows and not errors
    if args.require_complete and not complete:
        raise SystemExit(f"E18b incomplete: units={units}/{expected_units}; rows={len(rows)}/{expected_rows}; errors={errors}")

    keyed: dict[str, dict[tuple[str, int, str, int], dict]] = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], int(row["num_clients"]), row["partition"], int(row["seed"]))
        keyed[row["variant"]][key] = row
    paired: list[dict] = []
    for key, selected_row in keyed.get(selected, {}).items():
        feature_row = keyed.get("feature_alpha0", {}).get(key)
        if feature_row is None:
            errors.append(f"unpaired:{key}")
            continue
        dataset, client_count, partition, seed = key
        symmetric = float(selected_row["test_bound_tight_pairwise_certified_ratio"])
        cone = float(selected_row["test_bound_cone_hybrid_certified_ratio"])
        paired.append({
            "dataset": dataset,
            "num_clients": client_count,
            "partition": partition,
            "seed": seed,
            "selected_variant": selected,
            "alpha": selected_spec["alpha"],
            "selected_macro_f1": float(selected_row["test_local_macro_f1"]),
            "feature_macro_f1": float(feature_row["test_local_macro_f1"]),
            "macro_f1_difference": float(selected_row["test_local_macro_f1"]) - float(feature_row["test_local_macro_f1"]),
            "selected_accuracy": float(selected_row["test_local_accuracy"]),
            "feature_accuracy": float(feature_row["test_local_accuracy"]),
            "accuracy_difference": float(selected_row["test_local_accuracy"]) - float(feature_row["test_local_accuracy"]),
            "feature_certified_ratio": float(feature_row["test_bound_cone_hybrid_certified_ratio"]),
            "feature_certified_accuracy": float(feature_row["test_bound_cone_hybrid_certified_accuracy"]),
            "symmetric_certified_ratio": symmetric,
            "cone_raw_certified_ratio": float(selected_row["test_bound_cone_raw_certified_ratio"]),
            "cone_hybrid_certified_ratio": cone,
            "hybrid_minus_symmetric_certified_ratio": cone - symmetric,
            "symmetric_certified_accuracy": float(selected_row["test_bound_tight_pairwise_certified_accuracy"]),
            "cone_raw_certified_accuracy": float(selected_row["test_bound_cone_raw_certified_accuracy"]),
            "cone_hybrid_certified_accuracy": float(selected_row["test_bound_cone_hybrid_certified_accuracy"]),
            "hybrid_minus_symmetric_certified_accuracy": float(selected_row["test_bound_cone_hybrid_certified_accuracy"]) - float(selected_row["test_bound_tight_pairwise_certified_accuracy"]),
        })
    complete = complete and len(paired) * 2 == expected_rows and not errors

    bootstrap = config["bootstrap"]
    difference_metrics = (
        "macro_f1_difference",
        "hybrid_minus_symmetric_certified_ratio",
        "hybrid_minus_symmetric_certified_accuracy",
        "accuracy_difference",
    )
    overall: dict[str, object] = {"selected_variant": selected, "alpha": selected_spec["alpha"], "paired_units": len(paired)}
    intervals: dict[str, tuple[float, float]] = {}
    for index, metric in enumerate(difference_metrics):
        nested: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in paired:
            configuration = f"c{row['num_clients']}_{row['partition']}"
            nested[row["dataset"]][configuration].append(float(row[metric]))
        interval = hierarchical_interval(nested, draws=int(bootstrap["draws"]), seed=int(bootstrap["seed"]) + index + 1)
        intervals[metric] = interval
        overall[metric] = mean(float(row[metric]) for row in paired)
        overall[metric + "_ci_low"] = interval[0]
        overall[metric + "_ci_high"] = interval[1]
    for metric in (
        "selected_macro_f1",
        "feature_macro_f1",
        "selected_accuracy",
        "feature_accuracy",
        "feature_certified_ratio",
        "feature_certified_accuracy",
        "symmetric_certified_ratio",
        "cone_raw_certified_ratio",
        "cone_hybrid_certified_ratio",
        "symmetric_certified_accuracy",
        "cone_raw_certified_accuracy",
        "cone_hybrid_certified_accuracy",
    ):
        overall[metric] = mean(float(row[metric]) for row in paired)

    selected_rows = list(keyed.get(selected, {}).values())
    violations = sum(
        int(row["test_bound_cone_raw_certified_completion_violations"])
        + int(row["test_bound_cone_hybrid_certified_completion_violations"])
        + int(row["test_logit_bound_violations"])
        for row in selected_rows
    )
    claims = config["confirmation_claims"]
    claim_supported = (
        complete
        and violations <= int(claims["maximum_certificate_violations"])
        and intervals.get("hybrid_minus_symmetric_certified_ratio", (-np.inf, np.inf))[0] > float(claims["hybrid_minus_symmetric_certified_ratio_ci_low_min"])
        and intervals.get("macro_f1_difference", (-np.inf, np.inf))[0] >= float(claims["selected_minus_feature_macro_f1_ci_low_min"])
    )
    by_dataset = []
    for dataset in datasets:
        current = [row for row in paired if row["dataset"] == dataset]
        by_dataset.append({"dataset": dataset, "paired_units": len(current), **{metric: mean(float(row[metric]) for row in current) for metric in ("selected_macro_f1", "feature_macro_f1", "selected_accuracy", "feature_accuracy", "feature_certified_ratio", "feature_certified_accuracy", *difference_metrics, "symmetric_certified_ratio", "cone_raw_certified_ratio", "cone_hybrid_certified_ratio", "symmetric_certified_accuracy", "cone_raw_certified_accuracy", "cone_hybrid_certified_accuracy")}})
    summary = {
        "protocol": "FedBCert E18b frozen ReLU-cone independent confirmation",
        "complete": complete,
        "decision": "confirmation_complete_claim_supported" if claim_supported else "confirmation_complete_claim_not_supported" if complete and violations == 0 else "confirmation_audit_failed",
        "selected_variant": selected,
        "selected_alpha": selected_spec["alpha"],
        "operating_point_reselected_after_confirmation": False,
        "test_labels_used_for_selection": False,
        "certificate_violations": violations,
        "expected_runner_units": expected_units,
        "found_runner_units": units,
        "expected_metric_rows": expected_rows,
        "found_metric_rows": len(rows),
        "errors": errors,
    }
    write_csv(analysis / "paper_table_e18b_overall.csv", [overall])
    write_csv(analysis / "paper_table_e18b_by_dataset.csv", by_dataset)
    write_csv(analysis / "paper_data_e18b_paired.csv", paired)
    (analysis / "confirmation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and (not complete or violations != 0):
        raise SystemExit("E18b audit failed")


if __name__ == "__main__":
    main()
