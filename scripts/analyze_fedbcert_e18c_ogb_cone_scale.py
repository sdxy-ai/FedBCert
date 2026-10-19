#!/usr/bin/env python3
"""Audit the frozen E18c cone-hybrid ogbn-arxiv scale experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CONFIG = Path("configs/fedbcert_e18c_ogb_cone_scale_protocol.json")
PARENT_CONFIG = Path("configs/fedbcert_e18_cone_protocol.json")
E18_ANALYSIS = Path(
    "results_boundary_certified/boundary_certified_spectral_e18_cone/analysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=(
            "results_boundary_certified/"
            "boundary_certified_spectral_e18c_ogb_cone_scale"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def mean(values) -> float:
    data = np.asarray(list(values), dtype=float)
    return float(data.mean()) if data.size else float("nan")


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def paired_interval(
    values: list[float], *, draws: int, seed: int
) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    samples = generator.choice(data, size=(draws, data.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


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
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_hash = hashlib.sha256(PARENT_CONFIG.read_bytes()).hexdigest()
    root = Path(args.root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    if parent_hash != config["parent_e18_config_sha256"]:
        errors.append("parent_e18_protocol_hash")
    policy_path = E18_ANALYSIS / "frozen_cone_policy.json"
    confirmation_path = E18_ANALYSIS / "confirmation_summary.json"
    if not policy_path.exists():
        errors.append("missing_e18_policy")
        policy = {}
    else:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not confirmation_path.exists():
        errors.append("missing_e18b_confirmation")
        confirmation = {}
    else:
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if policy.get("decision") != "proceed_to_e18b":
        errors.append("e18_policy_decision")
    if policy.get("selected_variant") != "dual_alpha025":
        errors.append("e18_selected_variant")
    if abs(float(policy.get("selected_alpha", -1.0)) - 0.25) > 1e-12:
        errors.append("e18_selected_alpha")
    if confirmation.get("decision") != config["required_e18b_decision"]:
        errors.append("e18b_confirmation_decision")
    if confirmation.get("operating_point_reselected_after_confirmation") is not False:
        errors.append("e18b_reselection")

    variants = config["variants"]
    seeds = tuple(int(value) for value in config["seeds"])
    rows: list[dict] = []
    protocols: list[dict] = []
    units = 0
    node_count = None
    for variant, specification in variants.items():
        for seed in seeds:
            label = f"{variant}:seed{seed}"
            run = root / "runs" / variant / f"ogbn-arxiv_random_c3_seed{seed}"
            metric_path = run / "federated_metrics.csv"
            protocol_path = run / "protocol.json"
            complete_path = run / "run_complete.json"
            membership_path = run / "partition_membership.csv"
            required = (metric_path, protocol_path, complete_path, membership_path)
            if not all(path.exists() for path in required):
                errors.append(f"missing:{label}")
                continue
            units += 1
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol_args = protocol.get("arguments", {})
            official = protocol.get("immutable_split_bundle_audit", {}).get(
                "official_dataset_split"
            )
            checks = {
                "protocol_name": protocol_args.get("protocol_name") != config["protocol"],
                "scope": protocol.get("evaluation_scope") != "full",
                "test_selection": protocol.get("test_labels_used_for_selection") is not False,
                "official_split": official is not True,
                "dataset": protocol_args.get("dataset") != config["dataset"],
                "partition": protocol_args.get("partition") != config["partition"],
                "clients": int(protocol_args.get("num_clients", -1))
                != int(config["num_clients"]),
                "seed": int(protocol_args.get("seed", -1)) != seed,
                "suite": protocol_args.get("suite") != "fedavg_only",
                "recurrence": protocol.get("uncertainty_recurrence")
                != config["fixed_hyperparameters"]["uncertainty_recurrence"],
                "margin": protocol.get("certificate_margin_mode")
                != config["fixed_hyperparameters"]["certificate_margin_mode"],
                "cone": protocol.get("cone_certificate_audit", {}).get("enabled")
                is not True,
                "encoder": protocol_args.get("encoder_type")
                != specification["encoder_type"],
                "encoder_depth": int(protocol_args.get("encoder_depth", -1))
                != int(specification["encoder_depth"]),
                "alpha": abs(
                    float(protocol.get("spectral_fusion_scale", -1.0))
                    - float(specification["alpha"])
                )
                > 1e-12,
            }
            for name, failed in checks.items():
                if failed:
                    errors.append(f"{name}:{label}")
            with metric_path.open(encoding="utf-8-sig", newline="") as handle:
                current = list(csv.DictReader(handle))
            if len(current) != 1 or current[0].get("method") != "fedavg":
                errors.append(f"method_set:{label}")
                continue
            with membership_path.open(encoding="utf-8") as handle:
                current_node_count = sum(1 for _ in handle) - 1
            if node_count is None:
                node_count = current_node_count
            elif current_node_count != node_count:
                errors.append(f"node_count:{label}")
            rows.append({"variant": variant, **current[0]})
            protocols.append({"variant": variant, "seed": seed, **protocol})

    expected_units = int(config["expected_runner_units"])
    expected_rows = int(config["expected_metric_rows"])
    violations = sum(
        int(row["test_bound_cone_raw_certified_completion_violations"])
        + int(row["test_bound_cone_hybrid_certified_completion_violations"])
        + int(row["test_logit_bound_violations"])
        for row in rows
    )
    feature_min = min(
        (
            float(row["test_bound_cone_hybrid_certified_ratio"])
            for row in rows
            if row["variant"] == "feature_alpha0"
        ),
        default=float("nan"),
    )
    complete = (
        units == expected_units
        and len(rows) == expected_rows
        and not errors
        and violations
        <= int(config["required_audits"]["maximum_certificate_violations"])
        and feature_min
        >= float(config["required_audits"]["pure_feature_certified_ratio_min"])
    )

    keyed = {
        (row["variant"], int(row["seed"])): row
        for row in rows
    }
    paired: list[dict] = []
    for seed in seeds:
        feature = keyed.get(("feature_alpha0", seed))
        selected = keyed.get(("dual_alpha025", seed))
        if feature is None or selected is None:
            errors.append(f"unpaired:seed{seed}")
            continue
        symmetric_ratio = float(
            selected["test_bound_tight_pairwise_certified_ratio"]
        )
        cone_ratio = float(selected["test_bound_cone_hybrid_certified_ratio"])
        symmetric_accuracy = float(
            selected["test_bound_tight_pairwise_certified_accuracy"]
        )
        cone_accuracy = float(
            selected["test_bound_cone_hybrid_certified_accuracy"]
        )
        paired.append(
            {
                "seed": seed,
                "selected_macro_f1": float(selected["test_local_macro_f1"]),
                "feature_macro_f1": float(feature["test_local_macro_f1"]),
                "selected_minus_feature_macro_f1": float(
                    selected["test_local_macro_f1"]
                )
                - float(feature["test_local_macro_f1"]),
                "selected_worst_client_macro_f1": float(
                    selected["test_worst_client_local_macro_f1"]
                ),
                "feature_worst_client_macro_f1": float(
                    feature["test_worst_client_local_macro_f1"]
                ),
                "selected_minus_feature_worst_client_macro_f1": float(
                    selected["test_worst_client_local_macro_f1"]
                )
                - float(feature["test_worst_client_local_macro_f1"]),
                "symmetric_certified_ratio": symmetric_ratio,
                "cone_raw_certified_ratio": float(
                    selected["test_bound_cone_raw_certified_ratio"]
                ),
                "cone_hybrid_certified_ratio": cone_ratio,
                "hybrid_minus_symmetric_certified_ratio": cone_ratio
                - symmetric_ratio,
                "symmetric_certified_accuracy": symmetric_accuracy,
                "cone_raw_certified_accuracy": float(
                    selected["test_bound_cone_raw_certified_accuracy"]
                ),
                "cone_hybrid_certified_accuracy": cone_accuracy,
                "hybrid_minus_symmetric_certified_accuracy": cone_accuracy
                - symmetric_accuracy,
            }
        )
    complete = complete and len(paired) == len(seeds) and not errors

    table: list[dict] = []
    for variant, specification in variants.items():
        current = [row for row in rows if row["variant"] == variant]
        current_protocols = [
            row for row in protocols if row["variant"] == variant
        ]
        table.append(
            {
                "variant": variant,
                "alpha": specification["alpha"],
                "runs": len(current),
                "node_count": node_count,
                "test_pooled_macro_f1": mean(
                    float(row["test_local_macro_f1"]) for row in current
                ),
                "test_worst_client_macro_f1": mean(
                    float(row["test_worst_client_local_macro_f1"])
                    for row in current
                ),
                "symmetric_certified_ratio": mean(
                    float(row["test_bound_tight_pairwise_certified_ratio"])
                    for row in current
                ),
                "cone_raw_certified_ratio": mean(
                    float(row["test_bound_cone_raw_certified_ratio"])
                    for row in current
                ),
                "cone_hybrid_certified_ratio": mean(
                    float(row["test_bound_cone_hybrid_certified_ratio"])
                    for row in current
                ),
                "symmetric_certified_accuracy": mean(
                    float(row["test_bound_tight_pairwise_certified_accuracy"])
                    for row in current
                ),
                "cone_raw_certified_accuracy": mean(
                    float(row["test_bound_cone_raw_certified_accuracy"])
                    for row in current
                ),
                "cone_hybrid_certified_accuracy": mean(
                    float(row["test_bound_cone_hybrid_certified_accuracy"])
                    for row in current
                ),
                "mean_method_wall_seconds": mean(
                    float(row["method_wall_clock_seconds"]) for row in current
                ),
                "mean_total_wall_seconds": mean(
                    float(row["wall_clock_seconds"]) for row in current_protocols
                ),
                "mean_peak_cuda_memory_bytes": mean(
                    float(row.get("peak_cuda_memory_bytes") or 0)
                    for row in current_protocols
                ),
                "max_peak_cuda_memory_bytes": max(
                    (
                        float(row.get("peak_cuda_memory_bytes") or 0)
                        for row in current_protocols
                    ),
                    default=float("nan"),
                ),
                "mean_peak_process_rss_bytes": mean(
                    float(row.get("peak_process_rss_bytes") or 0)
                    for row in current_protocols
                ),
                "model_state_bytes": mean(
                    float(row["model_state_bytes"]) for row in current
                ),
                "total_model_communication_bytes": mean(
                    float(row["total_model_communication_bytes"])
                    for row in current
                ),
            }
        )

    bootstrap = config["bootstrap"]
    comparison: dict[str, object] = {
        "paired_seeds": len(paired),
        "selected_alpha": 0.25,
        "selected_certificate": config["selected_certificate"],
    }
    difference_metrics = (
        "selected_minus_feature_macro_f1",
        "selected_minus_feature_worst_client_macro_f1",
        "hybrid_minus_symmetric_certified_ratio",
        "hybrid_minus_symmetric_certified_accuracy",
    )
    for index, metric in enumerate(difference_metrics):
        values = [float(row[metric]) for row in paired]
        low, high = paired_interval(
            values,
            draws=int(bootstrap["draws"]),
            seed=int(bootstrap["seed"]) + index,
        )
        comparison[metric] = mean(values)
        comparison[metric + "_ci_low"] = low
        comparison[metric + "_ci_high"] = high

    summary = {
        "protocol": config["protocol"],
        "complete": complete,
        "decision": (
            "scale_audit_complete_no_selection"
            if complete
            else "scale_audit_failed"
        ),
        "dataset": config["dataset"],
        "node_count": node_count,
        "selected_variant": "dual_alpha025",
        "selected_alpha": 0.25,
        "selected_certificate": config["selected_certificate"],
        "expected_runner_units": expected_units,
        "found_runner_units": units,
        "expected_metric_rows": expected_rows,
        "found_metric_rows": len(rows),
        "certificate_violations": violations,
        "pure_feature_min_certified_ratio": finite_or_none(feature_min),
        "alpha_or_certificate_selected_on_ogb": False,
        "test_labels_used_for_selection": False,
        "errors": errors,
    }
    write_csv(analysis / "paper_table_e18c_ogb_cone_scale.csv", table)
    write_csv(analysis / "paper_table_e18c_ogb_cone_comparison.csv", [comparison])
    write_csv(analysis / "paper_data_e18c_ogb_cone_paired.csv", paired)
    write_csv(analysis / "paper_data_e18c_ogb_cone_all_metrics.csv", rows)
    (analysis / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        raise SystemExit(
            f"E18c incomplete or unsafe: units={units}/{expected_units}; "
            f"rows={len(rows)}/{expected_rows}; violations={violations}; "
            f"errors={errors}"
        )


if __name__ == "__main__":
    main()
