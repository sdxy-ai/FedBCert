#!/usr/bin/env python3
"""Analyze the no-selection E23 multi-safe-expert cover confirmation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fedbcert_e23_safe_cover_protocol.json"
DEFAULT_ROOT = ROOT / "results_boundary_certified/boundary_certified_spectral_e23_safe_cover"
METHODS = ["feature_ce", "feature_kd_cad", "spectral_cad", "safe_cover"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--require-complete", action="store_true")
    return p


def bootstrap_ci(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    samples = generator.choice(array, (int(draws), array.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    args = parser().parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
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
                if record.get("test_evaluation_count") != 1:
                    errors.append(f"test count: {path}")
                if record.get("teacher_test_evaluation_count_before_cover") != 0:
                    errors.append(f"teacher test leakage: {path}")
                if record.get("test_labels_used_for_training_or_selection") is not False:
                    errors.append(f"test selection leakage: {path}")
                if record.get("ordinary_teacher_prediction_preserved_for_all_methods") is not True:
                    errors.append(f"teacher prediction changed: {path}")
                if record.get("cover_is_exact_component_union") is not True:
                    errors.append(f"cover is not exact union: {path}")
                records.append(record)

    expected = int(config["expected_cover_evaluation_units"])
    teacher_units = sum(
        (args.result_root / "teachers" / f"{dataset}_{partition}_c3_seed{seed}" / "run_complete.json").exists()
        for dataset in config["datasets"]
        for partition in config["partitions"]
        for seed in config["confirmation_seeds"]
    )
    component_units = sum(
        (args.result_root / "components" / method / f"{dataset}_{partition}_c3_seed{seed}" / "metrics.json").exists()
        for method in ["feature_ce", "feature_kd_cad", "spectral_cad"]
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
        group = [record for record in records if record["dataset"] == dataset]
        if not group:
            continue
        for method in METHODS:
            rows.append({
                "dataset": dataset,
                "method": method,
                "units": len(group),
                "test_utility_macro_f1": float(np.mean([
                    record["metrics"][method]["test_utility_macro_f1"] for record in group
                ])),
                "test_certified_ratio": float(np.mean([
                    record["metrics"][method]["test_certified_ratio"] for record in group
                ])),
                "test_certified_accuracy": float(np.mean([
                    record["metrics"][method]["test_certified_accuracy"] for record in group
                ])),
            })

    cover_feature = [
        r["metrics"]["safe_cover"]["test_certified_ratio"]
        - r["metrics"]["feature_kd_cad"]["test_certified_ratio"]
        for r in records
    ]
    cover_spectral = [
        r["metrics"]["safe_cover"]["test_certified_ratio"]
        - r["metrics"]["spectral_cad"]["test_certified_ratio"]
        for r in records
    ]
    feature_cad_ce = [
        r["metrics"]["feature_kd_cad"]["test_certified_ratio"]
        - r["metrics"]["feature_ce"]["test_certified_ratio"]
        for r in records
    ]
    violations = sum(
        metrics["test_certified_completion_violations"]
        + metrics["test_logit_bound_violations"]
        for record in records
        for metrics in record["metrics"].values()
    )
    cover_feature_ci = bootstrap_ci(cover_feature, 23125) if cover_feature else [None, None]
    cover_spectral_ci = bootstrap_ci(cover_spectral, 23126) if cover_spectral else [None, None]
    feature_cad_ce_ci = bootstrap_ci(feature_cad_ce, 23127) if feature_cad_ce else [None, None]
    claims = config["claims"]
    supported = (
        complete
        and float(np.mean(cover_feature)) >= claims["mean_cover_minus_feature_cad_certified_ratio_min"]
        and cover_feature_ci[0] >= claims["cover_minus_feature_cad_certified_ratio_ci_low_min"]
        and violations <= claims["maximum_certificate_violations"]
    )
    decision = (
        "confirmation_complete_claim_supported"
        if supported
        else "confirmation_complete_claim_not_supported"
        if complete
        else "confirmation_incomplete"
    )
    analysis_dir = args.result_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (analysis_dir / "paper_table_e23_by_dataset.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    overall_rows: list[dict] = []
    if records:
        for method in METHODS:
            overall_rows.append({
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
        with (analysis_dir / "paper_table_e23_overall.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(overall_rows[0]))
            writer.writeheader()
            writer.writerows(overall_rows)
    summary = {
        "protocol": config["protocol"],
        "complete": complete,
        "decision": decision,
        "model_or_hyperparameter_selection_performed": False,
        "expected_cover_evaluation_units": expected,
        "found_cover_evaluation_units": len(records),
        "expected_teacher_units": int(config["expected_teacher_units"]),
        "found_teacher_units": teacher_units,
        "expected_component_training_units": int(config["expected_component_training_units"]),
        "found_component_training_units": component_units,
        "untouched_e22_confirmation_seeds": config["untouched_e22_confirmation_seeds"],
        "certificate_violations": violations,
        "mean_cover_minus_feature_cad_certified_ratio": float(np.mean(cover_feature)) if cover_feature else None,
        "cover_minus_feature_cad_certified_ratio_95ci": cover_feature_ci,
        "mean_cover_minus_spectral_cad_certified_ratio": float(np.mean(cover_spectral)) if cover_spectral else None,
        "cover_minus_spectral_cad_certified_ratio_95ci": cover_spectral_ci,
        "mean_feature_cad_minus_feature_ce_certified_ratio": float(np.mean(feature_cad_ce)) if feature_cad_ce else None,
        "feature_cad_minus_feature_ce_certified_ratio_95ci": feature_cad_ce_ci,
        "mean_feature_kd_cad_minus_feature_ce_certified_ratio": float(np.mean(feature_cad_ce)) if feature_cad_ce else None,
        "feature_kd_cad_minus_feature_ce_certified_ratio_95ci": feature_cad_ce_ci,
        "test_labels_used_for_selection": False,
        "errors": errors,
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
