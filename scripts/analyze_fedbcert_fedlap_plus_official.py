#!/usr/bin/env python3
"""Audit and summarize the 24-run FedLap+ official-code confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = ["Cora", "CiteSeer", "PubMed", "Photo"]
PARTITIONS = ["metis", "random"]
SEEDS = [21, 22, 23]
ADAPTER_REVISION = "fedlap-plus-e8-v1-paper-hparams"
FEDLAP_COMMIT = "652fba0bb2e9a660a31ce7e8abcf728c225d188c"
BOOTSTRAP_SEED = 20260821


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-root",
        default="results_boundary_certified/boundary_certified_spectral_e8_q1/official_bridges/fedlap_plus_core",
    )
    p.add_argument(
        "--matched-core-root",
        default="results_boundary_certified/boundary_certified_spectral_e8_q1/matched_core",
    )
    p.add_argument(
        "--bundle-root",
        default="results_boundary_certified/boundary_certified_spectral_e8_q1/official_bridges/bundles",
    )
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--require-complete", action="store_true")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ba_metrics(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matched = [row for row in rows if row["method"] == "fedbcert_ba"]
    if len(matched) != 1:
        raise RuntimeError(f"expected one fedbcert_ba row in {path}")
    return matched[0]


def bootstrap_interval(values: list[float], samples: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    means = array[generator.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(subset: list[dict[str, object]], samples: int) -> dict[str, object]:
    columns = {
        "fedlap_plus_pooled": "fedlap_plus_pooled_test_macro_f1",
        "fedlap_plus_worst": "fedlap_plus_worst_client_test_macro_f1",
        "fedbcert_ba_pooled": "fedbcert_ba_pooled_test_macro_f1",
        "fedbcert_ba_worst": "fedbcert_ba_worst_client_test_macro_f1",
        "ba_minus_fedlap_plus_pooled": "ba_minus_fedlap_plus_pooled_f1",
        "ba_minus_fedlap_plus_worst": "ba_minus_fedlap_plus_worst_f1",
    }
    output: dict[str, object] = {"runs": len(subset)}
    for label, column in columns.items():
        values = [float(row[column]) for row in subset]
        array = np.asarray(values, dtype=float)
        output[f"{label}_mean"] = float(array.mean())
        output[f"{label}_sample_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        if label.startswith("ba_minus_"):
            low, high = bootstrap_interval(values, samples)
            output[f"{label}_bootstrap_95_low"] = low
            output[f"{label}_bootstrap_95_high"] = high
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parser().parse_args()
    root = Path(args.result_root)
    matched_root = Path(args.matched_core_root)
    bundle_root = Path(args.bundle_root)
    expected = [
        f"{dataset}_{partition}_c3_seed{seed}"
        for dataset in DATASETS for partition in PARTITIONS for seed in SEEDS
    ]
    missing: list[str] = []
    rows: list[dict[str, object]] = []

    for run_name in expected:
        result_path = root / run_name / "fedlap_plus_official_metrics.json"
        if not result_path.exists():
            missing.append(run_name)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("adapter_revision") != ADAPTER_REVISION:
            raise RuntimeError(f"legacy/unrecognized adapter revision in {result_path}")
        if result.get("method") != "FedLap+-official-Arnoldi":
            raise RuntimeError(f"unexpected method in {result_path}")
        if result.get("source_commit") != FEDLAP_COMMIT:
            raise RuntimeError(f"unfrozen FedLap commit in {result_path}")
        if not result.get("exact_fedbcert_bundle_consumed"):
            raise RuntimeError(f"non-matched split in {result_path}")
        if not result.get("paper_hyperparameters_applied"):
            raise RuntimeError(f"paper hyperparameters not applied in {result_path}")
        if result.get("smoke_round_override"):
            raise RuntimeError(f"smoke run found in formal result root: {result_path}")
        if result.get("test_labels_used_for_training_or_selection"):
            raise RuntimeError(f"test-label leakage flag in {result_path}")
        if int(result.get("test_evaluation_count", -1)) != 1:
            raise RuntimeError(f"unexpected test evaluation count in {result_path}")
        variant = result.get("variant", {})
        expected_variant = {
            "structure_model": "LanczosLaplace",
            "structure_type": "hop2vec",
            "spectral_matrix": "lap",
            "spectral_method": "arnoldi",
        }
        for key, value in expected_variant.items():
            if variant.get(key) != value:
                raise RuntimeError(f"unexpected {key} in {result_path}")

        manifest_path = bundle_root / run_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_path = manifest_path.parent / manifest["bundle"]
        if sha256(bundle_path) != manifest["bundle_sha256"]:
            raise RuntimeError(f"current bundle hash mismatch: {bundle_path}")
        if result["bundle_sha256"] != manifest["bundle_sha256"]:
            raise RuntimeError(f"result/bundle hash mismatch in {result_path}")

        ba = read_ba_metrics(matched_root / run_name / "federated_metrics.csv")
        fedlap_f1 = float(result["test"]["pooled_macro_f1"])
        fedlap_worst = float(result["test"]["worst_client_macro_f1"])
        ba_f1 = float(ba["test_local_macro_f1"])
        ba_worst = float(ba["test_worst_client_local_macro_f1"])
        rows.append({
            "dataset": result["dataset"],
            "partition": result["partition"],
            "num_clients": result["num_clients"],
            "seed": result["seed"],
            "fedlap_plus_pooled_test_macro_f1": fedlap_f1,
            "fedlap_plus_worst_client_test_macro_f1": fedlap_worst,
            "fedbcert_ba_pooled_test_macro_f1": ba_f1,
            "fedbcert_ba_worst_client_test_macro_f1": ba_worst,
            "ba_minus_fedlap_plus_pooled_f1": ba_f1 - fedlap_f1,
            "ba_minus_fedlap_plus_worst_f1": ba_worst - fedlap_worst,
            "fedlap_plus_selected_round": result["selected_round"],
            "fedlap_plus_initialization_seconds": result["initialization_seconds"],
            "fedlap_plus_wall_seconds": result["wall_seconds"],
            "fedlap_plus_peak_cuda_memory_mb": result["peak_cuda_memory_mb"],
        })

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "missing_runs.txt").write_text(
        "\n".join(missing) + ("\n" if missing else ""), encoding="utf-8"
    )
    if missing and args.require_complete:
        raise SystemExit(f"missing {len(missing)}/24 runs: {', '.join(missing)}")
    if not rows:
        print(f"[INCOMPLETE] 0/24; missing list: {analysis_dir / 'missing_runs.txt'}")
        return

    write_csv(analysis_dir / "paper_data_fedlap_plus_official_paired.csv", rows)
    by_dataset = [
        {"dataset": dataset, **summarize([row for row in rows if row["dataset"] == dataset], args.bootstrap_samples)}
        for dataset in DATASETS if any(row["dataset"] == dataset for row in rows)
    ]
    write_csv(analysis_dir / "paper_table_fedlap_plus_official_by_dataset.csv", by_dataset)
    overall = summarize(rows, args.bootstrap_samples)
    write_csv(
        analysis_dir / "paper_table_fedlap_plus_official_overall.csv",
        [{"scope": "four_paper_supported_matched_c3_datasets", **overall}],
    )
    summary = {
        "expected_runs": 24,
        "completed_runs": len(rows),
        "missing_runs": missing,
        "official_commit_verified": True,
        "paper_hyperparameters_verified": True,
        "exact_bundle_verified": True,
        "current_bundle_hashes_verified": True,
        "test_selection_leakage_flags": 0,
        "test_evaluation_count_verified": True,
        "computers_excluded_as_not_paper_supported": True,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        **overall,
        "mean_fedlap_plus_pooled_test_macro_f1": overall["fedlap_plus_pooled_mean"],
        "mean_fedlap_plus_worst_client_test_macro_f1": overall["fedlap_plus_worst_mean"],
        "mean_ba_minus_fedlap_plus_pooled_f1": overall["ba_minus_fedlap_plus_pooled_mean"],
        "mean_ba_minus_fedlap_plus_worst_f1": overall["ba_minus_fedlap_plus_worst_mean"],
        "mean_fedlap_plus_initialization_seconds": float(np.mean([row["fedlap_plus_initialization_seconds"] for row in rows])),
        "mean_fedlap_plus_wall_seconds": float(np.mean([row["fedlap_plus_wall_seconds"] for row in rows])),
        "mean_fedlap_plus_peak_cuda_memory_mb": float(np.mean([row["fedlap_plus_peak_cuda_memory_mb"] for row in rows])),
        "max_fedlap_plus_peak_cuda_memory_mb": float(np.max([row["fedlap_plus_peak_cuda_memory_mb"] for row in rows])),
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
