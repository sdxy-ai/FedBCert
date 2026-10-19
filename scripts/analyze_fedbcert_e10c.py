#!/usr/bin/env python3
"""Audit E10c immutable bundles and strictly pair the frozen upgrade to baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = ("Cora", "CiteSeer", "PubMed", "Computers", "Photo")
PARTITIONS = ("metis", "random")
SEEDS = (21, 22, 23)
NATIVE = ("fedsage_plus", "feddep", "fedpub", "fedtad", "fedgta")
EXPECTED_VARIANT = {
    "Cora": "e8_linear_h64_d1",
    "CiteSeer": "e8_linear_h64_d1",
    "PubMed": "e8_linear_h64_d1",
    "Computers": "dual_h64_d2_s05_sep025",
    "Photo": "dual_h64_d2_s05_sep025",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--root",
        default="results_boundary_certified/boundary_certified_spectral_e10c",
    )
    value.add_argument(
        "--e8-root",
        default="results_boundary_certified/boundary_certified_spectral_e8_q1",
    )
    value.add_argument("--require-complete", action="store_true")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def interval(values: list[float], seed: int = 20260823) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(bootstrap, [2.5, 97.5]))


def identity(record: dict) -> tuple[str, str, int]:
    return record["dataset"], record["partition"], int(record["seed"])


def utility(record: dict) -> tuple[float, float]:
    return float(record["pooled_f1"]), float(record["worst_f1"])


def main() -> None:
    args = parser().parse_args()
    root = Path(args.root)
    e8 = Path(args.e8_root)
    analysis = root / "analysis"
    errors: list[str] = []
    expected_units = {
        (dataset, partition, seed)
        for dataset in DATASETS
        for partition in PARTITIONS
        for seed in SEEDS
    }

    upgraded: dict[tuple[str, str, int], dict] = {}
    same_arch_avg: dict[tuple[str, str, int], dict] = {}
    for metrics_path in root.glob("e8_bundle_paired/*/*/federated_metrics.csv"):
        rows = read_csv(metrics_path)
        methods = {row["method"]: row for row in rows}
        if set(methods) != {"fedavg", "fedbcert_ba"}:
            errors.append(f"method_set:{metrics_path}")
            continue
        key = identity(rows[0])
        variant = metrics_path.parents[1].name
        if variant != EXPECTED_VARIANT.get(key[0]):
            errors.append(f"variant:{key}:{variant}")
        protocol_path = metrics_path.with_name("protocol.json")
        completion_path = metrics_path.with_name("run_complete.json")
        if not protocol_path.exists() or not completion_path.exists():
            errors.append(f"completion:{key}")
            continue
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        audit = protocol.get("immutable_split_bundle_audit", {})
        checks = {
            "test_selection": protocol.get("test_labels_used_for_selection") is not False,
            "test_emission": protocol.get("test_metrics_emitted") is not True,
            "bundle_enabled": audit.get("enabled") is not True,
            "bundle_hash": audit.get("hash_verified") is not True,
            "bundle_identity": audit.get("identity_verified") is not True,
            "bundle_keys": audit.get("exact_keys_verified") is not True,
        }
        bundle_dir = e8 / "official_bridges" / "bundles" / (
            f"{key[0]}_{key[1]}_c3_seed{key[2]}"
        )
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        current_hash = sha256(bundle_dir / "split_bundle.pt")
        if current_hash != manifest["bundle_sha256"] or audit.get("bundle_sha256") != current_hash:
            checks["current_bundle_hash"] = True
        for label, failed in checks.items():
            if failed:
                errors.append(f"{label}:{key}")
        for target, method in ((same_arch_avg, "fedavg"), (upgraded, "fedbcert_ba")):
            row = methods[method]
            target[key] = {
                "dataset": key[0], "partition": key[1], "seed": key[2],
                "pooled_f1": float(row["test_local_macro_f1"]),
                "worst_f1": float(row["test_worst_client_local_macro_f1"]),
                "certified_accuracy": float(row["test_certified_accuracy"]),
                "certificate_violations": int(row["test_certified_completion_violations"])
                + int(row["test_logit_bound_violations"]),
                "bundle_sha256": current_hash,
            }

    old_ba: dict[tuple[str, str, int], dict] = {}
    for path in e8.glob("matched_core/*/federated_metrics.csv"):
        for row in read_csv(path):
            if row["method"] != "fedbcert_ba" or int(row["num_clients"]) != 3:
                continue
            key = identity(row)
            old_ba[key] = {
                "dataset": key[0], "partition": key[1], "seed": key[2],
                "pooled_f1": float(row["test_local_macro_f1"]),
                "worst_f1": float(row["test_worst_client_local_macro_f1"]),
                "certified_accuracy": float(row["test_certified_accuracy"]),
            }

    baselines: dict[str, dict[tuple[str, str, int], dict]] = {
        "FedAvg-same-arch": same_arch_avg,
        "FedBCert-BA-E8": old_ba,
    }
    for name in NATIVE:
        baselines[name] = {}
    for path in e8.glob("matched_native/*/run_complete.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if int(row["num_clients"]) != 3:
            continue
        if row.get("test_labels_used_for_selection") is not False:
            errors.append(f"native_test_selection:{path}")
        key = identity(row)
        baselines[row["baseline"]][key] = {
            "dataset": key[0], "partition": key[1], "seed": key[2],
            "pooled_f1": float(row["pooled_test_macro_f1"]),
            "worst_f1": float(row["worst_client_test_macro_f1"]),
        }

    for method, folder, filename in (
        ("FedStruct", "fedstruct_core", "fedstruct_official_metrics.json"),
        ("FedLap+", "fedlap_plus_core", "fedlap_plus_official_metrics.json"),
    ):
        baselines[method] = {}
        for path in (e8 / "official_bridges" / folder).glob(f"*/{filename}"):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("test_labels_used_for_training_or_selection") is not False:
                errors.append(f"official_test_selection:{path}")
            if row.get("exact_fedbcert_bundle_consumed") is not True:
                errors.append(f"official_bundle:{path}")
            key = identity(row)
            bundle_dir = e8 / "official_bridges" / "bundles" / (
                f"{key[0]}_{key[1]}_c3_seed{key[2]}"
            )
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            if row.get("bundle_sha256") != manifest["bundle_sha256"]:
                errors.append(f"official_hash:{path}")
            baselines[method][key] = {
                "dataset": key[0], "partition": key[1], "seed": key[2],
                "pooled_f1": float(row["test"]["pooled_macro_f1"]),
                "worst_f1": float(row["test"]["worst_client_macro_f1"]),
            }

    missing = sorted(expected_units - set(upgraded))
    if missing:
        errors.extend(f"missing_e10c:{unit}" for unit in missing)
    if set(same_arch_avg) != set(upgraded):
        errors.append("same_arch_fedavg_unit_mismatch")
    if set(old_ba) != expected_units:
        errors.append("old_ba_unit_mismatch")
    if any(record["certificate_violations"] for record in upgraded.values()):
        errors.append("e10c_certificate_violation")

    paired_rows: list[dict] = []
    for baseline_name, baseline_map in baselines.items():
        common = sorted(set(upgraded) & set(baseline_map))
        for dataset in (*DATASETS, "ALL"):
            subset = [key for key in common if dataset == "ALL" or key[0] == dataset]
            if not subset:
                continue
            pooled_delta = [utility(upgraded[key])[0] - utility(baseline_map[key])[0] for key in subset]
            worst_delta = [utility(upgraded[key])[1] - utility(baseline_map[key])[1] for key in subset]
            certified_delta = [
                float(upgraded[key]["certified_accuracy"])
                - float(baseline_map[key]["certified_accuracy"])
                for key in subset
                if "certified_accuracy" in baseline_map[key]
            ]
            pooled_low, pooled_high = interval(pooled_delta)
            worst_low, worst_high = interval(worst_delta)
            certified_low, certified_high = (
                interval(certified_delta) if certified_delta else ("N/A", "N/A")
            )
            paired_rows.append({
                "dataset": dataset,
                "baseline": baseline_name,
                "paired_units": len(subset),
                "upgrade_minus_baseline_pooled_f1": float(np.mean(pooled_delta)),
                "pooled_f1_ci95_low": pooled_low,
                "pooled_f1_ci95_high": pooled_high,
                "upgrade_minus_baseline_worst_f1": float(np.mean(worst_delta)),
                "worst_f1_ci95_low": worst_low,
                "worst_f1_ci95_high": worst_high,
                "upgrade_minus_baseline_certified_accuracy": (
                    float(np.mean(certified_delta)) if certified_delta else "N/A"
                ),
                "certified_accuracy_ci95_low": certified_low,
                "certified_accuracy_ci95_high": certified_high,
            })

    method_maps = {"FedBCert-BA-DE": upgraded, **baselines}
    utility_rows: list[dict] = []
    for dataset in (*DATASETS, "ALL"):
        for method, records in method_maps.items():
            subset = [
                value for key, value in records.items()
                if dataset == "ALL" or key[0] == dataset
            ]
            if not subset:
                continue
            pooled = [utility(row)[0] for row in subset]
            worst = [utility(row)[1] for row in subset]
            certified = [
                float(row["certified_accuracy"])
                for row in subset if "certified_accuracy" in row
            ]
            utility_rows.append({
                "dataset": dataset,
                "method": method,
                "runs": len(subset),
                "pooled_macro_f1_mean": float(np.mean(pooled)),
                "pooled_macro_f1_std": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
                "worst_client_macro_f1_mean": float(np.mean(worst)),
                "worst_client_macro_f1_std": float(np.std(worst, ddof=1)) if len(worst) > 1 else 0.0,
                "certified_accuracy_mean": float(np.mean(certified)) if certified else "N/A",
            })

    complete = not errors and set(upgraded) == expected_units
    overall_pairing = {
        row["baseline"]: {
            key: value for key, value in row.items()
            if key not in {"dataset", "baseline"}
        }
        for row in paired_rows if row["dataset"] == "ALL"
    }
    summary = {
        "protocol": "FedBCert E10c frozen upgraded model on E8 immutable bundles",
        "complete": complete,
        "completed_units": len(upgraded),
        "expected_units": len(expected_units),
        "all_current_bundle_hashes_verified": not any("hash" in error for error in errors),
        "certificate_violations": sum(
            row["certificate_violations"] for row in upgraded.values()
        ),
        "baseline_unit_counts": {
            method: len(records) for method, records in baselines.items()
        },
        "overall_strict_pairing": overall_pairing,
        "errors": errors,
    }
    if args.require_complete and not complete:
        raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / "paper_table_e10c_utility.csv", utility_rows)
    write_csv(analysis / "paper_table_e10c_strict_paired.csv", paired_rows)
    (analysis / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
