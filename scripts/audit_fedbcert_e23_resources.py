#!/usr/bin/env python3
"""Post-hoc, no-selection resource/interface audit for frozen E23 artifacts.

This script never trains a model and never reads labels. It reports serialized
safe-expert sizes, tensor counts, stored teacher-logit sizes, and all-node route
source proportions already recorded by the frozen E23 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch


def iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_tensors(child)


def state_stats(path: Path) -> tuple[int, int]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    tensors = list(iter_tensors(value))
    return sum(int(t.numel()) for t in tensors), sum(int(t.numel() * t.element_size()) for t in tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results_boundary_certified/boundary_certified_spectral_e23_safe_cover"),
    )
    args = parser.parse_args()
    root = args.root
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    component_rows: list[dict[str, Any]] = []
    for variant in ("feature_ce", "feature_kd_cad", "spectral_cad"):
        for state_path in sorted((root / "components" / variant).glob("*/safe_state.pt")):
            unit = state_path.parent.name
            metrics_path = state_path.parent / "metrics.json"
            record = json.loads(metrics_path.read_text(encoding="utf-8"))
            parameter_count, tensor_bytes = state_stats(state_path)
            component_rows.append(
                {
                    "unit": unit,
                    "dataset": record["dataset"],
                    "partition": record["partition"],
                    "seed": record["seed"],
                    "variant": variant,
                    "parameter_count": parameter_count,
                    "tensor_bytes": tensor_bytes,
                    "serialized_bytes": state_path.stat().st_size,
                }
            )

    if len(component_rows) != 72:
        raise RuntimeError(f"expected 72 frozen component states, found {len(component_rows)}")

    route_rows: list[dict[str, Any]] = []
    for metrics_path in sorted((root / "evaluation").glob("*/metrics.json")):
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        counts = record["cover_source_counts_all_nodes"]
        total = sum(int(v) for v in counts.values())
        route_rows.append(
            {
                "unit": metrics_path.parent.name,
                "dataset": record["dataset"],
                "partition": record["partition"],
                "seed": record["seed"],
                "node_count": total,
                "feature_safe_ratio": counts["feature_safe"] / total,
                "spectral_safe_additional_ratio": counts["spectral_safe_additional"] / total,
                "utility_unshielded_ratio": counts["utility_unshielded"] / total,
            }
        )
    if len(route_rows) != 24:
        raise RuntimeError(f"expected 24 frozen cover evaluations, found {len(route_rows)}")

    teacher_logit_paths = sorted((root / "teachers").glob("*/utility_logits.pt"))
    if len(teacher_logit_paths) != 24:
        raise RuntimeError(f"expected 24 teacher-logit artifacts, found {len(teacher_logit_paths)}")

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        by_variant[row["variant"]].append(row)

    variant_summary = {}
    for variant, rows in sorted(by_variant.items()):
        variant_summary[variant] = {
            "units": len(rows),
            "mean_parameter_count": mean(r["parameter_count"] for r in rows),
            "min_parameter_count": min(r["parameter_count"] for r in rows),
            "max_parameter_count": max(r["parameter_count"] for r in rows),
            "mean_serialized_megabytes": mean(r["serialized_bytes"] for r in rows) / 1_000_000,
        }

    keyed = defaultdict(dict)
    for row in component_rows:
        keyed[row["unit"]][row["variant"]] = row
    cover_payloads = []
    for unit, variants in keyed.items():
        cover_payloads.append(
            {
                "unit": unit,
                "parameter_count": variants["feature_kd_cad"]["parameter_count"]
                + variants["spectral_cad"]["parameter_count"],
                "serialized_bytes": variants["feature_kd_cad"]["serialized_bytes"]
                + variants["spectral_cad"]["serialized_bytes"],
            }
        )

    summary = {
        "protocol": "FedBCert E23 frozen post-hoc resource and route audit",
        "complete": True,
        "training_or_selection_performed": False,
        "labels_read": False,
        "component_units": len(component_rows),
        "evaluation_units": len(route_rows),
        "variant_summary": variant_summary,
        "safe_cover_additional_experts": 2,
        "safe_cover_mean_combined_parameter_count": mean(r["parameter_count"] for r in cover_payloads),
        "safe_cover_min_combined_parameter_count": min(r["parameter_count"] for r in cover_payloads),
        "safe_cover_max_combined_parameter_count": max(r["parameter_count"] for r in cover_payloads),
        "safe_cover_mean_combined_serialized_megabytes": mean(r["serialized_bytes"] for r in cover_payloads)
        / 1_000_000,
        "mean_stored_teacher_logits_megabytes": mean(p.stat().st_size for p in teacher_logit_paths) / 1_000_000,
        "mean_all_node_route_source_ratio": {
            "feature_safe": mean(r["feature_safe_ratio"] for r in route_rows),
            "spectral_safe_additional": mean(r["spectral_safe_additional_ratio"] for r in route_rows),
            "utility_unshielded": mean(r["utility_unshielded_ratio"] for r in route_rows),
        },
        "interpretation": (
            "Sizes cover only the two added safe experts and stored teacher logits. "
            "They exclude the underlying FedLap+ teacher model, global spectral preprocessing, and runtime."
        ),
    }

    (analysis / "resource_route_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, rows in (("resource_component_units.csv", component_rows), ("route_source_units.csv", route_rows)):
        with (analysis / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
