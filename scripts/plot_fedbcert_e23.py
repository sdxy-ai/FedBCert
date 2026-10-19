#!/usr/bin/env python3
"""Plot the frozen E23 multi-safe-expert certificate cover."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = (
    ROOT
    / "results_boundary_certified"
    / "boundary_certified_spectral_e23_safe_cover"
    / "analysis"
)
METHODS = ["feature_ce", "feature_kd_cad", "spectral_cad", "safe_cover"]
LABELS = ["Feature CE", "Feature KD+CAD", "Spectral CAD", "Safe cover"]
COLORS = ["#B8C4D8", "#5B8FF9", "#61DDAA", "#E8684A"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    by_dataset = read_rows(args.analysis_dir / "paper_table_e23_by_dataset.csv")
    overall = read_rows(args.analysis_dir / "paper_table_e23_overall.csv")
    datasets = list(dict.fromkeys(row["dataset"] for row in by_dataset)) + ["Overall"]
    values: dict[str, list[float]] = {method: [] for method in METHODS}
    for dataset in datasets[:-1]:
        subset = {row["method"]: row for row in by_dataset if row["dataset"] == dataset}
        for method in METHODS:
            values[method].append(float(subset[method]["test_certified_ratio"]))
    overall_map = {row["method"]: row for row in overall}
    for method in METHODS:
        values[method].append(float(overall_map[method]["test_certified_ratio"]))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
    })
    fig, ax = plt.subplots(figsize=(7.15, 3.45), constrained_layout=True)
    x = np.arange(len(datasets), dtype=float)
    width = 0.19
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width
    for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        bars = ax.bar(
            x + offsets[index], values[method], width=width,
            label=label, color=color, edgecolor="white", linewidth=0.5,
        )
        if method == "safe_cover":
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in values[method]],
                         padding=2, fontsize=7.5, color="#8C2D1D")

    ax.set_ylabel("Certified ratio")
    ax.set_xticks(x, datasets)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    figure_dir = args.output_dir if args.output_dir is not None else args.analysis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "figure_e23_safe_cover"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
