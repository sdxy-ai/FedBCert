#!/usr/bin/env python3
"""Create publication figures for the frozen E18b cone-certificate confirmation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6F7782"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_overall(
    dataset_rows: list[dict[str, str]], overall: dict[str, str]
) -> list[dict[str, str]]:
    row = dict(overall)
    row["dataset"] = "Overall"
    return dataset_rows + [row]


def grouped_panel(
    axis,
    rows: list[dict[str, str]],
    *,
    first_key: str,
    second_key: str,
    first_label: str,
    second_label: str,
    title: str,
    ylim: tuple[float, float],
) -> None:
    labels = [row["dataset"] for row in rows]
    first = np.asarray([float(row[first_key]) for row in rows])
    second = np.asarray([float(row[second_key]) for row in rows])
    x = np.arange(len(rows))
    width = 0.37
    axis.bar(x - width / 2, first, width, color=GRAY, label=first_label)
    axis.bar(x + width / 2, second, width, color=GREEN, label=second_label)
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylim(*ylim)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    for xpos, value in zip(x, second - first):
        axis.text(
            xpos,
            max(first[int(xpos)], second[int(xpos)]) + 0.012,
            f"{100 * value:+.1f} pp",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=GREEN,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default=(
            "results_boundary_certified/"
            "boundary_certified_spectral_e18_cone/analysis"
        ),
    )
    args = parser.parse_args()
    root = Path(args.analysis_root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    by_dataset = read_csv(root / "paper_table_e18b_by_dataset.csv")
    overall = read_csv(root / "paper_table_e18b_overall.csv")[0]
    rows = add_overall(by_dataset, overall)

    # A vertical stack remains legible in the single-column preprint and does
    # not compress seven dataset labels into three narrow horizontal panels.
    figure, axes = plt.subplots(3, 1, figsize=(7.4, 7.8))
    labels = [row["dataset"] for row in rows]
    x = np.arange(len(rows))
    width = 0.37
    feature = np.asarray([float(row["feature_macro_f1"]) for row in rows])
    selected = np.asarray([float(row["selected_macro_f1"]) for row in rows])
    axes[0].bar(x - width / 2, feature, width, color=GRAY, label=r"Feature ($\alpha=0$)")
    axes[0].bar(x + width / 2, selected, width, color=BLUE, label=r"Frozen ($\alpha=0.25$)")
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].set_ylim(0.25, 0.91)
    axes[0].set_title("(a) Pooled Macro-F1")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].set_axisbelow(True)
    for xpos, value in zip(x, selected - feature):
        axes[0].text(
            xpos,
            max(feature[int(xpos)], selected[int(xpos)]) + 0.012,
            f"{100 * value:+.1f} pp",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=BLUE,
        )
    grouped_panel(
        axes[1],
        rows,
        first_key="symmetric_certified_ratio",
        second_key="cone_hybrid_certified_ratio",
        first_label="Symmetric",
        second_label="Combined",
        title="(b) Certified coverage",
        ylim=(0.55, 1.03),
    )
    grouped_panel(
        axes[2],
        rows,
        first_key="symmetric_certified_accuracy",
        second_key="cone_hybrid_certified_accuracy",
        first_label="Symmetric",
        second_label="Combined",
        title="(c) Certified accuracy",
        ylim=(0.35, 0.91),
    )
    handles, legend_labels = [], []
    for axis in axes:
        local_handles, local_labels = axis.get_legend_handles_labels()
        for handle, label in zip(local_handles, local_labels):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"figure_e18_cone_confirmation.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    paired = read_csv(root / "paper_data_e18b_paired.csv")
    datasets = list(dict.fromkeys(row["dataset"] for row in paired))
    palette = dict(
        zip(datasets, [BLUE, ORANGE, GREEN, "#CC79A7", "#E69F00", "#56B4E9"])
    )
    figure, axis = plt.subplots(figsize=(5.8, 4.45))
    for dataset in datasets:
        subset = [row for row in paired if row["dataset"] == dataset]
        axis.scatter(
            [100 * float(row["macro_f1_difference"]) for row in subset],
            [
                100 * float(row["hybrid_minus_symmetric_certified_accuracy"])
                for row in subset
            ],
            s=27,
            alpha=0.72,
            color=palette[dataset],
            label=dataset,
            edgecolors="none",
        )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    axis.axvline(0, color="black", linewidth=0.8, alpha=0.55)
    axis.scatter(
        [100 * float(overall["macro_f1_difference"])],
        [100 * float(overall["hybrid_minus_symmetric_certified_accuracy"])],
        marker="*",
        s=185,
        color="black",
        label="Overall mean",
        zorder=5,
    )
    axis.set_xlabel(r"Macro-F1 gain over feature endpoint (pp)")
    axis.set_ylabel(r"Combined-certificate gain (pp)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"figure_e18_paired_tradeoff.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
