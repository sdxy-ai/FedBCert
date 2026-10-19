#!/usr/bin/env python3
"""Create the E15 fixed-frontier and certificate-tightening paper figures."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="results_boundary_certified/boundary_certified_spectral_e15",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    analysis = Path(args.root) / "analysis"
    frontier = read_rows(analysis / "paper_table_e15_frontier.csv")
    comparisons = read_rows(
        analysis / "paper_table_e15_hierarchical_comparisons.csv"
    )
    curve = sorted(
        (
            row
            for row in frontier
            if row["variant"] == "feature_alpha0"
            or row["variant"].startswith("dual_alpha")
        ),
        key=lambda row: float(row["alpha"]),
    )
    f1_comparisons = {
        row["variant"]: row
        for row in comparisons
        if row["metric"] == "test_local_macro_f1"
    }

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    axes[0].plot(
        [float(row["test_certified_ratio"]) for row in curve],
        [float(row["test_local_macro_f1"]) for row in curve],
        marker="o",
        linewidth=1.8,
    )
    for row in curve:
        alpha = float(row["alpha"])
        if alpha == 1.0:
            # Move the upper-left endpoint label above and into the panel.
            offset = (8, 9)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        elif alpha == 0.0:
            # Move the lower-right endpoint label below and into the panel.
            offset = (-10, -10)
            horizontal_alignment = "right"
            vertical_alignment = "top"
        else:
            offset = (5, 5)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        axes[0].annotate(
            rf"$\alpha$={alpha:g}",
            (
                float(row["test_certified_ratio"]),
                float(row["test_local_macro_f1"]),
            ),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va=vertical_alignment,
        )
    axes[0].set_xlabel("Test-set certified coverage")
    axes[0].set_ylabel("Pooled test Macro-F1")
    axes[0].set_title("Utility-certificate frontier")
    axes[0].margins(x=0.08, y=0.16)
    axes[0].grid(alpha=0.25)

    nonzero = [row for row in curve if float(row["alpha"]) > 0]
    x_values = [float(row["alpha"]) for row in nonzero]
    means = [
        float(f1_comparisons[row["variant"]]["mean_difference_vs_feature"])
        for row in nonzero
    ]
    low = [
        means[index]
        - float(f1_comparisons[row["variant"]]["hierarchical_ci_low"])
        for index, row in enumerate(nonzero)
    ]
    high = [
        float(f1_comparisons[row["variant"]]["hierarchical_ci_high"])
        - means[index]
        for index, row in enumerate(nonzero)
    ]
    axes[1].errorbar(
        x_values, means, yerr=[low, high], marker="o", linewidth=1.8, capsize=4
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"Spectral fusion coefficient $\alpha$")
    axes[1].set_ylabel("Macro-F1 difference from feature endpoint")
    axes[1].set_title("Paired utility difference (95% CI)")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels([f"{value:g}" for value in x_values])
    axes[1].set_xlim(0.05, 1.05)
    axes[1].grid(alpha=0.25)

    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"figure_e15_fixed_pareto.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    tightening_path = analysis / "paper_table_e15_certificate_tightening.csv"
    if not tightening_path.exists():
        return
    tightening = [
        row
        for row in read_rows(tightening_path)
        if row["metric"] == "certified_ratio"
    ]
    tightening.sort(key=lambda row: float(row["alpha"]))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    labels = {
        "tight_pairwise": "Retained-edge mass + class difference",
        "legacy_pairwise": "Legacy + pairwise",
        "tight_classwise": "Retained-edge mass + classwise",
        "legacy_classwise": "Legacy + classwise",
    }
    alpha_values = [float(row["alpha"]) for row in tightening]
    for method, label in labels.items():
        axes[0].plot(
            alpha_values,
            [float(row[method]) for row in tightening],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    axes[0].set_xlabel(r"Fixed spectral fusion coefficient $\alpha$")
    axes[0].set_ylabel("Test certified coverage")
    axes[0].set_title("Fixed-predictor certificate tightening")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    nonzero = [row for row in tightening if float(row["alpha"]) > 0]
    gain_specs = (
        ("retained_mass_gain", "Retained-edge-mass gain"),
        ("pairwise_margin_gain", "Class-difference gain"),
        ("combined_gain", "Combined gain"),
    )
    for name, label in gain_specs:
        gain_means = [float(row[name]) for row in nonzero]
        lows = [
            gain - float(row[name + "_ci_low"])
            for gain, row in zip(gain_means, nonzero)
        ]
        highs = [
            float(row[name + "_ci_high"]) - gain
            for gain, row in zip(gain_means, nonzero)
        ]
        axes[1].errorbar(
            [float(row["alpha"]) for row in nonzero],
            gain_means,
            yerr=[lows, highs],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=label,
        )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"Fixed spectral fusion coefficient $\alpha$")
    axes[1].set_ylabel("Certified-coverage lift")
    axes[1].set_title("Hierarchical 95% bootstrap intervals")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"figure_e15_certificate_tightening.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
