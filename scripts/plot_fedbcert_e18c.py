#!/usr/bin/env python3
"""Plot the frozen E18c ogbn-arxiv audit with the E19b no-graph control."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GRAY = "#6F7782"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pair_bars(axis, values, labels, colors, title, ylim) -> None:
    bars = axis.bar(range(len(values)), values, color=colors, width=0.62)
    axis.set_xticks(range(len(values)), labels)
    axis.set_ylim(*ylim)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (ylim[1] - ylim[0]) * 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default=(
            "results_boundary_certified/"
            "boundary_certified_spectral_e18c_ogb_cone_scale/analysis"
        ),
    )
    parser.add_argument(
        "--identity-analysis-root",
        default=(
            "results_boundary_certified/"
            "boundary_certified_spectral_e19b_ogb_no_graph/analysis"
        ),
    )
    args = parser.parse_args()
    root = Path(args.analysis_root)
    rows = {row["variant"]: row for row in read_csv(root / "paper_table_e18c_ogb_cone_scale.csv")}
    feature = rows["feature_alpha0"]
    selected = rows["dual_alpha025"]
    identity_rows = read_csv(
        Path(args.identity_analysis_root) / "paper_table_e19b_ogb_overall.csv"
    )
    if len(identity_rows) != 1:
        raise ValueError("E19b identity table must contain exactly one row")
    identity = identity_rows[0]

    figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.1))
    pair_bars(
        axes[0, 0],
        [
            float(feature["test_pooled_macro_f1"]),
            float(identity["identity_macro_f1"]),
            float(selected["test_pooled_macro_f1"]),
        ],
        ["Feature\n$\\alpha=0$", "$P=I$\n$\\alpha=.25$", "Observed $P$\n$\\alpha=.25$"],
        [GRAY, ORANGE, BLUE],
        "(a) Pooled Macro-F1",
        (0.17, 0.21),
    )
    pair_bars(
        axes[0, 1],
        [
            float(feature["symmetric_certified_accuracy"]),
            float(identity["identity_accuracy"]),
            float(identity["observed_accuracy"]),
        ],
        ["Feature", "$P=I$", "Observed $P$"],
        [GRAY, ORANGE, BLUE],
        "(b) Accuracy",
        (0.40, 0.47),
    )
    pair_bars(
        axes[1, 0],
        [
            float(identity["identity_certified_ratio"]),
            float(selected["symmetric_certified_ratio"]),
            float(selected["cone_hybrid_certified_ratio"]),
        ],
        ["$P=I$", "Observed\nsymmetric", "Observed\ncombined"],
        [ORANGE, GRAY, GREEN],
        "(c) Certified coverage",
        (0.0, 1.12),
    )
    pair_bars(
        axes[1, 1],
        [
            float(identity["identity_certified_accuracy"]),
            float(selected["symmetric_certified_accuracy"]),
            float(selected["cone_hybrid_certified_accuracy"]),
        ],
        ["$P=I$", "Observed\nsymmetric", "Observed\ncombined"],
        [ORANGE, GRAY, GREEN],
        "(d) Certified accuracy",
        (0.0, 0.50),
    )
    figure.suptitle(
        "ogbn-arxiv: frozen $\\alpha=0.25$ with parameter-matched $P=I$ control",
        fontsize=11,
    )
    figure.tight_layout()
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"figure_e18c_ogb_cone_scale.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
