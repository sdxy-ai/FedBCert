#!/usr/bin/env python3
"""Plot the fixed-predictor E27 information-granularity audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "exact": "Exact",
    "bin2": "Bin-2",
    "bin4": "Bin-4",
    "bin8": "Bin-8",
    "presence": "Presence",
    "client_max": "Client max",
    "universal": "Universal",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default="results_boundary_certified/boundary_certified_spectral_e27_ledger_granularity/analysis",
    )
    args = parser.parse_args()
    root = Path(args.analysis_root)
    table = root / "paper_table_e27_ledger_granularity_overall.csv"
    with table.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = list(LABELS)
    lookup = {row["certificate_information"]: row for row in rows}
    rows = [lookup[name] for name in order]
    labels = [LABELS[name] for name in order]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    x = range(len(rows))
    axes[0].plot(x, [float(row["mean_certified_ratio"]) for row in rows], "o-", label="All test nodes")
    axes[0].plot(x, [float(row["mean_boundary_certified_ratio"]) for row in rows], "s--", label="Boundary nodes")
    axes[0].plot(x, [float(row["mean_worst_client_certified_ratio"]) for row in rows], "^:", label="Worst client")
    axes[0].set_ylabel("Certified coverage")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Certified coverage")

    axes[1].bar(x, [float(row["mean_representation_radius"]) for row in rows], color="#4C78A8")
    axes[1].set_ylabel("Mean representation radius")
    axes[1].set_title("Certificate conservatism")
    for axis in axes:
        axis.set_xticks(list(x), labels, rotation=32, ha="right")
        axis.grid(axis="y", alpha=0.22)
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_root / f"figure_e27_ledger_granularity.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
