#!/usr/bin/env python3
"""Run the frozen E25 exact ReLU-cone finite-completion oracle audit.

The predictor is fixed in every trial.  The runner enumerates every physical
topology in the frozen toy completion family and analytically minimizes each
predicted-vs-competitor gap over independent remote hidden rows in
``{h >= 0 : ||h||_2 <= 1}``.  This provides an exact reference for auditing the
one-sided cone certificate rather than another sampled-completion check.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openfgl.flcore.fedbcert.certificate import (
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    certificate_statistics,
    cone_pairwise_certificate_statistics,
    positive_orthant_unit_support,
    retained_mass_uncertainty_basis,
    row_normalized_shift,
)
from scripts.run_fedbcert_e17_exact_completion_oracle import (
    completion_edge_indices,
    local_edge_index,
    polynomial_matrix,
    safe_divide,
)


CONFIG = ROOT / "configs" / "fedbcert_e25_exact_cone_oracle_protocol.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results_boundary_certified"
    / "boundary_certified_spectral_e25_exact_cone_oracle"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_scalar_interval(
    coefficients: torch.Tensor, direction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact interval for sum_j a_j <direction, h_j> on the positive unit ball."""
    support_lower, support_upper = positive_orthant_unit_support(direction)
    lower = torch.where(
        coefficients >= 0.0,
        coefficients * support_lower,
        coefficients * support_upper,
    ).sum()
    upper = torch.where(
        coefficients >= 0.0,
        coefficients * support_upper,
        coefficients * support_lower,
    ).sum()
    return lower, upper


def run_trial(
    *,
    seed: int,
    order_count: int,
    boundary_counts: list[int],
    remote_count: int,
    tolerance: float,
) -> list[dict]:
    torch.manual_seed(seed * 1009 + order_count)
    dtype = torch.float64
    local_count = len(boundary_counts)
    local_shift, local_degree = row_normalized_shift(
        local_edge_index(),
        local_count,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    missing = torch.tensor(boundary_counts, dtype=dtype)
    budget = boundary_uncertainty(local_degree, missing)
    basis = retained_mass_uncertainty_basis(local_shift, budget, order_count)

    model = PolynomialSpectralClassifier(
        input_dim=5,
        hidden_dim=6,
        num_classes=3,
        order_count=order_count,
        dropout=0.0,
        certificate_margin_mode="pairwise_difference",
        encoder_type="linear_relu",
    ).to(dtype=dtype).eval()
    with torch.no_grad():
        model.theta.copy_(torch.randn(order_count, dtype=dtype) * 0.45)
        local_features = torch.randn(local_count, 5, dtype=dtype)
        local_hidden = model.bounded_encoding(local_features)
        weights = model.classifier.weight.detach()
        bias = model.classifier.bias.detach()
        local_poly = polynomial_matrix(local_shift, model.theta)
        local_filtered = local_poly @ local_hidden
        local_logits = local_filtered @ weights.t() + bias

    predicted = local_logits.argmax(dim=1)
    class_count = int(weights.size(0))
    exact_minimum_gap = torch.full(
        (local_count, class_count), float("inf"), dtype=dtype
    )
    exact_maximum_absolute_change = torch.zeros(
        (local_count, class_count), dtype=dtype
    )
    completion_count = 0
    total_nodes = local_count + remote_count

    for full_edges in completion_edge_indices(boundary_counts, remote_count):
        completion_count += 1
        full_shift, _ = row_normalized_shift(
            full_edges,
            total_nodes,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        full_poly = polynomial_matrix(full_shift, model.theta)
        coefficients = full_poly[:local_count].clone()
        coefficients[:, :local_count] -= local_poly
        for node in range(local_count):
            own = int(predicted[node])
            coeff_local = coefficients[node, :local_count]
            coeff_remote = coefficients[node, local_count:]
            for competitor in range(class_count):
                if competitor == own:
                    continue
                direction = weights[own] - weights[competitor]
                local_change = torch.sum(
                    coeff_local[:, None]
                    * local_hidden
                    * direction[None, :]
                )
                remote_lower, remote_upper = remote_scalar_interval(
                    coeff_remote, direction
                )
                change_lower = local_change + remote_lower
                change_upper = local_change + remote_upper
                nominal_gap = (
                    local_logits[node, own] - local_logits[node, competitor]
                )
                exact_minimum_gap[node, competitor] = torch.minimum(
                    exact_minimum_gap[node, competitor],
                    nominal_gap + change_lower,
                )
                exact_maximum_absolute_change[node, competitor] = torch.maximum(
                    exact_maximum_absolute_change[node, competitor],
                    torch.maximum(change_lower.abs(), change_upper.abs()),
                )

    own_mask = torch.nn.functional.one_hot(
        predicted, num_classes=class_count
    ).bool()
    exact_minimum_gap = exact_minimum_gap.masked_fill(own_mask, float("inf"))
    exact_worst_gap = exact_minimum_gap.min(dim=1).values
    exact_robust = exact_worst_gap > tolerance

    stats = {
        "retained_symmetric": certificate_statistics(model, local_logits, basis),
        "cone_raw": cone_pairwise_certificate_statistics(
            model,
            local_features,
            local_shift,
            budget,
            local_logits,
            basis,
            count_semantics="exact",
            combine_symmetric=False,
        ),
        "cone_hybrid": cone_pairwise_certificate_statistics(
            model,
            local_features,
            local_shift,
            budget,
            local_logits,
            basis,
            count_semantics="exact",
            combine_symmetric=True,
        ),
    }

    rows: list[dict] = []
    for variant, result in stats.items():
        certified = result.worst_slack > tolerance
        certified_count = int(certified.sum())
        exact_count = int(exact_robust.sum())
        lower_bound_gap = exact_worst_gap - result.worst_slack
        finite = torch.isfinite(lower_bound_gap)
        positive = (
            finite
            & (exact_worst_gap > tolerance)
            & (result.worst_slack > tolerance)
        )
        ratios = (
            result.worst_slack[positive] / exact_worst_gap[positive]
        ).detach().cpu().numpy()
        gaps = lower_bound_gap[finite].detach().cpu().numpy()
        rows.append(
            {
                "seed": seed,
                "order_count": order_count,
                "variant": variant,
                "completion_count": completion_count,
                "local_node_count": local_count,
                "exact_robust_node_count": exact_count,
                "certified_node_count": certified_count,
                "certified_and_exact_robust_count": int(
                    (certified & exact_robust).sum()
                ),
                "certificate_recall": safe_divide(
                    int((certified & exact_robust).sum()), exact_count
                ),
                "certificate_precision": safe_divide(
                    int((certified & exact_robust).sum()), certified_count
                ),
                "soundness_violations": int((certified & ~exact_robust).sum()),
                "lower_bound_violations": int(
                    (result.worst_slack > exact_worst_gap + tolerance).sum()
                ),
                "mean_exact_minus_certificate_slack": (
                    float(np.mean(gaps)) if gaps.size else float("nan")
                ),
                "maximum_exact_minus_certificate_slack": (
                    float(np.max(gaps)) if gaps.size else float("nan")
                ),
                "median_certificate_to_exact_positive_gap_ratio": (
                    float(np.median(ratios)) if ratios.size else float("nan")
                ),
                "minimum_exact_worst_gap": float(
                    exact_worst_gap.detach().min()
                ),
                "mean_exact_pairwise_absolute_change": float(
                    exact_maximum_absolute_change[~own_mask].detach().mean()
                ),
            }
        )
    return rows


def aggregate(rows: list[dict], variant: str) -> dict:
    current = [row for row in rows if row["variant"] == variant]
    exact_total = sum(int(row["exact_robust_node_count"]) for row in current)
    certified_total = sum(int(row["certified_node_count"]) for row in current)
    overlap = sum(
        int(row["certified_and_exact_robust_count"]) for row in current
    )
    finite_ratio = np.asarray(
        [row["median_certificate_to_exact_positive_gap_ratio"] for row in current],
        dtype=float,
    )
    finite_ratio = finite_ratio[np.isfinite(finite_ratio)]
    return {
        "variant": variant,
        "trials": len(current),
        "enumerated_graph_completions": sum(
            int(row["completion_count"]) for row in current
        ),
        "exact_robust_nodes": exact_total,
        "certified_nodes": certified_total,
        "certificate_recall": safe_divide(overlap, exact_total),
        "certificate_precision": safe_divide(overlap, certified_total),
        "soundness_violations": sum(
            int(row["soundness_violations"]) for row in current
        ),
        "lower_bound_violations": sum(
            int(row["lower_bound_violations"]) for row in current
        ),
        "mean_exact_minus_certificate_slack": float(
            np.nanmean(
                [row["mean_exact_minus_certificate_slack"] for row in current]
            )
        ),
        "median_trial_certificate_to_exact_positive_gap_ratio": (
            float(np.median(finite_ratio)) if finite_ratio.size else float("nan")
        ),
    }


def bootstrap_recall(
    rows: list[dict], variant: str, *, draws: int, seed: int
) -> tuple[float, float]:
    current = [row for row in rows if row["variant"] == variant]
    seed_groups: dict[int, list[dict]] = defaultdict(list)
    for row in current:
        seed_groups[int(row["seed"])].append(row)
    cluster_ids = sorted(seed_groups)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        numerator = sum(
            int(row["certified_and_exact_robust_count"])
            for index in selected
            for row in seed_groups[cluster_ids[int(index)]]
        )
        denominator = sum(
            int(row["exact_robust_node_count"])
            for index in selected
            for row in seed_groups[cluster_ids[int(index)]]
        )
        samples.append(safe_divide(numerator, denominator))
    return tuple(np.nanpercentile(samples, [2.5, 97.5]).tolist())


def bootstrap_recall_gain(
    rows: list[dict], *, draws: int, seed: int
) -> tuple[float, float]:
    lookup = {
        (int(row["seed"]), int(row["order_count"]), row["variant"]): row
        for row in rows
    }
    seed_ids = sorted({int(row["seed"]) for row in rows})
    orders_by_seed = {
        seed_id: sorted(
            {int(row["order_count"]) for row in rows if int(row["seed"]) == seed_id}
        )
        for seed_id in seed_ids
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        chosen = rng.integers(0, len(seed_ids), size=len(seed_ids))
        counts = {}
        for variant in ("retained_symmetric", "cone_hybrid"):
            numerator = denominator = 0
            for index in chosen:
                seed_id = seed_ids[int(index)]
                for order_count in orders_by_seed[seed_id]:
                    row = lookup[(seed_id, order_count, variant)]
                    numerator += int(row["certified_and_exact_robust_count"])
                    denominator += int(row["exact_robust_node_count"])
            counts[variant] = safe_divide(numerator, denominator)
        samples.append(counts["cone_hybrid"] - counts["retained_symmetric"])
    return tuple(np.nanpercentile(samples, [2.5, 97.5]).tolist())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(table: list[dict], output: Path) -> None:
    labels = ["Symmetric", "ReLU-cone only", "Combined"]
    variants = ["retained_symmetric", "cone_raw", "cone_hybrid"]
    lookup = {row["variant"]: row for row in table}
    recall = [100.0 * lookup[name]["certificate_recall"] for name in variants]
    fig, axis = plt.subplots(1, 1, figsize=(4.8, 3.2))
    colors = ["#6B7280", "#3B82F6", "#0F766E"]
    bars = axis.bar(labels, recall, color=colors)
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_ylabel("Recall among exactly robust nodes (%)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=10)
    fig.suptitle("Exact worst-case evaluation on finite completions")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_e25_exact_cone_oracle.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = args.output_root.resolve()
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    tolerance = float(config["required_audits"]["float64_absolute_tolerance"])

    rows: list[dict] = []
    for seed in config["seeds"]:
        for order_count in config["filter_orders"]:
            trial_rows = run_trial(
                seed=int(seed),
                order_count=int(order_count),
                boundary_counts=[int(value) for value in config["boundary_stub_counts"]],
                remote_count=int(config["remote_node_count"]),
                tolerance=tolerance,
            )
            rows.extend(trial_rows)
            print(
                f"[DONE] seed={seed} order={order_count} "
                f"completions={trial_rows[0]['completion_count']}"
            )

    variants = list(config["certificate_variants"])
    table = [aggregate(rows, variant) for variant in variants]
    draws = int(config["bootstrap"]["draws"])
    bootstrap_seed = int(config["bootstrap"]["seed"])
    for index, row in enumerate(table):
        low, high = bootstrap_recall(
            rows, row["variant"], draws=draws, seed=bootstrap_seed + index
        )
        row["certificate_recall_ci_low"] = low
        row["certificate_recall_ci_high"] = high

    lookup = {row["variant"]: row for row in table}
    gain = (
        lookup["cone_hybrid"]["certificate_recall"]
        - lookup["retained_symmetric"]["certificate_recall"]
    )
    gain_ci = bootstrap_recall_gain(rows, draws=draws, seed=bootstrap_seed + 99)
    soundness = sum(int(row["soundness_violations"]) for row in rows)
    lower_bound = sum(int(row["lower_bound_violations"]) for row in rows)
    expected_trials = int(config["completion_family"]["expected_trials"])
    expected_completions = int(
        config["completion_family"]["expected_total_graph_completions"]
    )
    complete = (
        len(rows) == expected_trials * len(variants)
        and sum(int(row["completion_count"]) for row in rows if row["variant"] == variants[0])
        == expected_completions
        and soundness == 0
        and lower_bound == 0
    )
    summary = {
        "protocol": config["protocol"],
        "complete": complete,
        "decision": "exact_cone_oracle_audit_pass" if complete else "exact_cone_oracle_audit_fail",
        "config_sha256": sha256(CONFIG),
        "trials": expected_trials,
        "enumerated_graph_completions": expected_completions,
        "exact_remote_positive_ball_optimization": True,
        "soundness_violations": soundness,
        "lower_bound_violations": lower_bound,
        "symmetric_recall": lookup["retained_symmetric"]["certificate_recall"],
        "cone_raw_recall": lookup["cone_raw"]["certificate_recall"],
        "cone_hybrid_recall": lookup["cone_hybrid"]["certificate_recall"],
        "cone_hybrid_minus_symmetric_recall": gain,
        "cone_hybrid_minus_symmetric_recall_95ci": list(gain_ci),
        "cone_hybrid_precision": lookup["cone_hybrid"]["certificate_precision"],
        "training_or_selection_performed": False,
        "test_labels_used": False,
    }
    write_csv(analysis / "paper_data_e25_exact_cone_trials.csv", rows)
    write_csv(analysis / "paper_table_e25_exact_cone_oracle.csv", table)
    (analysis / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot(table, analysis / "figures")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        raise SystemExit("E25 exact cone oracle audit failed")


if __name__ == "__main__":
    main()
