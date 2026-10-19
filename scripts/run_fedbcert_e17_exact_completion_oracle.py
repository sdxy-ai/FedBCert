#!/usr/bin/env python3
"""Run the frozen E17 exact finite-completion oracle audit.

The predictor is fixed within each trial.  We enumerate every physical graph
completion in the frozen toy family and analytically optimize every remote
initial hidden row over the unit L2 ball.  The resulting minimum pairwise gaps
are exact for that finite completion family, so the script can report both
soundness and certificate recall among exactly robust nodes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openfgl.flcore.fedbcert import (  # noqa: E402
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    retained_mass_uncertainty_basis,
    row_normalized_shift,
    uncertainty_basis,
)


CONFIG = Path("configs/fedbcert_e17_exact_oracle_protocol.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=(
            "results_boundary_certified/"
            "boundary_certified_spectral_e17_exact_oracle"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def local_edge_index() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )


def completion_edge_indices(
    boundary_counts: list[int], remote_count: int
):
    local_count = len(boundary_counts)
    remote_nodes = list(range(local_count, local_count + remote_count))
    choices = [
        list(itertools.combinations(remote_nodes, count))
        for count in boundary_counts
    ]
    remote_pairs = list(itertools.combinations(remote_nodes, 2))
    base_pairs = [(0, 1), (1, 2), (2, 3)]
    for endpoints in itertools.product(*choices):
        boundary_pairs = [
            (local, remote)
            for local, selected in enumerate(endpoints)
            for remote in selected
        ]
        for mask in range(1 << len(remote_pairs)):
            pairs = base_pairs + boundary_pairs + [
                pair
                for bit, pair in enumerate(remote_pairs)
                if mask & (1 << bit)
            ]
            directed = list(pairs) + [(right, left) for left, right in pairs]
            yield torch.as_tensor(directed, dtype=torch.long).t().contiguous()


def polynomial_matrix(shift: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    dense = shift.to_dense().to(torch.float64)
    current = torch.eye(dense.size(0), dtype=torch.float64)
    output = theta[0].to(torch.float64) * current
    for order in range(1, int(theta.numel())):
        current = dense @ current
        output = output + theta[order].to(torch.float64) * current
    return output


def radius_from_basis(
    theta: torch.Tensor, basis: list[torch.Tensor]
) -> torch.Tensor:
    radius = torch.zeros_like(basis[0], dtype=torch.float64)
    for order in range(1, len(basis)):
        radius = radius + theta[order].abs().to(torch.float64) * basis[order]
    return radius


def certificate_values(
    logits: torch.Tensor,
    weights: torch.Tensor,
    radius: torch.Tensor,
    *,
    pairwise: bool,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = logits.argmax(dim=1)
    own_logits = logits.gather(1, predicted[:, None])
    own_weights = weights[predicted]
    if pairwise:
        multiplier = (
            own_weights[:, None, :] - weights[None, :, :]
        ).norm(p=2, dim=2)
    else:
        multiplier = own_weights.norm(p=2, dim=1)[:, None] + weights.norm(
            p=2, dim=1
        )[None, :]
    bounds = radius[:, None] * multiplier
    slack = own_logits - logits - bounds
    own = torch.nn.functional.one_hot(
        predicted, num_classes=logits.size(1)
    ).bool()
    slack = slack.masked_fill(own, float("inf"))
    worst_slack = slack.min(dim=1).values
    return worst_slack > tolerance, worst_slack, bounds


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def run_trial(
    *,
    seed: int,
    order_count: int,
    boundary_counts: list[int],
    remote_count: int,
    tolerance: float,
) -> tuple[list[dict], dict]:
    torch.manual_seed(seed * 1009 + order_count)
    local_count = len(boundary_counts)
    local_shift, local_degree = row_normalized_shift(
        local_edge_index(),
        local_count,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    missing = torch.tensor(boundary_counts, dtype=torch.float64)
    budget = boundary_uncertainty(local_degree, missing)
    bases = {
        "retained": retained_mass_uncertainty_basis(
            local_shift, budget, order_count
        ),
        "legacy": uncertainty_basis(local_shift, budget, order_count),
    }

    model = PolynomialSpectralClassifier(
        input_dim=5,
        hidden_dim=6,
        num_classes=3,
        order_count=order_count,
        dropout=0.0,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    with torch.no_grad():
        model.theta.copy_(torch.randn(order_count) * 0.45)
        local_features = torch.randn(local_count, 5)
        local_hidden = model.bounded_encoding(local_features).to(torch.float64)
        weights = model.classifier.weight.detach().to(torch.float64)
        bias = model.classifier.bias.detach().to(torch.float64)
        local_poly = polynomial_matrix(local_shift, model.theta)
        local_filtered = local_poly @ local_hidden
        local_logits = local_filtered @ weights.t() + bias

    predicted = local_logits.argmax(dim=1)
    class_count = int(weights.size(0))
    exact_minimum_gap = torch.full(
        (local_count, class_count), float("inf"), dtype=torch.float64
    )
    exact_maximum_absolute_change = torch.zeros(
        (local_count, class_count), dtype=torch.float64
    )
    completion_count = 0
    total_nodes = local_count + remote_count

    for full_edges in completion_edge_indices(boundary_counts, remote_count):
        completion_count += 1
        full_shift, _ = row_normalized_shift(
            full_edges,
            total_nodes,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        full_poly = polynomial_matrix(full_shift, model.theta)
        coefficients = full_poly[:local_count].clone()
        coefficients[:, :local_count] -= local_poly
        for node in range(local_count):
            own = int(predicted[node])
            coeff_local = coefficients[node, :local_count]
            remote_mass = coefficients[node, local_count:].abs().sum()
            for competitor in range(class_count):
                if competitor == own:
                    continue
                difference_weight = weights[own] - weights[competitor]
                local_change = torch.sum(
                    coeff_local[:, None]
                    * local_hidden
                    * difference_weight[None, :]
                )
                remote_worst = remote_mass * difference_weight.norm(p=2)
                nominal_gap = local_logits[node, own] - local_logits[node, competitor]
                minimum_gap = nominal_gap + local_change - remote_worst
                maximum_absolute_change = local_change.abs() + remote_worst
                exact_minimum_gap[node, competitor] = torch.minimum(
                    exact_minimum_gap[node, competitor], minimum_gap
                )
                exact_maximum_absolute_change[node, competitor] = torch.maximum(
                    exact_maximum_absolute_change[node, competitor],
                    maximum_absolute_change,
                )

    own = torch.nn.functional.one_hot(predicted, num_classes=class_count).bool()
    exact_minimum_gap = exact_minimum_gap.masked_fill(own, float("inf"))
    exact_worst_gap = exact_minimum_gap.min(dim=1).values
    exact_robust = exact_worst_gap > tolerance

    variants = (
        ("retained_pairwise", "retained", True),
        ("legacy_pairwise", "legacy", True),
        ("retained_classwise", "retained", False),
        ("legacy_classwise", "legacy", False),
    )
    rows: list[dict] = []
    for variant, recurrence, pairwise in variants:
        radius = radius_from_basis(model.theta, bases[recurrence])
        certified, worst_slack, bounds = certificate_values(
            local_logits,
            weights,
            radius,
            pairwise=pairwise,
            tolerance=tolerance,
        )
        evaluated = ~own
        positive_bounds = evaluated & (bounds > tolerance)
        ratios = (
            exact_maximum_absolute_change[positive_bounds]
            / bounds[positive_bounds]
        ).detach().cpu().numpy()
        soundness_violations = int((certified & ~exact_robust).sum())
        bound_violations = int(
            (
                exact_maximum_absolute_change[positive_bounds]
                > bounds[positive_bounds] + tolerance
            ).sum()
        )
        certified_count = int(certified.sum())
        exact_count = int(exact_robust.sum())
        rows.append({
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
            "soundness_violations": soundness_violations,
            "pairwise_bound_violations": bound_violations,
            "mean_representation_radius": float(radius.detach().mean()),
            "minimum_certificate_slack": float(worst_slack.detach().min()),
            "minimum_exact_worst_gap": float(exact_worst_gap.detach().min()),
            "maximum_exact_to_certificate_pairwise_ratio": (
                float(np.max(ratios)) if ratios.size else float("nan")
            ),
            "median_exact_to_certificate_pairwise_ratio": (
                float(np.median(ratios)) if ratios.size else float("nan")
            ),
        })
    trial = {
        "seed": seed,
        "order_count": order_count,
        "completion_count": completion_count,
        "exact_robust_node_count": int(exact_robust.sum()),
        "local_node_count": local_count,
    }
    return rows, trial


def mean(values) -> float:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    return float(data.mean()) if data.size else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], group_key: str | None = None) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["variant"], row[group_key] if group_key else "overall")
        groups[key].append(row)
    output: list[dict] = []
    for (variant, group_value), current in sorted(groups.items()):
        exact_total = sum(int(row["exact_robust_node_count"]) for row in current)
        certified_total = sum(int(row["certified_node_count"]) for row in current)
        true_certified = sum(
            int(row["certified_and_exact_robust_count"]) for row in current
        )
        result = {
            "variant": variant,
            "trials": len(current),
            "enumerated_graph_completions": sum(
                int(row["completion_count"]) for row in current
            ),
            "exact_robust_nodes": exact_total,
            "certified_nodes": certified_total,
            "certificate_recall": safe_divide(true_certified, exact_total),
            "certificate_precision": safe_divide(true_certified, certified_total),
            "soundness_violations": sum(
                int(row["soundness_violations"]) for row in current
            ),
            "pairwise_bound_violations": sum(
                int(row["pairwise_bound_violations"]) for row in current
            ),
            "mean_representation_radius": mean(
                row["mean_representation_radius"] for row in current
            ),
            "maximum_exact_to_certificate_pairwise_ratio": max(
                float(row["maximum_exact_to_certificate_pairwise_ratio"])
                for row in current
            ),
            "median_trial_exact_to_certificate_pairwise_ratio": mean(
                row["median_exact_to_certificate_pairwise_ratio"]
                for row in current
            ),
        }
        if group_key:
            result[group_key] = group_value
        output.append(result)
    return output


def bootstrap_recall_statistics(
    rows: list[dict], *, draws: int, seed: int
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Seed-clustered paired bootstrap for recalls and certificate gains."""
    variants = sorted({str(row["variant"]) for row in rows})
    seed_ids = sorted({int(row["seed"]) for row in rows})
    orders_by_seed = {
        seed_id: sorted(
            {int(row["order_count"]) for row in rows if int(row["seed"]) == seed_id}
        )
        for seed_id in seed_ids
    }
    lookup = {
        (int(row["seed"]), int(row["order_count"]), str(row["variant"])): row
        for row in rows
    }
    generator = np.random.default_rng(seed)
    samples = {variant: np.empty(draws, dtype=float) for variant in variants}
    for draw in range(draws):
        selected = generator.integers(0, len(seed_ids), size=len(seed_ids))
        for variant in variants:
            numerator = 0
            denominator = 0
            for index in selected:
                seed_id = seed_ids[int(index)]
                for order_count in orders_by_seed[seed_id]:
                    row = lookup[(seed_id, order_count, variant)]
                    numerator += int(row["certified_and_exact_robust_count"])
                    denominator += int(row["exact_robust_node_count"])
            samples[variant][draw] = numerator / denominator if denominator else np.nan
    intervals = {
        variant: tuple(
            float(value)
            for value in np.nanquantile(samples[variant], [0.025, 0.975])
        )
        for variant in variants
    }
    baseline = samples["legacy_classwise"]
    gains = {
        variant: tuple(
            float(value)
            for value in np.nanquantile(samples[variant] - baseline, [0.025, 0.975])
        )
        for variant in variants
    }
    return intervals, gains


def plot_results(table: list[dict], figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    labels = {
        "retained_pairwise": "Retained\n+ pairwise",
        "legacy_pairwise": "Legacy\n+ pairwise",
        "retained_classwise": "Retained\n+ classwise",
        "legacy_classwise": "Legacy\n+ classwise",
    }
    order = list(labels)
    lookup = {row["variant"]: row for row in table}
    recall = [100.0 * float(lookup[name]["certificate_recall"]) for name in order]
    recall_low = [
        100.0 * float(lookup[name]["certificate_recall_ci_low"])
        for name in order
    ]
    recall_high = [
        100.0 * float(lookup[name]["certificate_recall_ci_high"])
        for name in order
    ]
    recall_error = np.asarray([
        [value - low for value, low in zip(recall, recall_low)],
        [high - value for value, high in zip(recall, recall_high)],
    ])
    radius = [float(lookup[name]["mean_representation_radius"]) for name in order]
    colors = ["#0072B2", "#56B4E9", "#E69F00", "#999999"]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    positions = np.arange(len(order))
    bars = axes[0].bar(
        positions,
        recall,
        color=colors,
        yerr=recall_error,
        capsize=3,
        error_kw={"linewidth": 1.0},
    )
    axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axes[0].set_ylabel("Recall among exactly robust nodes (%)")
    axes[0].set_xticks(positions, [labels[name] for name in order])
    axes[0].set_ylim(0, max(100.0, max(recall) * 1.12))
    axes[0].grid(axis="y", alpha=0.25)
    radius_bars = axes[1].bar(positions, radius, color=colors)
    axes[1].bar_label(radius_bars, fmt="%.2f", padding=3, fontsize=8)
    axes[1].set_ylabel("Mean representation radius")
    axes[1].set_xticks(positions, [labels[name] for name in order])
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("E17 exact finite-completion oracle")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figure_root / f"figure_e17_exact_oracle.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    output = Path(args.output_root)
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    tolerance = float(config["required_audits"]["float64_absolute_tolerance"])
    all_rows: list[dict] = []
    trials: list[dict] = []
    for seed in config["seeds"]:
        for order_count in config["filter_orders"]:
            rows, trial = run_trial(
                seed=int(seed),
                order_count=int(order_count),
                boundary_counts=list(config["boundary_stub_counts"]),
                remote_count=int(config["remote_node_count"]),
                tolerance=tolerance,
            )
            all_rows.extend(rows)
            trials.append(trial)
            print(
                f"[E17] seed={seed} K={order_count} "
                f"completions={trial['completion_count']} "
                f"exact_robust={trial['exact_robust_node_count']}/"
                f"{trial['local_node_count']}"
            )

    table = aggregate(all_rows)
    by_order = aggregate(all_rows, "order_count")
    intervals, gain_intervals = bootstrap_recall_statistics(
        all_rows,
        draws=int(config["bootstrap"]["draws"]),
        seed=int(config["bootstrap"]["seed"]),
    )
    for row in table:
        low, high = intervals[str(row["variant"])]
        gain_low, gain_high = gain_intervals[str(row["variant"])]
        row["certificate_recall_ci_low"] = low
        row["certificate_recall_ci_high"] = high
        row["recall_gain_over_legacy_classwise_ci_low"] = gain_low
        row["recall_gain_over_legacy_classwise_ci_high"] = gain_high
    expected_trials = int(config["completion_family"]["expected_trials"])
    expected_completions = int(
        config["completion_family"]["expected_total_graph_completions"]
    )
    total_completions = sum(int(row["completion_count"]) for row in trials)
    soundness_violations = sum(
        int(row["soundness_violations"]) for row in all_rows
    )
    bound_violations = sum(
        int(row["pairwise_bound_violations"]) for row in all_rows
    )
    complete = (
        len(trials) == expected_trials
        and total_completions == expected_completions
        and soundness_violations
        <= int(config["required_audits"]["maximum_soundness_violations"])
        and bound_violations
        <= int(config["required_audits"]["maximum_bound_violations"])
    )
    main_row = next(
        row for row in table if row["variant"] == "retained_pairwise"
    )
    legacy_row = next(
        row for row in table if row["variant"] == "legacy_classwise"
    )
    summary = {
        "protocol": config["protocol"],
        "complete": complete,
        "decision": "exact_oracle_audit_pass" if complete else "exact_oracle_audit_fail",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "trials": len(trials),
        "enumerated_graph_completions": total_completions,
        "exact_remote_feature_optimization": True,
        "test_labels_used": False,
        "training_or_selection_performed": False,
        "soundness_violations": soundness_violations,
        "pairwise_bound_violations": bound_violations,
        "proposed_certificate_recall": main_row["certificate_recall"],
        "legacy_certificate_recall": legacy_row["certificate_recall"],
        "recall_gain_over_legacy": (
            float(main_row["certificate_recall"])
            - float(legacy_row["certificate_recall"])
        ),
        "recall_gain_over_legacy_ci": list(
            gain_intervals["retained_pairwise"]
        ),
        "recurrence_only_recall_gain": (
            float(main_row["certificate_recall"])
            - float(next(
                row for row in table if row["variant"] == "legacy_pairwise"
            )["certificate_recall"])
        ),
        "pairwise_only_recall_gain": (
            float(main_row["certificate_recall"])
            - float(next(
                row for row in table if row["variant"] == "retained_classwise"
            )["certificate_recall"])
        ),
        "maximum_proposed_exact_to_bound_ratio": main_row[
            "maximum_exact_to_certificate_pairwise_ratio"
        ],
    }
    write_csv(analysis / "paper_data_e17_trial_metrics.csv", all_rows)
    write_csv(analysis / "paper_table_e17_exact_oracle.csv", table)
    write_csv(analysis / "paper_table_e17_exact_oracle_by_order.csv", by_order)
    plot_results(table, analysis / "figures")
    (analysis / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not complete:
        raise SystemExit("E17 exact completion oracle audit failed")


if __name__ == "__main__":
    main()
