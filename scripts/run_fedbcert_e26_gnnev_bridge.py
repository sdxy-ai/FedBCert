#!/usr/bin/env python3
"""Run the frozen E26 GNNev common-predictor numerical bridge.

The official GNNev verifier sees a finite known endpoint universe. FedBCert
sees only a one-stub upper bound and therefore certifies a strict superset.
Both paths evaluate the same nonnegative two-layer mean-GraphSAGE predictor,
which is exactly the polynomial ``W P^2 H + q`` on every translated graph.
An exhaustive four-completion oracle independently audits GNNev's Boolean
decision and the graph/task translation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fedbcert_e26_known_universe_verifier_bridge.json"
RESULT_ROOT = (
    ROOT
    / "results_boundary_certified"
    / "boundary_certified_spectral_e26_known_universe_verifier"
)
GNNEV_ROOT = Path(
    os.environ.get("GNNEV_ROOT", ROOT / "third_party" / "GNNEV")
).resolve()
GUROBI_COMPAT = Path(
    os.environ.get("GUROBI_COMPAT", ROOT / ".vendor" / "gurobi12_compat")
).resolve()
DEFAULT_OUTPUT = RESULT_ROOT / "analysis"

# Keep the solver compatibility wheel isolated from the OpenFGL environment.
for entry in (ROOT, GNNEV_ROOT, GUROBI_COMPAT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import gurobipy as gp  # noqa: E402
from gnn import GraphSAGE_nc  # noqa: E402
import gnnev.encoding_nodeclassify as gnnev_nc  # noqa: E402
from gnnev.gnnev import GNNev  # noqa: E402

from openfgl.flcore.fedbcert.certificate import (  # noqa: E402
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    certificate_statistics,
    cone_pairwise_certificate_statistics,
    retained_mass_uncertainty_basis,
    row_normalized_shift,
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


def git_head(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_source_matches_commit_ignoring_eol(repo: Path) -> bool:
    """Accept a checkout whose only tracked difference is CRLF normalization."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            "--ignore-space-at-eol",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def percentile_bootstrap_mean(
    values: list[float], *, seed: int = 260026, samples: int = 10000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def directed_base_edges(local_edges: list[list[int]], node_count: int) -> list[tuple[int, int]]:
    edges = {(node, node) for node in range(node_count)}
    for left, right in local_edges:
        edges.add((int(left), int(right)))
        edges.add((int(right), int(left)))
    return sorted(edges)


def edge_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def positive_unit_rows(generator: torch.Generator, rows: int, cols: int) -> torch.Tensor:
    values = torch.rand((rows, cols), generator=generator, dtype=torch.float64) + 0.05
    return torch.nn.functional.normalize(values, p=2, dim=1)


def build_models(
    generator: torch.Generator,
    input_dim: int,
    class_count: int,
) -> tuple[GraphSAGE_nc, PolynomialSpectralClassifier, torch.Tensor, torch.Tensor]:
    weight = torch.randn(
        (class_count, input_dim), generator=generator, dtype=torch.float64
    )
    weight = torch.nn.functional.normalize(weight, p=2, dim=1)
    bias = 0.05 * torch.randn(class_count, generator=generator, dtype=torch.float64)

    official = GraphSAGE_nc(input_dim, input_dim, class_count, "mean", 2).double().eval()
    first = official.layer1.module_0
    second = official.layer2.module_0
    with torch.no_grad():
        first.lin_l.weight.copy_(torch.eye(input_dim, dtype=torch.float64))
        first.lin_l.bias.zero_()
        first.lin_r.weight.zero_()
        second.lin_l.weight.copy_(weight)
        second.lin_l.bias.copy_(bias)
        second.lin_r.weight.zero_()

    polynomial = PolynomialSpectralClassifier(
        input_dim=input_dim,
        hidden_dim=input_dim,
        num_classes=class_count,
        order_count=3,
        dropout=0.0,
        encoder_type="linear_relu",
        certificate_margin_mode="pairwise_difference",
    ).double().eval()
    with torch.no_grad():
        polynomial.encoder.weight.copy_(torch.eye(input_dim, dtype=torch.float64))
        polynomial.encoder.bias.zero_()
        polynomial.theta.copy_(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        polynomial.classifier.weight.copy_(weight)
        polynomial.classifier.bias.copy_(bias)
    return official, polynomial, weight, bias


class TinyDataset:
    def __init__(self, data: Data, num_features: int, num_classes: int) -> None:
        self.data = data
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)

    def __getitem__(self, index: int) -> Data:
        if index != 0:
            raise IndexError(index)
        return self.data


def run_gnnev(
    model: GraphSAGE_nc,
    data: Data,
    task: dict,
    num_features: int,
    num_classes: int,
) -> tuple[str, float]:
    # The official helper expects WLS credentials. An empty parameter mapping
    # activates the isolated valid restricted license without changing any
    # verifier equation or solver setting.
    gnnev_nc.get_gurobi_license = lambda _path: {"OutputFlag": 0}
    verifier = GNNev(
        model,
        TinyDataset(data, num_features, num_classes),
        task,
        log_file="unused.log",
    )
    verifier.logger = logging.getLogger(f"fedbcert-e26-{task['trial_seed']}")
    verifier.data = data
    verifier.node = int(task["node"])
    verifier.node_set = set(int(node) for node in task["task_nodes"])
    verifier.edge_set = set(tuple(map(int, edge)) for edge in task["task_edges"])
    verifier.fragile_edge_set_add = set(
        tuple(map(int, edge)) for edge in task["task_fragile_edges_add"]
    )
    verifier.fragile_edge_set_del = set(
        tuple(map(int, edge)) for edge in task["task_fragile_edges_del"]
    )
    start = time.perf_counter()
    verifier.nc_mip_solver_env()
    for layer_id in range(verifier.num_layers, 0, -1):
        verifier.nc_mip_encode_layer(layer_id - 1)
    result = verifier.nc_mip_check_sat()
    elapsed = time.perf_counter() - start
    verifier.grb.dispose()
    return result, elapsed


def exact_known_universe(
    model: GraphSAGE_nc,
    features: torch.Tensor,
    base_edges: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    target: int,
    predicted: int,
) -> tuple[bool, float, float]:
    minimum_gap = float("inf")
    maximum_logit_change = 0.0
    with torch.no_grad():
        base_logits = model(Data(x=features, edge_index=edge_tensor(base_edges)))[target]
        completions: list[list[tuple[int, int]]] = [base_edges]
        completions.extend([sorted(set(base_edges) | {edge}) for edge in candidates])
        for edges in completions:
            logits = model(Data(x=features, edge_index=edge_tensor(edges)))[target]
            maximum_logit_change = max(
                maximum_logit_change,
                float((logits - base_logits).abs().max()),
            )
            for competitor in range(int(logits.numel())):
                if competitor != predicted:
                    minimum_gap = min(
                        minimum_gap,
                        float(logits[predicted] - logits[competitor]),
                    )
    return minimum_gap > 1e-8, minimum_gap, maximum_logit_change


def run_trial(config: dict, seed: int) -> dict:
    model_cfg = config["model"]
    graph_cfg = config["graph"]
    feature_dim = int(model_cfg["input_dim"])
    class_count = int(model_cfg["num_classes"])
    local_count = int(graph_cfg["local_node_count"])
    remote_count = int(graph_cfg["remote_node_count"])
    total_count = local_count + remote_count
    target = int(graph_cfg["target_node"])
    candidates = [tuple(map(int, edge)) for edge in graph_cfg["candidate_directed_additions"]]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    features = positive_unit_rows(generator, total_count, feature_dim)
    official, polynomial, _, _ = build_models(generator, feature_dim, class_count)

    base_edges = directed_base_edges(graph_cfg["local_undirected_edges"], total_count)
    base_data = Data(x=features, edge_index=edge_tensor(base_edges))
    with torch.no_grad():
        official_logits = official(base_data)

    local_edge_index = torch.tensor(
        graph_cfg["local_undirected_edges"], dtype=torch.long
    ).t().contiguous()
    local_shift, local_degree = row_normalized_shift(
        local_edge_index,
        local_count,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    local_features = features[:local_count]
    with torch.no_grad():
        polynomial_logits = polynomial(local_features, local_shift)
        hidden = polynomial.bounded_encoding(local_features)
    predictor_logit_error = float(
        (official_logits[:local_count] - polynomial_logits).abs().max()
    )
    hidden_identity_error = float((hidden - local_features).abs().max())
    first_preactivation_minimum = float(local_features.min())
    predicted = int(official_logits[target].argmax())

    missing = torch.zeros(local_count, dtype=torch.float64)
    missing[target] = 1.0
    row_budget = boundary_uncertainty(local_degree, missing)
    basis = retained_mass_uncertainty_basis(local_shift, row_budget, 3)
    symmetric = certificate_statistics(polynomial, polynomial_logits, basis)
    cone = cone_pairwise_certificate_statistics(
        polynomial,
        local_features,
        local_shift,
        row_budget,
        polynomial_logits,
        basis,
        count_semantics="upper_bound",
        combine_symmetric=True,
    )
    fedbcert_symmetric = bool(symmetric.certified[target])
    fedbcert_cone = bool(cone.certified[target])

    exact_robust, exact_minimum_gap, maximum_logit_change = exact_known_universe(
        official,
        features,
        base_edges,
        candidates,
        target,
        predicted,
    )
    task = {
        "protocol": config["protocol"],
        "trial_seed": int(seed),
        "category": "node classification",
        "solver": "gurobi",
        "global_budget": int(graph_cfg["global_edge_budget"]),
        "local_budget": int(graph_cfg["local_edge_budget"]),
        "objective": "default",
        "predict": predicted,
        "label": -1,
        "iv": False,
        "tight_bound": True,
        "feature_budget_list": {},
        "feature_budget_all": 0.0,
        "node": target,
        "task_nodes": list(range(total_count)),
        "task_edges": [list(edge) for edge in base_edges],
        "task_fragile_edges_add": [list(edge) for edge in candidates],
        "task_fragile_edges_del": [],
    }
    gnnev_result, solver_seconds = run_gnnev(
        official,
        base_data,
        task,
        feature_dim,
        class_count,
    )
    gnnev_robust = gnnev_result == "unsat"
    return {
        "seed": int(seed),
        "predicted_class": predicted,
        "known_completion_count": 1 + len(candidates),
        "exact_known_robust": exact_robust,
        "gnnev_result": gnnev_result,
        "gnnev_known_robust": gnnev_robust,
        "fedbcert_symmetric_certified": fedbcert_symmetric,
        "fedbcert_cone_hybrid_certified": fedbcert_cone,
        "exact_minimum_pairwise_gap": exact_minimum_gap,
        "maximum_known_completion_logit_change": maximum_logit_change,
        "fedbcert_symmetric_worst_slack": float(
            symmetric.worst_slack[target].detach()
        ),
        "fedbcert_cone_hybrid_worst_slack": float(
            cone.worst_slack[target].detach()
        ),
        "predictor_maximum_absolute_logit_error": predictor_logit_error,
        "hidden_identity_maximum_absolute_error": hidden_identity_error,
        "first_preactivation_minimum": first_preactivation_minimum,
        "gnnev_solver_seconds": solver_seconds,
        "gnnev_matches_enumeration": gnnev_robust == exact_robust,
        "fedbcert_cone_sound_on_known_subset": not (fedbcert_cone and not exact_robust),
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    observed_commit = git_head(GNNEV_ROOT)
    rows = [run_trial(config, int(seed)) for seed in config["trial_seeds"]]

    exact_count = sum(bool(row["exact_known_robust"]) for row in rows)
    gnnev_count = sum(bool(row["gnnev_known_robust"]) for row in rows)
    symmetric_count = sum(bool(row["fedbcert_symmetric_certified"]) for row in rows)
    cone_count = sum(bool(row["fedbcert_cone_hybrid_certified"]) for row in rows)
    gnnev_agreement = sum(bool(row["gnnev_matches_enumeration"]) for row in rows)
    cone_overlap = sum(
        bool(row["fedbcert_cone_hybrid_certified"])
        and bool(row["exact_known_robust"])
        for row in rows
    )
    symmetric_overlap = sum(
        bool(row["fedbcert_symmetric_certified"])
        and bool(row["exact_known_robust"])
        for row in rows
    )
    unknown_count = sum(row["gnnev_result"] == "unknown" for row in rows)
    soundness_violations = sum(
        not bool(row["fedbcert_cone_sound_on_known_subset"]) for row in rows
    )
    robust_rows = [row for row in rows if bool(row["exact_known_robust"])]
    recall_gain_values = [
        float(bool(row["fedbcert_cone_hybrid_certified"]))
        - float(bool(row["fedbcert_symmetric_certified"]))
        for row in robust_rows
    ]
    recall_gain = float(np.mean(recall_gain_values)) if recall_gain_values else float("nan")
    recall_gain_ci = percentile_bootstrap_mean(recall_gain_values)
    maximum_logit_error = max(
        float(row["predictor_maximum_absolute_logit_error"]) for row in rows
    )
    maximum_hidden_error = max(
        float(row["hidden_identity_maximum_absolute_error"]) for row in rows
    )
    table = [
        {
            "verifier": "GNNev 1.1",
            "information_set": "known endpoints, features, and candidate edges",
            "robust_or_certified_count": gnnev_count,
            "recall_against_known_exact_robust": safe_ratio(gnnev_count, exact_count),
            "precision_against_known_exact_robust": safe_ratio(gnnev_count, gnnev_count),
            "soundness_violations": 0 if gnnev_agreement == len(rows) else len(rows) - gnnev_agreement,
        },
        {
            "verifier": "FedBCert symmetric",
            "information_set": "one endpoint-blind boundary-count upper bound",
            "robust_or_certified_count": symmetric_count,
            "recall_against_known_exact_robust": safe_ratio(symmetric_overlap, exact_count),
            "precision_against_known_exact_robust": safe_ratio(symmetric_overlap, symmetric_count),
            "soundness_violations": sum(
                bool(row["fedbcert_symmetric_certified"])
                and not bool(row["exact_known_robust"])
                for row in rows
            ),
        },
        {
            "verifier": "FedBCert cone-hybrid",
            "information_set": "one endpoint-blind boundary-count upper bound",
            "robust_or_certified_count": cone_count,
            "recall_against_known_exact_robust": safe_ratio(cone_overlap, exact_count),
            "precision_against_known_exact_robust": safe_ratio(cone_overlap, cone_count),
            "soundness_violations": soundness_violations,
        },
    ]
    errors = []
    if observed_commit != config["official_commit"]:
        errors.append("official GNNev commit mismatch")
    if not git_source_matches_commit_ignoring_eol(GNNEV_ROOT):
        errors.append("official GNNev source differs from the fixed commit beyond EOL normalization")
    if tuple(gp.gurobi.version()) != (12, 0, 3):
        errors.append("isolated Gurobi compatibility version mismatch")
    if maximum_logit_error > float(config["verification"]["numeric_tolerance"]):
        errors.append("GraphSAGE and polynomial predictor logits differ")
    if maximum_hidden_error > float(config["verification"]["numeric_tolerance"]):
        errors.append("positive feature encoder is not identity")
    if gnnev_agreement != len(rows):
        errors.append("GNNev decision disagrees with exhaustive known-universe oracle")
    if unknown_count:
        errors.append("GNNev returned UNKNOWN")
    if soundness_violations:
        errors.append("FedBCert certified a known-universe counterexample")
    if not all(float(row["first_preactivation_minimum"]) > 0.0 for row in rows):
        errors.append("first-layer ReLU identity premise failed")

    summary = {
        "protocol": config["protocol"],
        "complete": not errors and len(rows) == len(config["trial_seeds"]),
        "decision": "external_verifier_bridge_complete" if not errors else "external_verifier_bridge_failed",
        "errors": errors,
        "trial_count": len(rows),
        "known_completion_count": sum(int(row["known_completion_count"]) for row in rows),
        "official_repository": config["official_repository"],
        "official_commit_expected": config["official_commit"],
        "official_commit_observed": observed_commit,
        "official_commit_verified": observed_commit == config["official_commit"],
        "official_source_matches_commit_ignoring_eol": git_source_matches_commit_ignoring_eol(
            GNNEV_ROOT
        ),
        "gurobipy_version": ".".join(map(str, gp.gurobi.version())),
        "gurobi_solver_usable": True,
        "known_exact_robust_count": exact_count,
        "gnnev_robust_count": gnnev_count,
        "gnnev_enumeration_agreement_count": gnnev_agreement,
        "gnnev_unknown_results": unknown_count,
        "fedbcert_symmetric_certified_count": symmetric_count,
        "fedbcert_cone_hybrid_certified_count": cone_count,
        "fedbcert_cone_hybrid_recall_on_known_exact_robust": safe_ratio(cone_overlap, exact_count),
        "fedbcert_cone_hybrid_precision_on_known_subset": safe_ratio(cone_overlap, cone_count),
        "cone_hybrid_minus_symmetric_recall_on_known_exact_robust": recall_gain,
        "cone_hybrid_minus_symmetric_recall_95ci": list(recall_gain_ci),
        "fedbcert_soundness_violations_on_known_subset": soundness_violations,
        "maximum_predictor_logit_translation_error": maximum_logit_error,
        "maximum_hidden_identity_error": maximum_hidden_error,
        "mean_gnnev_solver_seconds": float(np.mean([row["gnnev_solver_seconds"] for row in rows])),
        "set_relation": config["set_relation"]["relation"],
        "test_labels_used": False,
        "model_or_hyperparameter_selection_performed": False,
        "config_sha256": sha256(CONFIG),
    }
    write_csv(args.output_root / "paper_data_e26_gnnev_bridge_trials.csv", rows)
    write_csv(args.output_root / "paper_table_e26_gnnev_bridge.csv", table)
    (args.output_root / "bridge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not summary["complete"]:
        raise SystemExit("E26 GNNev bridge audit failed")


if __name__ == "__main__":
    main()
