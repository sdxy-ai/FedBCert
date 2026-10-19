"""Confirmation-only gates/statistics, reusing immutable E28b development code."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fedbcert_e28b_common import (
    ROOT, MODEL_NAMES, evaluate_validation, sha256, tensor_digest,
)

PROTOCOL = ROOT / "configs/fedbcert_e28b_confirmation_protocol.json"


def load_protocol():
    specification = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    base_path = ROOT / specification["base_protocol"]
    if sha256(base_path) != specification["base_protocol_sha256"]:
        raise RuntimeError("base scientific protocol changed")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    return specification, base


def matrix(specification):
    c = specification["confirmation"]
    return [dict(seed=seed, h=h, rho=rho, model=name,
                 graph_id=f"h{h:.2f}_rho{rho:.2f}_seed{seed}")
            for seed in c["seeds"] for h in c["homophily"]
            for rho in c["cross_client_mass"] for name in c["models"]]


def validate_specification(spec, base):
    c = spec["confirmation"]
    if c["seeds"] != base["seeds"]["confirmation_reserved_only"]:
        raise RuntimeError("confirmation seed replacement")
    if c["homophily"] != base["generator"]["homophily_grid"] or c["cross_client_mass"] != base["generator"]["cross_client_mass_grid"]:
        raise RuntimeError("confirmation grid changed")
    if c["models"] != list(MODEL_NAMES) or len(matrix(spec)) != c["training_runs"] or c["training_runs"] != 288:
        raise RuntimeError("confirmation matrix contract")
    if len({unit["graph_id"] for unit in matrix(spec)}) != c["graph_units"] or c["graph_units"] != 96:
        raise RuntimeError("confirmation graph count contract")
    if set(c["seeds"]) & set(base["seeds"]["development"] + base["seeds"]["engineering"]):
        raise RuntimeError("seed namespaces overlap")
    if c["primary_cell"] != {"h": .9, "rho": .4}:
        raise RuntimeError("locked primary cell changed")


def check_index(root, index):
    for relative, expected in index.items():
        if sha256(Path(root) / relative) != expected:
            raise RuntimeError(f"committed artifact hash mismatch: {relative}")


def require_test_unlock(output, spec):
    """Whole-matrix checkpoint commit is mandatory before first test target use."""
    path = Path(output) / "checkpoint_commit.json"
    if not path.is_file():
        raise RuntimeError("test remains sealed: checkpoint commit absent")
    commit = json.loads(path.read_text(encoding="utf-8"))
    if not commit.get("all_training_complete") or commit["training_runs"] != spec["confirmation"]["training_runs"]:
        raise RuntimeError("test remains sealed: incomplete training")
    if commit["matrix"] != matrix(spec) or commit["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("test remains sealed: protocol/matrix mismatch")
    expected = set()
    for unit in matrix(spec):
        prefix = f"training/{unit['graph_id']}"
        expected.update(f"{prefix}/{unit['model']}/{name}" for name in (
            "initial_state.pt", "selected_state.pt", "validation_trajectory.json", "training_record.json"))
        expected.update(f"{prefix}/{name}" for name in ("sealed_graph_bundle.pt", "graph_manifest.json"))
    if commit.get("graph_units") != spec["confirmation"]["graph_units"] or set(commit["artifact_sha256"]) != expected:
        raise RuntimeError("test remains sealed: missing/unexpected committed training artifacts")
    if commit.get("test_evaluated_before_commit") is not False:
        raise RuntimeError("test remains sealed: pre-commit test isolation mismatch")
    check_index(output, commit["artifact_sha256"])
    return commit


def test_role_projection(data):
    """Evaluation ONLY: expose original test nodes through legacy evaluator API.

    Graph, features, model and true split arrays in the saved bundle do not
    change. Never pass this projected view to training/checkpoint selection.
    """
    projected = dict(data)
    projected["masks"] = dict(data["masks"])
    projected["masks"]["validation"] = data["masks"]["test_sealed"]
    projected["masks"]["test_sealed"] = data["masks"]["validation"]
    return projected


def evaluate_test(base, data, name, state, *, commit_verified):
    if commit_verified is not True:
        raise RuntimeError("test remains sealed: caller has not verified commit")
    before = tensor_digest(state)
    rows, evidence, audit = evaluate_validation(base, test_role_projection(data), name, state)
    nodes = evidence["nodes"]
    if len(nodes) != int(data["masks"]["test_sealed"].sum()) or not bool(data["masks"]["test_sealed"][nodes].all()):
        raise RuntimeError("test role isolation mismatch")
    if bool((data["masks"]["validation"] | data["masks"]["train"] | data["masks"]["audit_unused"])[nodes].any()):
        raise RuntimeError("test evidence contains non-test nodes")
    if tensor_digest(state) != before:
        raise RuntimeError("test evaluation mutated checkpoint")
    return rows, evidence, audit


def exact_sign_p(values, null_margin=0.0):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("sign test requires complete finite seed values")
    successes = int((values > null_margin).sum())
    n = len(values)
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / (2 ** n), successes


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(p) - rank) * float(p[index]))
        adjusted[index] = min(1.0, maximum)
    return adjusted.tolist()


def resample_indices(spec):
    b = spec["statistics"]["bootstrap"]
    n = len(spec["confirmation"]["seeds"])
    return np.random.default_rng(b["seed"]).integers(0, n, size=(b["draws"], n))


def estimate(values, indices):
    if all(value is None for value in values):
        return dict(mean=None, ci95=None, seed_count=len(values), undefined=True)
    if any(value is None for value in values):
        raise ValueError("partial missing subgroup cannot be imputed")
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all() or len(vector) != indices.shape[1]:
        raise ValueError("incomplete/nonfinite bootstrap unit values")
    bootstrap_means = vector[indices].mean(1)
    return dict(mean=float(vector.mean()), ci95=np.quantile(bootstrap_means, [.025, .975]).tolist(),
                seed_count=len(vector), undefined=False)


def row(record, group, certificate=None):
    entries = [r for r in record["rows"] if r["group"] == group and (certificate is None or r["certificate"] == certificate)]
    if not entries:
        raise ValueError("missing group/certificate")
    return entries[0]


def paired_metrics(p, identity, mlp):
    all_p = row(p, "all", "cone_hybrid")
    exposure = row(p, "exposed", "cone_hybrid")
    symmetric = row(p, "exposed", "retained_pairwise")
    all_i, all_mlp = row(identity, "all"), row(mlp, "all")
    n = exposure["n"]
    difference = lambda a, b: None if a is None or b is None else a - b
    return dict(
        p_macro_f1=all_p["macro_f1"], i_macro_f1=all_i["macro_f1"], mlp_macro_f1=all_mlp["macro_f1"],
        p_minus_i_all_macro_f1=difference(all_p["macro_f1"], all_i["macro_f1"]),
        p_minus_i_exposed_macro_f1=difference(exposure["macro_f1"], row(identity, "exposed")["macro_f1"]),
        p_minus_mlp_all_macro_f1=difference(all_p["macro_f1"], all_mlp["macro_f1"]),
        test_exposure_fraction=n / all_p["n"],
        exposed_softmax_change_fraction=exposure["softmax_change_fraction"],
        exposed_category_flip_fraction=exposure["category_flip_fraction"],
        hybrid_exposed_certified_ratio=exposure["certified_ratio"],
        symmetric_exposed_certified_ratio=symmetric["certified_ratio"],
        hybrid_minus_symmetric_exposed_ratio=difference(exposure["certified_ratio"], symmetric["certified_ratio"]),
        hybrid_all_certified_ratio=all_p["certified_ratio"],
        symmetric_all_certified_ratio=row(p, "all", "retained_pairwise")["certified_ratio"],
        hybrid_exposed_certified_accuracy=exposure["certified_accuracy"],
    )


def primary_decision(primary, tests, base, reliability_ok=True, resources_ok=True):
    from analyze_fedbcert_e28b_controlled import cell_gates
    values = {key: value["mean"] for key, value in primary["estimates"].items()}
    values["validation_exposure_fraction"] = values["test_exposure_fraction"]  # Legacy gate API only.
    gates = cell_gates(values, base["stopping"]["development_cell_gate"])
    gates["C3_reliability"] = bool(reliability_ok)
    gates["R_resources"] = bool(resources_ok)
    required = {"p_minus_i_all_macro_f1", "p_minus_i_exposed_macro_f1", "p_minus_mlp_all_macro_f1", "hybrid_minus_symmetric_exposed_ratio"}
    gates["primary_holm_and_intervals"] = (len(tests) == 4 and {test["metric"] for test in tests} == required
        and all(test["holm_p"] <= .05 and test["ci95"][0] > test["null_margin"] for test in tests))
    return gates, all(gates.values())
