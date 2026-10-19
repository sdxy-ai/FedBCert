import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fedbcert_e28b_common import generate_graph, sha256, tensor_digest, train_model
from fedbcert_e28b_confirmation_common import (
    PROTOCOL, check_index, estimate, evaluate_test, exact_sign_p, holm_adjust,
    load_protocol, matrix, paired_metrics, primary_decision, require_test_unlock,
    resample_indices, test_role_projection as project_test_role, validate_specification,
)
from audit_fedbcert_e28b_confirmation import audit_raw_rows


@pytest.fixture
def protocols():
    torch.set_num_threads(1)
    return load_protocol()


@pytest.fixture
def graph(protocols):
    return generate_graph(protocols[1], 7, .9, .4, block_size=20)


def test_locked_matrix_and_independence(protocols):
    spec, base = protocols
    validate_specification(spec, base)
    units = matrix(spec)
    assert len(units) == 288 and len({u["graph_id"] for u in units}) == 96
    assert {u["seed"] for u in units} == set(range(28101, 28109))
    assert sha256(ROOT / spec["base_protocol"]) == spec["base_protocol_sha256"]
    assert base["training"]["rounds"] == 100
    assert spec["confirmation"]["primary_cell"] == {"h": .9, "rho": .4}


@pytest.mark.parametrize("field,value", [("seeds", [28001]), ("homophily", [.9]),
    ("models", ["observed_p"]), ("primary_cell", {"h": .9, "rho": .1}), ("graph_units", 95)])
def test_contract_mutations_rejected(protocols, field, value):
    spec, base = copy.deepcopy(protocols)
    spec["confirmation"][field] = value
    with pytest.raises(RuntimeError):
        validate_specification(spec, base)


def test_evaluation_projection_is_not_inplace(graph):
    projected = project_test_role(graph)
    assert projected is not graph and projected["masks"] is not graph["masks"]
    for key in ("features", "labels", "full_edge_index", "membership"):
        assert projected[key] is graph[key]
    assert projected["masks"]["validation"] is graph["masks"]["test_sealed"]
    assert int(graph["masks"]["validation"].sum()) == 24
    assert int(graph["masks"]["test_sealed"].sum()) == 48


def test_no_test_evaluation_without_commit(protocols, graph):
    with pytest.raises(RuntimeError, match="remains sealed"):
        evaluate_test(protocols[1], graph, "observed_p", {}, commit_verified=False)


def mock_commit(spec):
    return dict(all_training_complete=True, training_runs=288, graph_units=96,
        matrix=matrix(spec), protocol_sha256=sha256(PROTOCOL), artifact_sha256={},
        test_evaluated_before_commit=False)


@pytest.mark.parametrize("mutation", ["absent", "incomplete", "matrix", "protocol", "artifacts", "premature"])
def test_missing_commit_never_unlocks(tmp_path, protocols, mutation):
    spec, _ = protocols
    commit = mock_commit(spec)
    if mutation == "incomplete":
        commit["all_training_complete"] = False
    elif mutation == "matrix":
        commit["matrix"] = commit["matrix"][:-1]
    elif mutation == "protocol":
        commit["protocol_sha256"] = "bad"
    elif mutation == "premature":
        commit["test_evaluated_before_commit"] = True
    if mutation != "absent":
        (tmp_path / "checkpoint_commit.json").write_text(json.dumps(commit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="remains sealed"):
        require_test_unlock(tmp_path, spec)


def test_complete_commit_and_hash_tampering(tmp_path, protocols):
    spec, _ = protocols
    commit = mock_commit(spec)
    for unit in matrix(spec):
        folder = tmp_path / "training" / unit["graph_id"]
        for path in [folder / "sealed_graph_bundle.pt", folder / "graph_manifest.json"] + [
                folder / unit["model"] / name for name in (
                    "initial_state.pt", "selected_state.pt", "validation_trajectory.json", "training_record.json")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"engineering-only fake artifact, not a graph/checkpoint")
            commit["artifact_sha256"][str(path.relative_to(tmp_path))] = sha256(path)
    (tmp_path / "checkpoint_commit.json").write_text(json.dumps(commit), encoding="utf-8")
    assert len(require_test_unlock(tmp_path, spec)["artifact_sha256"]) == 1344
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        require_test_unlock(tmp_path, spec)


@pytest.mark.parametrize("name", ["observed_p", "identity", "strong_mlp"])
def test_original_test_only_state_immutable_and_raw_recomputation(protocols, graph, name):
    base = protocols[1]
    state, _ = train_model(base, graph, name, torch.device("cpu"), None, rounds=2)
    digest = tensor_digest(state)
    rows, evidence, audit = evaluate_test(base, graph, name, state, commit_verified=True)
    assert len(evidence["nodes"]) == 48
    assert set(evidence["nodes"].tolist()) == set(torch.where(graph["masks"]["test_sealed"])[0].tolist())
    assert tensor_digest(state) == digest
    assert not audit["reliability"]["certified_flip_nodes"]
    assert audit["reliability"]["bound_violations"] == 0
    audit_raw_rows(dict(rows=rows), evidence)
    changed = copy.deepcopy(graph)
    changed["labels"][graph["masks"]["validation"] | graph["masks"]["audit_unused"]] ^= 1
    rows2, evidence2, _ = evaluate_test(base, changed, name, state, commit_verified=True)
    assert rows2 == rows
    assert torch.equal(evidence["nominal_logits"], evidence2["nominal_logits"])
    if name != "observed_p":
        assert evidence["certificates"]["structural_invariance"].all()
        assert not evidence["flip"].any()
    broken = copy.deepcopy(rows)
    broken[0]["certified_n"] += 1
    with pytest.raises(RuntimeError, match="raw subgroup counts"):
        audit_raw_rows(dict(rows=broken), evidence)


def test_test_labels_do_not_train_model(protocols, graph):
    changed = copy.deepcopy(graph)
    changed["labels"][graph["masks"]["test_sealed"]] ^= 1
    states = [train_model(protocols[1], data, "observed_p", torch.device("cpu"), None, rounds=2)[0]
              for data in (graph, changed)]
    assert tensor_digest(states[0]) == tensor_digest(states[1])


@pytest.mark.parametrize("values,margin,successes,p", [([1]*8, 0, 8, 1/256),
    ([1]*7+[0], 0, 7, 9/256), ([0]*8, 0, 0, 1), ([-.009]*8, -.01, 8, 1/256)])
def test_exact_sign_consistency(values, margin, successes, p):
    got, count = exact_sign_p(values, margin)
    assert count == successes and got == pytest.approx(p)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_sign_values(values):
    with pytest.raises(ValueError):
        exact_sign_p(values)


def test_holm_stepdown_and_permutation():
    raw = [.01, .04, .03, .002]
    adjusted = [.03, .06, .06, .008]
    assert holm_adjust(raw) == pytest.approx(adjusted)
    order = [3, 2, 0, 1]
    assert holm_adjust([raw[i] for i in order]) == pytest.approx([adjusted[i] for i in order])
    assert holm_adjust([1/256]*4) == pytest.approx([4/256]*4)


def test_bootstrap_unit_indices_and_missing_values(protocols):
    a, b = resample_indices(protocols[0]), resample_indices(protocols[0])
    assert np.array_equal(a, b) and a.shape == (10000, 8)
    assert a.min() == 0 and a.max() == 7
    result = estimate([.03]*8, a)
    assert result["mean"] == pytest.approx(.03) and result["ci95"] == pytest.approx([.03, .03])
    assert estimate([None]*8, a)["undefined"]
    with pytest.raises(ValueError, match="partial missing"):
        estimate([None]+[.1]*7, a)
    with pytest.raises(ValueError):
        estimate([.1]*7, a)


def test_zero_cut_subgroups_are_undefined(protocols):
    data = generate_graph(protocols[1], 7, .9, 0, block_size=20)
    records = []
    for name in ("observed_p", "identity", "strong_mlp"):
        state, _ = train_model(protocols[1], data, name, torch.device("cpu"), None, rounds=2)
        rows, _, _ = evaluate_test(protocols[1], data, name, state, commit_verified=True)
        records.append(dict(rows=rows))
    metrics = paired_metrics(*records)
    assert metrics["test_exposure_fraction"] == 0
    assert metrics["hybrid_exposed_certified_ratio"] is None
    assert metrics["p_minus_i_exposed_macro_f1"] is None
    assert metrics["hybrid_minus_symmetric_exposed_ratio"] is None


def test_confirmation_requires_all_point_statistical_and_integrity_gates(protocols):
    spec, base = protocols
    metrics = dict(p_minus_i_all_macro_f1=.04, p_minus_i_exposed_macro_f1=.04,
        p_minus_mlp_all_macro_f1=.04, test_exposure_fraction=1,
        exposed_softmax_change_fraction=.08, exposed_category_flip_fraction=.03,
        hybrid_exposed_certified_ratio=.6, hybrid_minus_symmetric_exposed_ratio=.4)
    primary = dict(estimates={key: dict(mean=value) for key, value in metrics.items()})
    tests = [dict(**hypothesis, holm_p=.015625, ci95=[.02, .06])
             for hypothesis in spec["statistics"]["primary_family"]]
    assert primary_decision(primary, tests, base)[1]
    assert not primary_decision(primary, tests[:-1], base)[1]
    assert not primary_decision(primary, tests, base, reliability_ok=False)[1]
    assert not primary_decision(primary, tests, base, resources_ok=False)[1]
    bad = copy.deepcopy(tests)
    bad[0]["holm_p"] = .1
    assert not primary_decision(primary, bad, base)[1]
    bad[0]["holm_p"], bad[0]["ci95"] = .01, [0, .06]
    assert not primary_decision(primary, bad, base)[1]
    primary["estimates"]["p_minus_i_all_macro_f1"]["mean"] = 0
    assert not primary_decision(primary, tests, base)[1]
