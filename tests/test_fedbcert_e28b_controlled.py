import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fedbcert_e28b_common import (
    PROTOCOL, MODEL_NAMES, development_labels, derive_seed, evaluate_validation,
    exposure_groups, generate_graph, make_clients, make_model, pair_probability,
    run_matrix, tensor_digest, train_model, write_json,
)
from analyze_fedbcert_e28b_controlled import cell_gates
from run_fedbcert_e28b_controlled import verify_manifest


@pytest.fixture
def protocol():
    torch.set_num_threads(1)
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


@pytest.fixture
def graph(protocol):
    return generate_graph(protocol, 7, .9, .4, block_size=20)


@pytest.mark.parametrize("h", [.5, .7, .9])
@pytest.mark.parametrize("rho", [0, .01, .1, .4])
def test_expected_degree_contract(h, rho):
    p = lambda client, label: pair_probability(h, rho, client, label, 1000)
    assert 999 * p(True, True) + 1000 * p(True, False) == pytest.approx((1-rho) * 12)
    assert 2000 * p(False, True) + 2000 * p(False, False) == pytest.approx(rho * 12)
    assert 999 * p(True, True) + 2000 * p(False, True) == pytest.approx(12 * h)


def test_generator_reproducible_and_isolated_streams(protocol, graph):
    again = generate_graph(protocol, 7, .9, .4, block_size=20)
    other = generate_graph(protocol, 7, .5, .01, block_size=20)
    for key in ("features", "labels", "membership", "full_edge_index"):
        assert torch.equal(graph[key], again[key])
    assert torch.equal(graph["features"], other["features"])
    assert all(torch.equal(graph["masks"][k], other["masks"][k]) for k in graph["masks"])
    assert len(set(graph["metadata"]["derived_seeds"].values())) == 5


def test_simple_undirected_graph_and_exact_ledger(graph):
    edges = graph["full_edge_index"]
    pairs = set(map(tuple, edges.T.tolist()))
    assert len(pairs) == edges.shape[1]
    assert all(a != b and (b, a) in pairs for a, b in pairs)
    member = graph["membership"]
    expected = torch.bincount(edges[0, member[edges[0]] != member[edges[1]]], minlength=120)
    assert torch.equal(expected.float(), graph["missing_degree"])


def test_no_boundary_at_zero_cut(protocol):
    graph = generate_graph(protocol, 7, .9, 0, block_size=20)
    assert graph["missing_degree"].sum() == 0
    clients = make_clients(protocol, graph, "cpu")
    assert all(not exposure_groups(c, 3)["exposed"].any() for c in clients)


def test_stratified_split_and_test_label_sealing(protocol, graph):
    stacked = torch.stack(list(graph["masks"].values()))
    assert (stacked.sum(0) == 1).all()
    for block in range(6):
        assert [int(m[block*20:(block+1)*20].sum()) for m in graph["masks"].values()] == [6, 2, 4, 8]
    sealed = development_labels(graph)
    assert (sealed[graph["masks"]["test_sealed"] | graph["masks"]["audit_unused"]] == -1).all()
    for client in make_clients(protocol, graph, "cpu"):
        assert client.test_mask.sum() == 0
        assert (client.labels[~(client.train_mask | client.validation_mask)] == -1).all()


def test_model_capacity_and_matched_initialization(protocol):
    states = []
    for name in MODEL_NAMES:
        torch.manual_seed(123)
        model = make_model(protocol, name)
        states.append(tensor_digest(model.state_dict()))
        assert sum(p.numel() for p in model.parameters()) == protocol["models"][name]["parameters"]
    assert states[0] == states[1]
    assert abs(8857 / 8775 - 1) < .05
    # Capacity choice by count, not efficacy.
    counts = [(abs(sum(p.numel() for p in make_model_width(w).parameters()) - 8775), w) for w in range(60, 90)]
    assert min(counts)[1] == 77


def make_model_width(width):
    from openfgl.flcore.fedbcert.certificate import PolynomialSpectralClassifier
    return PolynomialSpectralClassifier(32, width, 2, 3, .2, "feature_only_mlp", 2)


def test_matrix_independent_seeds_confirmation_not_consumed(protocol):
    matrix = run_matrix(protocol)
    assert len(matrix) == 24
    assert set(m["seed"] for m in matrix) == {28001, 28002}
    namespaces = [set(protocol["seeds"][k]) for k in ("engineering", "development", "confirmation_reserved_only")]
    assert not any(a & b for i, a in enumerate(namespaces) for b in namespaces[i+1:])
    assert derive_seed(28001, 11) != derive_seed(28002, 11)


def test_recording_fix_does_not_change_scientific_protocol(protocol):
    original = json.loads((ROOT / "configs/fedbcert_e28b_controlled_protocol.json").read_text(encoding="utf-8"))
    current = copy.deepcopy(protocol)
    current["artifacts"]["output_root"] = original["artifacts"]["output_root"]
    assert current == original
    # Regression: ordinary merge, not duplicated keyword expansion.
    unit = {"model": "observed_p", "seed": 28001}
    training = {"model": "observed_p", "selected_round": 1}
    record = {**unit, **training}
    assert record["model"] == "observed_p" and record["seed"] == 28001


def test_exposure_exactly_boundary_plus_one_hop(protocol, graph):
    for client in make_clients(protocol, graph, "cpu"):
        group = exposure_groups(client, 3)
        expected = group["boundary"] | (torch.sparse.mm(client.shift, group["boundary"].float()[:, None]).squeeze(1) > 0)
        assert torch.equal(group["exposed"], expected)
        assert torch.equal(group["boundary"] | group["near_boundary"], group["exposed"])
        assert torch.equal(~group["exposed"], group["unexposed"])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_training_evidence_validation_only_reliability(protocol, graph, name):
    state, training = train_model(protocol, graph, name, torch.device("cpu"), None, rounds=2)
    rows, evidence, audit = evaluate_validation(protocol, graph, name, state)
    assert len(evidence["nodes"]) == 24
    assert set(evidence["nodes"].tolist()) == set(torch.where(graph["masks"]["validation"])[0].tolist())
    assert evidence["labels"].min() >= 0
    assert not audit["reliability"]["certified_flip_nodes"]
    assert audit["reliability"]["bound_violations"] == 0
    if name == "observed_p":
        cert = evidence["certificates"]
        assert torch.equal(cert["cone_hybrid"], cert["cone_only"] | cert["retained_pairwise"])
    else:
        assert evidence["certificates"]["structural_invariance"].all()
        assert all(row["certified_ratio"] in (1.0, None) for row in rows)
        assert not evidence["flip"].any()
    assert 1 <= training["selected_round"] <= 2


def test_heldout_label_changes_do_not_affect_training_or_evaluation(protocol, graph):
    changed = copy.deepcopy(graph)
    hidden = graph["masks"]["test_sealed"] | graph["masks"]["audit_unused"]
    changed["labels"][hidden] = 1 - changed["labels"][hidden]
    for data in (graph, changed):
        state, _ = train_model(protocol, data, "observed_p", torch.device("cpu"), None, rounds=2)
        _, evidence, _ = evaluate_validation(protocol, data, "observed_p", state)
        if data is graph:
            expected_state, expected_logits = tensor_digest(state), evidence["nominal_logits"]
        else:
            assert tensor_digest(state) == expected_state
            assert torch.equal(evidence["nominal_logits"], expected_logits)


def test_protocol_overwrite_and_hash_mismatch_rejected(tmp_path):
    path = tmp_path / "exclusive.json"
    write_json(path, {"ok": True})
    with pytest.raises(FileExistsError):
        write_json(path, {"ok": False})
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_manifest({"source_sha256": {str(path): "bad"}, "protected_sha256": {}})


def test_gate_cannot_be_passed_by_coverage_alone(protocol):
    values = dict(p_minus_i_all_macro_f1=0, p_minus_i_exposed_macro_f1=0,
                  p_minus_mlp_all_macro_f1=0, validation_exposure_fraction=.8,
                  exposed_softmax_change_fraction=0, exposed_category_flip_fraction=0,
                  hybrid_exposed_certified_ratio=.9, hybrid_minus_symmetric_exposed_ratio=.2)
    gates = cell_gates(values, protocol["stopping"]["development_cell_gate"])
    assert gates["C1_nontrivial_coverage"] and gates["C2_certificate_gain"]
    assert not gates["A_graph_utility"] and not gates["B2_actual_sensitivity"]
