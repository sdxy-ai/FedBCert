from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier, certificate_statistics,
    cone_pairwise_certificate_statistics, retained_mass_uncertainty_basis,
    row_normalized_shift,
)
from scripts.audit_fedbcert_e28_exposure import (
    check_output_root, exposure_masks, load_saved_membership,
    safe_div, subgroup_metrics, write_json,
)
from scripts.analyze_fedbcert_e28_exposure import coverage_decomposition

ROOT = Path(__file__).resolve().parents[1]


def chain_shift():
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return row_normalized_shift(edges, 4, device=torch.device("cpu"))[0]


@pytest.mark.parametrize("power,expected", [
    (0, [False, False, False, False]),
    (1, [True, False, False, False]),
    (2, [True, True, False, False]),
    (3, [True, True, True, False]),
])
def test_exposure_counts_boundary_row_change_as_one_step(power, expected):
    masks = exposure_masks(chain_shift(), torch.tensor([1., 0., 0., 0.]), power)
    assert masks["exposed"].tolist() == expected
    assert torch.equal(masks["direct"] | masks["indirect"], masks["exposed"])
    assert not bool((masks["direct"] & masks["indirect"]).any())
    assert torch.equal(masks["exposed"] | masks["non_exposed"], masks["all"])


def test_no_boundary_has_empty_exposure_and_zero_radius():
    shift = chain_shift()
    counts = torch.zeros(4)
    groups = exposure_masks(shift, counts, 2)
    basis = retained_mass_uncertainty_basis(shift, counts, 3)
    assert not bool(groups["exposed"].any())
    assert all(torch.equal(item, counts) for item in basis)


@pytest.mark.parametrize("counts", [
    torch.tensor([-1., 0., 0., 0.]),
    torch.tensor([0.5, 0., 0., 0.]),
    torch.tensor([float("nan"), 0., 0., 0.]),
])
def test_invalid_exact_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        exposure_masks(chain_shift(), counts, 2)


def test_ca_and_conditional_accuracy_have_different_denominators():
    logits = torch.tensor([[2., 0.], [2., 0.], [0., 2.], [0., 2.]])
    labels = torch.tensor([0, 1, 1, 0])
    certified = torch.tensor([True, True, False, False])
    metrics = subgroup_metrics(logits, labels, certified, torch.ones(4, dtype=torch.bool))
    assert metrics["certified_ratio"] == 0.5
    assert metrics["certified_accuracy"] == 0.25
    assert metrics["accuracy_given_certified"] == 0.5
    assert metrics["wrong_certified_count"] == 1


def test_empty_group_is_na_not_zero_or_one():
    metrics = subgroup_metrics(torch.ones(2, 2), torch.zeros(2, dtype=torch.long),
                               torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    assert metrics["node_count"] == 0
    assert metrics["certified_ratio"] is None
    assert metrics["certified_accuracy"] is None
    assert metrics["accuracy_given_certified"] is None
    assert safe_div(0, 0) is None
    json.dumps(metrics, allow_nan=False)


def test_certificate_calls_preserve_logits_and_model_and_union_contains_both():
    torch.manual_seed(28)
    model = PolynomialSpectralClassifier(3, 4, 2, 3, 0., "dual_feature_spectral", 2,
                                        spectral_fusion_scale=0.25,
                                        certificate_margin_mode="pairwise_difference").eval()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    features = torch.randn(4, 3)
    shift = chain_shift()
    budget = torch.tensor([0.4, 0., 0., 0.])
    basis = retained_mass_uncertainty_basis(shift, budget, 3)
    with torch.no_grad():
        logits = model(features, shift)
        symmetric = certificate_statistics(model, logits, basis)
        raw = cone_pairwise_certificate_statistics(model, features, shift, budget, logits, basis,
                                                   combine_symmetric=False)
        combined = cone_pairwise_certificate_statistics(model, features, shift, budget, logits, basis)
        assert torch.equal(logits, model(features, shift))
    assert torch.equal(combined.certified, symmetric.certified | raw.certified)
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    groups = exposure_masks(shift, torch.tensor([1., 0., 0., 0.]), 2)
    # Learned coefficients never enter the primary structural subgroup.
    model.theta.data.zero_()
    assert torch.equal(groups["exposed"], exposure_masks(shift, torch.tensor([1., 0., 0., 0.]), 2)["exposed"])


def test_membership_requires_each_node_once(tmp_path):
    path = tmp_path / "membership.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([["node_id", "client_id"], [0, 0], [1, 1], [2, 0]])
    assert load_saved_membership(path, 3, 2).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        load_saved_membership(path, 4, 2)


def test_existing_artifacts_are_never_overwritten(tmp_path):
    path = tmp_path / "audit.json"
    write_json(path, {"original": True})
    write_json(path, {"original": True})
    with pytest.raises(RuntimeError, match="overwrite"):
        write_json(path, {"original": False})
    assert json.loads(path.read_text())["original"] is True


def test_output_cannot_target_source_results():
    with pytest.raises(ValueError):
        check_output_root(ROOT / "results_boundary_certified/boundary_certified_spectral_e18_cone")


def test_protocol_is_retrospective_full_matrix_and_selection_free():
    config = json.loads((ROOT / "configs/fedbcert_e28a_exposure_protocol.json").read_text(encoding="utf-8"))
    assert config["status"] == "retrospective_scope_fixed_before_audit_execution"
    assert config["expected_units"] == 6 * 2 * 2 * 3 == 72
    assert config["maximum_propagation_power"] == 2
    for key in ("no_training", "no_checkpoint_or_certificate_selection", "no_partition_regeneration",
                "not_new_independent_confirmation", "no_reinterpretation_of_e19_e22_or_e24"):
        assert config["claims"][key] is True
    assert config["exposure"]["uses_learned_coefficients_or_predictions"] is False


def test_decomposition_uses_within_unit_masses_not_product_of_means():
    rows = []
    # Different exposures and rates make marginal-average multiplication wrong.
    for identity, exposed_count, exposed_symmetric, exposed_combined in (
        ("a", 2, 0, 2), ("b", 8, 4, 4),
    ):
        for group, count, sym, combined in (
            ("all", 10, exposed_symmetric + 10 - exposed_count, exposed_combined + 10 - exposed_count),
            ("exposed", exposed_count, exposed_symmetric, exposed_combined),
            ("non_exposed", 10 - exposed_count, 10 - exposed_count, 10 - exposed_count),
        ):
            for certificate, certified in (("symmetric", sym), ("combined", combined)):
                rows.append({"identity": identity, "dataset": "toy", "num_clients": 3,
                             "partition": "random", "seed": 1, "group": group,
                             "certificate": certificate, "node_count": count,
                             "certified_count": certified, "certified_ratio": certified / count})
    decomposed = coverage_decomposition(rows)
    assert [row["exposed_coverage_gain_mass"] for row in decomposed] == [0.2, 0.0]
    assert all(row["non_exposed_coverage_gain_mass"] == 0 for row in decomposed)
