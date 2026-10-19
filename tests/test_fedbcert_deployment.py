from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from openfgl.flcore.fedbcert.certificate import (
    PolynomialSpectralClassifier, boundary_uncertainty, certificate_statistics,
    cone_pairwise_certificate_statistics, retained_mass_uncertainty_basis,
    row_normalized_shift,
)
from openfgl.flcore.fedbcert.deployment import BoundaryLedgerReceipt, FrozenLocalDeployment
from scripts.demo_fedbcert_deployment import build_demo, illustrative_inputs, write_new_report


def make_snapshot():
    model, features, edges = illustrative_inputs()
    snapshot = FrozenLocalDeployment(
        model, features, edges, view_version="view-v1", preprocessing_version="preprocessing-v1"
    )
    receipt = BoundaryLedgerReceipt(
        bounds=torch.tensor([0, 1, 3]), ledger_version="ledger-v1", source="external-attestation",
        view_sha256=snapshot.view_sha256, preprocessing_version="preprocessing-v1",
        upper_bound_attested=True,
    )
    return snapshot, receipt, model, features, edges


def test_demo_shows_three_states_and_explicit_universal_without_changing_output():
    report = build_demo()
    assert report["state_counts"]["valid_upper_bound_receipt"] == {
        "Certified": 2, "Deferred": 1, "Unavailable": 0,
    }
    assert report["state_counts"]["missing_ledger"]["Unavailable"] == 3
    assert report["state_counts"]["explicit_universal_without_ledger"]["Unavailable"] == 0
    outputs = [item["nominal_logits"] for item in report["cases"].values()]
    assert all(item == outputs[0] for item in outputs)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change,reason", [
    ({"upper_bound_attested": False}, "upper_bound_not_externally_attested"),
    ({"upper_bound_attested": "yes"}, "upper_bound_not_externally_attested"),
    ({"ledger_version": ""}, "missing_ledger_version_or_source"),
    ({"source": " "}, "missing_ledger_version_or_source"),
    ({"view_sha256": "stale"}, "ledger_view_mismatch"),
    ({"preprocessing_version": "stale"}, "ledger_preprocessing_mismatch"),
    ({"bounds": torch.tensor([-1, 0, 0])}, "ledger_bounds_must_be_node_aligned_nonnegative_integers"),
    ({"bounds": torch.tensor([0., 0.5, 0.])}, "ledger_bounds_must_be_node_aligned_nonnegative_integers"),
    ({"bounds": torch.tensor([0., float("nan"), 0.])}, "ledger_bounds_must_be_node_aligned_nonnegative_integers"),
    ({"bounds": torch.tensor([False, True, True])}, "ledger_bounds_must_be_node_aligned_nonnegative_integers"),
    ({"bounds": torch.tensor([0, 1])}, "ledger_bounds_must_be_node_aligned_nonnegative_integers"),
    ({"bounds": torch.tensor([[0, 1, 3]])}, "ledger_bounds_must_be_node_aligned_nonnegative_integers"),
])
def test_invalid_receipt_is_unavailable_not_wrong_prediction(change, reason):
    snapshot, receipt, *_ = make_snapshot()
    result = snapshot.certify(replace(receipt, **change))
    assert result["nominal_logits"] == snapshot.nominal_logits.tolist()
    assert all(row["state"] == "Unavailable" and row["reason"] == reason for row in result["records"])
    assert all(row["minimum_gap_lower_bound"] is None for row in result["records"])
    json.dumps(result, allow_nan=False)


def test_no_silent_fallback_and_no_ignored_ledger():
    snapshot, receipt, *_ = make_snapshot()
    missing = snapshot.certify()
    assert missing["mode"] == "bounded_ledger"
    assert missing["records"][0]["reason"] == "missing_ledger_no_automatic_fallback"
    with pytest.raises(ValueError, match="takes no ledger"):
        snapshot.certify(receipt, mode="universal")
    with pytest.raises(ValueError, match="explicitly"):
        snapshot.certify(mode="automatic")


def test_adapter_matches_existing_two_certificates_on_the_same_model():
    snapshot, receipt, model, features, edges = make_snapshot()
    shift, degree = row_normalized_shift(edges, 3, device=torch.device("cpu"), dtype=features.dtype)
    budget = boundary_uncertainty(degree, receipt.bounds.to(features.dtype))
    basis = retained_mass_uncertainty_basis(shift, budget, 3)
    with torch.no_grad():
        logits = model(features, shift)
        sym = certificate_statistics(model, logits, basis)
        cone = cone_pairwise_certificate_statistics(
            model, features, shift, budget, logits, basis, combine_symmetric=False
        )
    result = snapshot.certify(receipt)
    assert torch.equal(logits, snapshot.nominal_logits)
    for index, row in enumerate(result["records"]):
        assert row["symmetric_gap_lower_bound"] == sym.worst_slack[index].item()
        assert row["cone_gap_lower_bound"] == cone.worst_slack[index].item()
        gap = max(sym.worst_slack[index].item(), cone.worst_slack[index].item())
        assert row["minimum_gap_lower_bound"] == gap
        assert (row["state"] == "Certified") == (gap > result["margin_guard"])


def test_nonfinite_certificate_arithmetic_is_unavailable_with_finite_nominal_output():
    model, features, edges = illustrative_inputs()
    with torch.no_grad():
        model.classifier.weight.mul_(1e200)
    snapshot = FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    receipt = BoundaryLedgerReceipt(
        bounds=torch.tensor([0, 1, 3]), ledger_version="l", source="fixture",
        view_sha256=snapshot.view_sha256, preprocessing_version="p", upper_bound_attested=True
    )
    result = snapshot.certify(receipt)
    assert bool(torch.isfinite(snapshot.nominal_logits).all())
    assert all(row["state"] == "Unavailable" and row["reason"] == "nonfinite_certificate_arithmetic"
               for row in result["records"])
    json.dumps(result, allow_nan=False)


def test_snapshot_owns_copies_and_never_mutates_caller_state_or_logits():
    snapshot, receipt, model, features, edges = make_snapshot()
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    nominal = snapshot.nominal_logits
    before = snapshot.certify(receipt)
    snapshot.certify(mode="universal")
    assert all(torch.equal(value, original_state[key]) for key, value in model.state_dict().items())
    assert not model.training
    # Later updates of the caller's objects cannot alter the old snapshot.
    features.zero_()
    edges.zero_()
    with torch.no_grad():
        model.classifier.weight.zero_()
    model.train()
    nominal.zero_()  # The accessor returns a copy too.
    assert snapshot.certify(receipt) == before


@pytest.mark.parametrize("target", ["model", "features", "shift", "degree", "logits", "version"])
def test_internal_snapshot_tampering_refuses_old_certificate(target):
    snapshot, receipt, *_ = make_snapshot()
    with torch.no_grad():
        if target == "model":
            snapshot._model.theta[1] += 0.01
        elif target == "features":
            snapshot._features[0, 0] += 1.0
        elif target == "shift":
            snapshot._shift.values()[0] += 0.01
        elif target == "degree":
            snapshot._degree[0] += 1
        elif target == "logits":
            snapshot._logits[0, 0] += 1
        else:
            snapshot.view_version = "view-v2"
    with pytest.raises(RuntimeError, match="snapshot changed"):
        snapshot.certify(receipt)


def test_refining_safe_upper_bounds_cannot_remove_guarded_certificates():
    snapshot, receipt, *_ = make_snapshot()
    coarse = snapshot.certify(replace(receipt, bounds=torch.tensor([3, 3, 6])))
    fine = snapshot.certify(receipt)
    assert all(a["state"] != "Certified" or b["state"] == "Certified"
               for a, b in zip(coarse["records"], fine["records"]))


def test_attestation_is_an_external_assumption_not_a_local_truth_detector():
    snapshot, receipt, *_ = make_snapshot()
    result = snapshot.certify(replace(receipt, bounds=torch.zeros(3, dtype=torch.long)))
    # Local inputs cannot tell whether a purported zero count is actually true.
    assert all(row["state"] == "Certified" for row in result["records"])
    assert "ledger_truth_or_completeness" in result["not_guaranteed"]


@pytest.mark.parametrize("guard", [0., -1., float("nan"), float("inf")])
def test_invalid_numerical_guards_rejected(guard):
    snapshot, receipt, *_ = make_snapshot()
    with pytest.raises(ValueError, match="margin_guard"):
        snapshot.certify(receipt, margin_guard=guard)


def test_guard_is_strict_and_ties_are_deferred_conservatively():
    snapshot, receipt, *_ = make_snapshot()
    result = snapshot.certify(receipt)
    gap = result["records"][0]["minimum_gap_lower_bound"]
    assert snapshot.certify(receipt, margin_guard=gap)["records"][0]["state"] == "Deferred"
    model, features, edges = illustrative_inputs()
    with torch.no_grad():
        model.classifier.weight.zero_()
    tied = FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    assert all(row["state"] == "Deferred" for row in tied.certify(mode="universal")["records"])


@pytest.mark.parametrize("encoder,depth", [
    ("bounded_residual_mlp", 2), ("feature_only_mlp", 2),
    ("shielded_sage_spectral", 2), ("shielded_diffusion_spectral", 2),
])
def test_unsupported_predictor_branches_are_not_certified_by_this_adapter(encoder, depth):
    model = PolynomialSpectralClassifier(2, 2, 2, 3, 0., encoder, depth).double().eval()
    _, features, edges = illustrative_inputs()
    with pytest.raises(ValueError, match="unsupported predictor"):
        FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")


@pytest.mark.parametrize("problem", ["training", "nonfinite", "scale", "override", "hook"])
def test_noncontract_model_refused_without_changing_it(problem):
    model, features, edges = illustrative_inputs()
    if problem == "training":
        model.train()
    elif problem == "nonfinite":
        with torch.no_grad():
            model.theta[0] = float("nan")
    elif problem == "scale":
        model.spectral_fusion_scale = 0.25
    elif problem == "override":
        model.bounded_encoding = lambda inputs: -inputs
    else:
        model.encoder.register_forward_hook(lambda *_: None)
    with pytest.raises(ValueError):
        FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    assert model.training == (problem == "training")


@pytest.mark.parametrize("problem", ["nan_features", "dtype", "foreign_node", "float_edges", "duplicate_ids"])
def test_invalid_local_inference_inputs_refused(problem):
    model, features, edges = illustrative_inputs()
    ids = None
    if problem == "nan_features":
        features[0, 0] = float("nan")
    elif problem == "dtype":
        features = features.float()
    elif problem == "foreign_node":
        edges = torch.tensor([[0], [3]])
    elif problem == "float_edges":
        edges = edges.float()
    else:
        ids = ["same"] * 3
    with pytest.raises(ValueError):
        FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p", node_ids=ids)


def test_dual_feature_relu_predictor_supported_but_not_a_teacher_claim():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = PolynomialSpectralClassifier(
            2, 3, 2, 3, 0., "dual_feature_spectral", 2,
            spectral_fusion_scale=0.25, certificate_margin_mode="pairwise_difference"
        ).double().eval()
    _, features, edges = illustrative_inputs()
    snapshot = FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    result = snapshot.certify(mode="universal")
    assert result["certificate_object"] == "same_frozen_polynomial_predictor_class_invariance"
    assert "arbitrary_GNN_or_teacher_invariance" in result["not_guaranteed"]


def test_nonparameter_model_configuration_is_bound_to_the_snapshot_too():
    model = PolynomialSpectralClassifier(
        2, 3, 2, 3, 0., "dual_feature_spectral", 2,
        certificate_margin_mode="pairwise_difference"
    ).double().eval()
    _, features, edges = illustrative_inputs()
    snapshot = FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    snapshot._model.feature_norms[0].eps *= 10
    with pytest.raises(RuntimeError, match="snapshot changed"):
        snapshot.certify(mode="universal")


def test_canonical_unit_graph_and_node_order_have_explicit_fingerprints():
    model, features, edges = illustrative_inputs()
    a = FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    redundant = torch.tensor([[1, 0, 0, 0, 2], [0, 1, 1, 0, 2]])
    b = FrozenLocalDeployment(model, features, redundant, view_version="v", preprocessing_version="p")
    reordered = FrozenLocalDeployment(
        model, features, edges, view_version="v", preprocessing_version="p", node_ids=("1", "0", "2")
    )
    assert a.view_sha256 == b.view_sha256
    assert torch.equal(a.nominal_logits, b.nominal_logits)
    assert a.view_sha256 != reordered.view_sha256


def test_float32_uses_original_dtype_and_also_keeps_nominal_logits():
    model, features, edges = illustrative_inputs()
    model = model.float()
    features = features.float()
    shift, _ = row_normalized_shift(edges, 3, device=torch.device("cpu"))
    with torch.no_grad():
        original = model(features, shift)
    snapshot = FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    result = snapshot.certify(mode="universal")
    assert torch.equal(original, snapshot.nominal_logits)
    assert result["snapshot"]["dtype"] == "torch.float32"
    assert result["numerical_assurance"].endswith("not_outward_rounded_proof")


@pytest.mark.parametrize("signed_coefficients", [False, True])
def test_certified_nodes_keep_their_class_on_sampled_legal_undirected_completions(signed_coefficients):
    model, features, edges = illustrative_inputs()
    if signed_coefficients:
        with torch.no_grad():
            model.theta.copy_(torch.tensor([0.7, 0.4, -0.1], dtype=torch.float64))
    snapshot = FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")
    receipt = BoundaryLedgerReceipt(
        bounds=torch.tensor([0, 1, 3]), ledger_version="l", source="offline-fixture",
        view_sha256=snapshot.view_sha256, preprocessing_version="p", upper_bound_attested=True
    )
    result = snapshot.certify(receipt)
    accepted = torch.tensor([row["state"] == "Certified" for row in result["records"]])
    assert bool(accepted.any())
    generator = torch.Generator().manual_seed(918)
    # Completions are used only by this OFFLINE test, never by the adapter.
    for b1 in (0, 1):
        for b2 in range(4):
            crossing = [(1, 3)] * b1 + [(2, 3 + offset) for offset in range(b2)]
            for remote_internal in ([], [(3, 4), (4, 5)]):
                pairs = [(0, 1)] + crossing + remote_internal
                complete_edges = torch.tensor(pairs, dtype=torch.long).T
                shift, _ = row_normalized_shift(complete_edges, 6, device=torch.device("cpu"), dtype=features.dtype)
                for _ in range(3):
                    remote_features = torch.randn(3, 2, generator=generator, dtype=features.dtype)
                    complete_features = torch.cat([features, remote_features])
                    with torch.no_grad():
                        completed_class = model(complete_features, shift)[:3].argmax(1)
                    assert torch.equal(completed_class[accepted], snapshot.nominal_logits.argmax(1)[accepted])


def test_new_report_refuses_overwrite_even_when_contents_identical(tmp_path):
    path = tmp_path / "new" / "demo.json"
    report = build_demo()
    write_new_report(path, report)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_report(path, report)
    assert path.read_bytes() == original
