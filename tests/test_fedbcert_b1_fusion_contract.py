"""Tiny B1 regressions: reject metadata mismatch without changing forward."""
from __future__ import annotations

import pytest
import torch

from openfgl.flcore.fedbcert.certificate import (
    PolynomialSpectralClassifier, certificate_statistics,
    cone_pairwise_certificate_statistics, retained_mass_uncertainty_basis,
)


ENTRIES = ("radius", "symmetric", "cone", "combined")
UNSCALED = ("linear_relu", "bounded_residual_mlp", "shielded_sage_spectral", "shielded_diffusion_spectral")
DUAL = ("dual_feature_spectral", "dual_feature_identity_null_spectral")


def _model(kind, scale):
    return PolynomialSpectralClassifier(
        2, 3, 3, 3, 0.0, encoder_type=kind,
        encoder_depth=1 if kind == "linear_relu" else 2,
        spectral_fusion_scale=scale, certificate_margin_mode="pairwise_difference",
    ).double().eval()


def _inputs():
    return (torch.eye(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64).to_sparse_coo().coalesce(),
            torch.tensor([1.0, 0.4], dtype=torch.float64))


@torch.no_grad()
def _invoke(entry, model, x, p, budget):
    basis = retained_mass_uncertainty_basis(p, budget, model.order_count)
    if entry == "radius":
        return model.boundary_radius(basis)
    logits = model.spectral_logits(x, p) if model.has_agreement_shield else model(x, p)
    if entry == "symmetric":
        return certificate_statistics(model, logits, basis)
    return cone_pairwise_certificate_statistics(
        model, x, p, budget, logits, basis,
        count_semantics="upper_bound", combine_symmetric=(entry == "combined"),
    )


@pytest.mark.parametrize("entry", ENTRIES)
def test_counterexample_keeps_forward_class_flip_but_rejects_certificate(entry):
    model = PolynomialSpectralClassifier(
        2, 2, 2, 2, 0.0, spectral_fusion_scale=0.0,
        certificate_margin_mode="pairwise_difference",
    ).double().eval()
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.encoder.bias.zero_()
        model.theta.copy_(torch.tensor([0.0, 1.0], dtype=torch.float64))
        model.classifier.weight.copy_(torch.tensor([[1., -2.], [0., 0.]], dtype=torch.float64))
        model.classifier.bias.zero_()
        x = torch.eye(2, dtype=torch.float64)
        local = torch.ones(1, 1, dtype=torch.float64).to_sparse_coo()
        full = torch.full((2, 2), 0.5, dtype=torch.float64).to_sparse_coo()
        before = model(x[:1], local)
        completed = model(x, full)
        assert torch.equal(before, torch.tensor([[1., 0.]], dtype=torch.float64))
        assert torch.equal(completed[:1], torch.tensor([[-0.5, 0.]], dtype=torch.float64))
        assert before.argmax(1).item() == 0 and completed[0].argmax().item() == 1
        with pytest.raises(ValueError, match="requires fusion scale=1"):
            _invoke(entry, model, x[:1], local, torch.ones(1, dtype=torch.float64))
        assert torch.equal(model(x[:1], local), before)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("kind", UNSCALED)
@pytest.mark.parametrize("scale", [0.0, 0.25, 2.0])
def test_unscaled_spectral_targets_reject_nonunit_scale(entry, kind, scale):
    with pytest.raises(ValueError, match="requires fusion scale=1"):
        _invoke(entry, _model(kind, scale), *_inputs())


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -float("inf"), -0.5])
def test_mutated_nonfinite_or_negative_scale_is_rejected(entry, scale):
    model = _model("dual_feature_spectral", .25)
    model.spectral_fusion_scale = scale
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _invoke(entry, model, *_inputs())


@pytest.mark.parametrize("entry", ENTRIES)
def test_feature_only_nonzero_metadata_is_rejected(entry):
    with pytest.raises(ValueError, match="requires fusion scale=0"):
        _invoke(entry, _model("feature_only_mlp", .25), *_inputs())


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("kind", DUAL)
def test_declared_dual_requires_actual_feature_classifier(entry, kind):
    model = _model(kind, .25)
    model.feature_classifier = None
    with pytest.raises(ValueError, match="requires a feature classifier"):
        _invoke(entry, model, *_inputs())


@pytest.mark.parametrize("kind", (*DUAL, "feature_only_mlp"))
def test_real_graph_invariant_zero_scale_preserves_tie_certificates(kind):
    model = _model(kind, 0.)
    head = model.classifier if kind == "feature_only_mlp" else model.feature_classifier
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
        assert torch.equal(_invoke("radius", model, *_inputs()), torch.zeros(2, dtype=torch.float64))
        for entry in ENTRIES[1:]:
            stats = _invoke(entry, model, *_inputs())
            assert stats.certified.all() and torch.equal(stats.worst_slack, torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("kind", DUAL)
@pytest.mark.parametrize("scale", [.25, 1.])
@torch.no_grad()
def test_valid_dual_signed_polynomial_forward_and_symmetric_values_unchanged(kind, scale):
    torch.manual_seed(1301)
    model = _model(kind, scale)
    model.theta.copy_(torch.tensor([.2, -.7, .3], dtype=torch.float64))
    x, p, budget = _inputs()
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    logits = model(x, p)
    assert torch.equal(logits, model.feature_classifier(model.feature_encoding(x)) + scale * model.spectral_logits(x, p))
    basis = retained_mass_uncertainty_basis(p, budget, model.order_count)
    radius = scale * sum(model.theta[k].abs() * basis[k] for k in range(1, model.order_count))
    sym = certificate_statistics(model, logits, basis)
    assert torch.equal(sym.representation_radius, radius)
    predicted = logits.argmax(1)
    directions = model.classifier.weight[predicted, None, :] - model.classifier.weight[None, :, :]
    expected = logits.gather(1, predicted[:, None]) - logits - radius[:, None] * directions.norm(dim=2)
    expected.scatter_(1, predicted[:, None], float("inf"))
    assert torch.equal(sym.worst_slack, expected.min(1).values)
    cone = _invoke("cone", model, x, p, budget)
    combined = _invoke("combined", model, x, p, budget)
    assert torch.equal(combined.worst_slack, torch.maximum(cone.worst_slack, sym.worst_slack))
    assert torch.equal(model(x, p), logits)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())
