from __future__ import annotations

import math
from types import MethodType

import pytest
import torch

from openfgl.flcore.fedbcert.certificate import (
    PolynomialSpectralClassifier,
    certificate_statistics,
    retained_mass_uncertainty_basis,
)
from openfgl.flcore.fedbcert.geometry_baselines import (
    evaluate_geometry_baselines,
    positive_orthant_direction_span,
)


def _model(kind: str = "dual_feature_spectral", alpha: float = 0.25, order: int = 4):
    return PolynomialSpectralClassifier(
        input_dim=3, hidden_dim=5, num_classes=3, order_count=order,
        dropout=0.2, encoder_type=kind,
        encoder_depth=1 if kind == "linear_relu" else 2,
        spectral_fusion_scale=1.0 if kind == "linear_relu" else alpha,
        certificate_margin_mode="pairwise_difference",
    ).double().eval()


def _inputs(seed: int, count: int = 6):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(count, 3, generator=generator, dtype=torch.float64)
    shift = torch.rand(count, count, generator=generator, dtype=torch.float64)
    shift = shift / shift.sum(dim=1, keepdim=True)
    budget = 2.0 * torch.rand(count, generator=generator, dtype=torch.float64)
    return features, shift.to_sparse_coo().coalesce(), budget


@pytest.mark.parametrize("kind", ["linear_relu", "dual_feature_spectral"])
def test_signed_coefficients_diameter_closed_form_and_geometry_order(kind: str):
    for seed in range(10):
        torch.manual_seed(seed)
        model = _model(kind)
        with torch.no_grad():
            model.theta.copy_(torch.tensor([0.7, -0.8, 0.3, -0.2], dtype=torch.float64))
        features, shift, budget = _inputs(seed)
        basis = retained_mass_uncertainty_basis(shift, budget, model.order_count)
        with torch.no_grad():
            logits = model(features, shift)
            historical = certificate_statistics(model, logits, basis)
        result = evaluate_geometry_baselines(
            model, features, shift, budget, logits=logits, basis=basis, guard=1e-8,
        )
        old, diameter, cone = (result[key] for key in (
            "ball_pairwise", "nonnegative_diameter", "cone",
        ))
        assert torch.equal(old.radius, historical.representation_radius)
        assert torch.equal(old.worst_slack, historical.worst_slack)
        assert torch.equal(diameter.radius, old.radius / math.sqrt(2.0))
        predicted = logits.argmax(dim=1)
        differences = model.classifier.weight[predicted, None] - model.classifier.weight[None]
        expected = logits.gather(1, predicted[:, None]) - logits
        expected = expected - diameter.radius[:, None] * differences.norm(dim=2)
        expected.scatter_(1, predicted[:, None], float("inf"))
        assert torch.equal(diameter.worst_slack, expected.min(dim=1).values)
        assert torch.all(diameter.worst_slack >= old.worst_slack - 1e-12)
        assert torch.all(cone.worst_slack >= diameter.worst_slack - 1e-12)
        assert torch.all(~old.certified | diameter.certified)
        assert torch.all(~diameter.certified | cone.certified)
        for item in result.values():
            assert torch.isfinite(item.worst_slack).all()
            assert torch.isfinite(item.radius).all()
            assert not item.worst_slack.requires_grad


def test_identity_shift_signed_polynomial_cone_closed_form():
    torch.manual_seed(103)
    model = _model(order=4)
    with torch.no_grad():
        model.theta.copy_(torch.tensor([-0.2, 0.7, -0.6, 0.4], dtype=torch.float64))
    features, _, budget = _inputs(103)
    shift = torch.eye(features.size(0), dtype=torch.float64).to_sparse_coo().coalesce()
    result = evaluate_geometry_baselines(model, features, shift, budget)
    with torch.no_grad():
        hidden = model.bounded_encoding(features)
        logits = model(features, shift)
        spectral = model.spectral_logits(features, shift)
        m = 1.0 - budget / 2.0
        radius = sum(
            model.theta[k].abs() * 2.0 * (1.0 - m.pow(k))
            for k in range(1, model.order_count)
        ) * model.spectral_fusion_scale
        expected_slack = []
        for v, a in enumerate(logits.argmax(dim=1).tolist()):
            pair = []
            for c in range(logits.size(1)):
                if c == a:
                    continue
                q = model.classifier.weight[a] - model.classifier.weight[c]
                lower = -q.clamp_max(0.0).norm()
                upper = q.clamp_min(0.0).norm()
                initial = hidden[v] @ q
                bound = model.theta[0] * initial
                for k in range(1, model.order_count):
                    support = lower if model.theta[k] >= 0.0 else upper
                    bound += model.theta[k] * (m[v].pow(k) * initial + (1.0 - m[v].pow(k)) * support)
                bound += model.classifier.bias[a] - model.classifier.bias[c]
                invariant = logits[v, a] - logits[v, c]
                invariant -= model.spectral_fusion_scale * (spectral[v, a] - spectral[v, c])
                pair.append(invariant + model.spectral_fusion_scale * bound)
            expected_slack.append(torch.stack(pair).min())
    assert torch.allclose(result["ball_pairwise"].radius, radius, atol=1e-12, rtol=1e-12)
    assert torch.allclose(result["cone"].worst_slack, torch.stack(expected_slack), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.25])
def test_zero_boundary_and_zero_alpha(alpha: float):
    model = _model(alpha=alpha)
    features, shift, budget = _inputs(4)
    if alpha != 0.0:
        budget.zero_()
    else:
        with torch.no_grad():
            model.feature_classifier.weight.zero_()
            model.feature_classifier.bias.zero_()
    result = evaluate_geometry_baselines(model, features, shift, budget)
    with torch.no_grad():
        logits = model(features, shift)
        predicted = logits.argmax(dim=1)
        nominal = logits.gather(1, predicted[:, None]) - logits
        nominal.scatter_(1, predicted[:, None], float("inf"))
        nominal = nominal.min(dim=1).values
    for item in result.values():
        assert torch.equal(item.radius, torch.zeros_like(budget))
        assert torch.allclose(item.worst_slack, nominal, atol=1e-12, rtol=0.0)
        if alpha == 0.0:
            assert item.certified.all()


def test_guard_changes_only_acceptance_and_not_historical_model_state():
    model = _model()
    features, shift, budget = _inputs(8)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    ordinary = evaluate_geometry_baselines(model, features, shift, budget, guard=0.0)
    guarded = evaluate_geometry_baselines(model, features, shift, budget, guard=1000.0)
    for key in ordinary:
        assert torch.equal(ordinary[key].worst_slack, guarded[key].worst_slack)
        assert not guarded[key].certified.any()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert not model.training


def test_negative_hidden_domain_and_unsupported_architecture_are_rejected():
    model = _model()
    features, shift, budget = _inputs(7)
    model.bounded_encoding = MethodType(
        lambda self, value: -torch.ones(value.size(0), 5, dtype=value.dtype) / 3.0,
        model,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_geometry_baselines(model, features, shift, budget)
    model = _model()
    model.encoder_type = "bounded_residual_mlp"
    with pytest.raises(ValueError, match="unsupported encoder"):
        evaluate_geometry_baselines(model, features, shift, budget)


def test_eval_finiteness_stochasticity_guard_and_cached_snapshot_checks():
    model = _model()
    features, shift, budget = _inputs(11)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        evaluate_geometry_baselines(model, features, shift, budget)
    model.eval()
    for guard in [-1.0, float("inf"), float("nan")]:
        with pytest.raises(ValueError, match="guard"):
            evaluate_geometry_baselines(model, features, shift, budget, guard=guard)
    with pytest.raises(ValueError, match="row stochastic"):
        evaluate_geometry_baselines(model, features, 0.5 * shift, budget)
    invalid = budget.clone()
    invalid[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_geometry_baselines(model, features, shift, invalid)
    with torch.no_grad():
        logits = model(features, shift)
    with pytest.raises(ValueError, match="cached logits"):
        evaluate_geometry_baselines(model, features, shift, budget, logits=logits + 0.1)
    basis = retained_mass_uncertainty_basis(shift, budget, model.order_count)
    basis[-1] = torch.zeros_like(budget)
    with pytest.raises(ValueError, match="cached basis"):
        evaluate_geometry_baselines(model, features, shift, budget, basis=basis)
    model = _model("linear_relu")
    model.spectral_fusion_scale = 0.0
    with pytest.raises(ValueError, match="forward does not apply"):
        evaluate_geometry_baselines(model, features, shift, budget)


def test_direction_span_closed_form_and_diameter_bound():
    q = torch.tensor([[3.0, -4.0], [3.0, 4.0], [-1.0, 1.0], [0.0, 0.0]])
    span = positive_orthant_direction_span(q)
    assert torch.allclose(span, torch.tensor([7.0, 5.0, 2.0, 0.0]))
    assert torch.all(span <= math.sqrt(2.0) * q.norm(dim=1) + 1e-6)
    with pytest.raises(ValueError, match="finite"):
        positive_orthant_direction_span(torch.tensor([[float("nan")]]))


def test_float32_zero_relu_and_sparse_csr_paths():
    model = _model().float()
    features, shift, budget = _inputs(77)
    features, shift, budget = features.float(), shift.float(), budget.float()
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.zero_()
        model.theta.copy_(torch.tensor([0.1, -0.7, 0.3, -0.4]))
    coo = evaluate_geometry_baselines(model, features, shift, budget)
    with pytest.warns(UserWarning, match="Sparse CSR tensor support"):
        csr_shift = shift.to_sparse_csr()
    csr = evaluate_geometry_baselines(model, features, csr_shift, budget)
    for key in coo:
        assert torch.isfinite(coo[key].worst_slack).all()
        assert torch.allclose(coo[key].worst_slack, csr[key].worst_slack, atol=2e-6, rtol=2e-6)
        assert torch.equal(coo[key].certified, csr[key].certified)
    assert torch.all(coo["cone"].worst_slack >= coo["nonnegative_diameter"].worst_slack - 2e-6)
    assert torch.all(coo["nonnegative_diameter"].worst_slack >= coo["ball_pairwise"].worst_slack - 2e-6)


def test_nonfinite_parameters_are_rejected_before_acceptance():
    model = _model()
    features, shift, budget = _inputs(88)
    with torch.no_grad():
        model.classifier.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_geometry_baselines(model, features, shift, budget)
