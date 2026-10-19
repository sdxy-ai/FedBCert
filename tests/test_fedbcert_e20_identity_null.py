from __future__ import annotations

import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    cone_pairwise_certificate_statistics,
    retained_mass_uncertainty_basis,
)


def sparse_shift(dense: torch.Tensor) -> torch.Tensor:
    return dense.to_sparse_coo().coalesce()


def make_model() -> PolynomialSpectralClassifier:
    torch.manual_seed(20)
    return PolynomialSpectralClassifier(
        input_dim=5,
        hidden_dim=8,
        num_classes=3,
        order_count=3,
        dropout=0.0,
        encoder_type="dual_feature_identity_null_spectral",
        encoder_depth=2,
        spectral_fusion_scale=0.25,
        certificate_margin_mode="pairwise_difference",
    ).eval()


def test_identity_null_branch_is_exactly_zero_for_identity_shift() -> None:
    model = make_model()
    features = torch.randn(7, 5)
    identity = sparse_shift(torch.eye(7))

    spectral = model.spectral_logits(features, identity)
    fused, _, feature, _ = model.training_logits(features, identity)

    assert model.classifier.bias is None
    assert torch.count_nonzero(spectral).item() == 0
    assert torch.equal(fused, feature)


def test_identity_null_subtraction_does_not_change_completion_difference() -> None:
    model = make_model()
    features = torch.randn(4, 5)
    observed = sparse_shift(
        torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )
    )
    completed = sparse_shift(
        torch.tensor(
            [
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.5],
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.5],
            ]
        )
    )
    hidden = model.bounded_encoding(features)

    def uncentered(shift: torch.Tensor) -> torch.Tensor:
        current = hidden
        output = torch.zeros_like(hidden)
        for order in range(1, model.order_count):
            current = torch.sparse.mm(shift, current)
            output = output + model.theta[order] * current
        return output

    centered_difference = model.filter(hidden, completed) - model.filter(
        hidden, observed
    )
    uncentered_difference = uncentered(completed) - uncentered(observed)
    assert torch.allclose(centered_difference, uncentered_difference, atol=1e-7)


def test_identity_null_cone_certificate_executes_without_bias() -> None:
    model = make_model()
    features = torch.randn(5, 5)
    shift = sparse_shift(torch.eye(5))
    row_budget = torch.full((5,), 0.4)
    basis = retained_mass_uncertainty_basis(
        shift, row_budget, model.order_count
    )
    logits = model(features, shift)
    stats = cone_pairwise_certificate_statistics(
        model,
        features,
        shift,
        row_budget,
        logits,
        basis,
        count_semantics="upper_bound",
    )
    assert stats.certified.shape == (5,)
    assert torch.isfinite(stats.representation_radius).all()
