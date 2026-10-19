from __future__ import annotations

import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    cone_pairwise_certificate_statistics,
    positive_orthant_unit_support,
    retained_mass_uncertainty_basis,
)


def test_positive_orthant_support_is_exact() -> None:
    direction = torch.tensor([[3.0, -4.0], [-5.0, -12.0], [0.0, 0.0]])
    lower, upper = positive_orthant_unit_support(direction)
    assert torch.allclose(lower, torch.tensor([-4.0, -13.0, 0.0]))
    assert torch.allclose(upper, torch.tensor([3.0, 0.0, 0.0]))


def _identity_shift(count: int) -> torch.Tensor:
    return torch.eye(count).to_sparse_coo().coalesce()


def test_cone_hybrid_contains_the_symmetric_certificate() -> None:
    torch.manual_seed(18)
    model = PolynomialSpectralClassifier(
        input_dim=3,
        hidden_dim=4,
        num_classes=3,
        order_count=3,
        dropout=0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        spectral_fusion_scale=0.25,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    features = torch.randn(5, 3)
    shift = _identity_shift(5)
    row_budget = torch.tensor([0.0, 0.4, 1.0, 1.5, 2.0])
    basis = retained_mass_uncertainty_basis(shift, row_budget, 3)
    with torch.no_grad():
        logits = model(features, shift)
        raw = cone_pairwise_certificate_statistics(
            model, features, shift, row_budget, logits, basis,
            combine_symmetric=False,
        )
        hybrid = cone_pairwise_certificate_statistics(
            model, features, shift, row_budget, logits, basis,
            combine_symmetric=True,
        )
    assert torch.all(~raw.certified | hybrid.certified)
    from openfgl.flcore.fedbcert import certificate_statistics

    symmetric = certificate_statistics(model, logits, basis)
    assert torch.all(~symmetric.certified | hybrid.certified)
    assert torch.all(hybrid.worst_slack >= symmetric.worst_slack)


def test_cone_certificate_is_strictly_tighter_for_a_positive_direction() -> None:
    model = PolynomialSpectralClassifier(
        input_dim=1,
        hidden_dim=2,
        num_classes=2,
        order_count=2,
        dropout=0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        spectral_fusion_scale=1.0,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    with torch.no_grad():
        model.encoder.weight.copy_(torch.tensor([[1.0], [1.0]]))
        model.encoder.bias.zero_()
        model.theta.copy_(torch.tensor([0.0, 1.0]))
        model.classifier.weight.copy_(torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
        model.classifier.bias.zero_()
        assert model.feature_classifier is not None
        model.feature_classifier.weight.zero_()
        model.feature_classifier.bias.zero_()
    features = torch.ones(1, 1)
    shift = _identity_shift(1)
    row_budget = torch.ones(1)
    basis = retained_mass_uncertainty_basis(shift, row_budget, 2)
    with torch.no_grad():
        logits = model(features, shift)
        raw = cone_pairwise_certificate_statistics(
            model, features, shift, row_budget, logits, basis,
            combine_symmetric=False,
        )
    from openfgl.flcore.fedbcert import certificate_statistics

    symmetric = certificate_statistics(model, logits, basis)
    assert not symmetric.certified.item()
    assert raw.certified.item()
    assert raw.worst_slack.item() > 0.7


def test_cone_certificate_is_sound_for_zero_or_maximum_stub_count() -> None:
    torch.manual_seed(1801)
    model = PolynomialSpectralClassifier(
        input_dim=2,
        hidden_dim=3,
        num_classes=2,
        order_count=3,
        dropout=0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        spectral_fusion_scale=0.5,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    local_features = torch.randn(2, 2)
    local_shift = _identity_shift(2)
    # Node zero has an upper bound of one missing edge: delta=2/(1+1)=1.
    row_budget = torch.tensor([1.0, 0.0])
    basis = retained_mass_uncertainty_basis(local_shift, row_budget, 3)
    with torch.no_grad():
        hidden = model.bounded_encoding(local_features)
        local_spectral = model.classifier(model.filter(hidden, local_shift))
        invariant_logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
        local_logits = invariant_logits + 0.5 * local_spectral
        stats = cone_pairwise_certificate_statistics(
            model,
            local_features,
            local_shift,
            row_budget,
            local_logits,
            basis,
            count_semantics="upper_bound",
            combine_symmetric=False,
        )
        assert bool(stats.certified.all())

        for actual_stub_count in (0, 1):
            for _ in range(32):
                remote = torch.rand(1, hidden.size(1))
                remote = torch.nn.functional.normalize(remote, p=2, dim=1)
                full_hidden = torch.cat([hidden, remote], dim=0)
                if actual_stub_count == 0:
                    full_shift = torch.eye(3).to_sparse_coo().coalesce()
                else:
                    # The first local row retains half its self mass and sends
                    # half to the unknown endpoint.  The remote row is a valid
                    # convex combination and does not need to be observed.
                    dense = torch.tensor(
                        [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
                    )
                    full_shift = dense.to_sparse_coo().coalesce()
                full_spectral = model.classifier(
                    model.filter(full_hidden, full_shift)
                )[:2]
                full_logits = invariant_logits + 0.5 * full_spectral
                assert torch.equal(
                    local_logits.argmax(dim=1), full_logits.argmax(dim=1)
                )
