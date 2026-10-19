from __future__ import annotations

import torch
from types import SimpleNamespace

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    certificate_statistics,
    project_auxiliary_gradient,
    retained_mass_uncertainty_basis,
    uncertainty_basis,
)
from openfgl.flcore.fedbcert.federated import (
    _completion_logits_with_aligned_nodewise_encoding,
    _logit_audit_tolerance,
)


def row_stochastic_shift() -> torch.Tensor:
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    values = torch.tensor([0.75, 0.25, 0.20, 0.80])
    return torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()


def test_retained_mass_recurrence_tightens_legacy_and_is_budget_monotone() -> None:
    shift = row_stochastic_shift()
    budget = torch.tensor([0.4, 1.0])
    legacy = uncertainty_basis(shift, budget, 4)
    tight = retained_mass_uncertainty_basis(shift, budget, 4)
    upper = retained_mass_uncertainty_basis(shift, budget + 0.2, 4)
    for order in range(4):
        assert torch.all(tight[order] <= legacy[order] + 1e-7)
        assert torch.all(tight[order] <= upper[order] + 1e-7)
    assert torch.any(tight[2] < legacy[2])


def test_pairwise_weight_difference_is_tighter_than_classwise_sum() -> None:
    basis = [torch.zeros(1), torch.ones(1), torch.ones(1)]
    common = dict(
        input_dim=2,
        hidden_dim=2,
        num_classes=2,
        order_count=3,
        dropout=0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        spectral_fusion_scale=0.25,
    )
    loose = PolynomialSpectralClassifier(
        **common, certificate_margin_mode="classwise_sum"
    )
    tight = PolynomialSpectralClassifier(
        **common, certificate_margin_mode="pairwise_difference"
    )
    tight.load_state_dict(loose.state_dict())
    with torch.no_grad():
        loose.classifier.weight.copy_(torch.tensor([[1.0, 0.0], [0.5, 0.0]]))
        tight.classifier.weight.copy_(loose.classifier.weight)
        loose.theta.copy_(torch.tensor([1.0, 0.5, 0.0]))
        tight.theta.copy_(loose.theta)
    logits = torch.tensor([[0.10, 0.00]])
    loose_stats = certificate_statistics(loose, logits, basis)
    tight_stats = certificate_statistics(tight, logits, basis)
    assert tight_stats.worst_slack.item() > loose_stats.worst_slack.item()
    assert not loose_stats.certified.item()
    assert tight_stats.certified.item()


def test_zero_spectral_scale_is_the_pure_feature_degeneration() -> None:
    model = PolynomialSpectralClassifier(
        input_dim=2,
        hidden_dim=2,
        num_classes=2,
        order_count=3,
        dropout=0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        spectral_fusion_scale=0.0,
        certificate_margin_mode="pairwise_difference",
    )
    basis = [torch.zeros(2), torch.ones(2), torch.full((2,), 2.0)]
    # The second row deliberately ties.  Since alpha=0 makes every logit
    # graph invariant, deterministic argmax remains invariant as well.
    logits = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    stats = certificate_statistics(model, logits, basis)
    assert torch.equal(stats.representation_radius, torch.zeros(2))
    assert torch.equal(stats.certified, torch.ones(2, dtype=torch.bool))


def test_feature_only_mlp_is_graph_invariant_and_smaller_than_dual() -> None:
    feature_only = PolynomialSpectralClassifier(
        input_dim=3,
        hidden_dim=4,
        num_classes=2,
        order_count=3,
        dropout=0.0,
        encoder_type="feature_only_mlp",
        encoder_depth=2,
        spectral_fusion_scale=0.0,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    dual = PolynomialSpectralClassifier(
        input_dim=3,
        hidden_dim=4,
        num_classes=2,
        order_count=3,
        dropout=0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        spectral_fusion_scale=0.25,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    features = torch.randn(3, 3)
    identity = torch.eye(3).to_sparse_coo()
    dense_mean = torch.full((3, 3), 1.0 / 3.0).to_sparse_coo()
    with torch.no_grad():
        left = feature_only(features, identity)
        right = feature_only(features, dense_mean)
    assert torch.equal(left, right)
    assert sum(p.numel() for p in feature_only.parameters()) < sum(
        p.numel() for p in dual.parameters()
    )


def test_logit_audit_tolerance_covers_float32_batch_roundoff_only() -> None:
    local = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    completion = local + torch.tensor([[5e-6, 2e-5]], dtype=torch.float32)
    tolerance = _logit_audit_tolerance(local, completion)
    assert tolerance[0, 0] >= 5e-6
    assert tolerance[0, 1] < 1e-4
    assert torch.equal(local.argmax(dim=1), completion.argmax(dim=1))


def test_aligned_completion_is_exact_for_pure_feature_model() -> None:
    model = PolynomialSpectralClassifier(
        input_dim=3,
        hidden_dim=4,
        num_classes=2,
        order_count=3,
        dropout=0.0,
        encoder_type="feature_only_mlp",
        encoder_depth=2,
        spectral_fusion_scale=0.0,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    features = torch.randn(5, 3)
    clients = [
        SimpleNamespace(global_nodes=torch.tensor([0, 2]), features=features[[0, 2]]),
        SimpleNamespace(global_nodes=torch.tensor([1, 3, 4]), features=features[[1, 3, 4]]),
    ]
    local = torch.empty(5, 2)
    with torch.no_grad():
        for client in clients:
            count = int(client.global_nodes.numel())
            identity = torch.eye(count).to_sparse_coo()
            local[client.global_nodes] = model(client.features, identity)
        full = _completion_logits_with_aligned_nodewise_encoding(
            model, clients, torch.eye(5).to_sparse_coo(), local
        )
    assert torch.equal(local, full)


def test_norm_cap_proves_the_weaker_primary_descent_bound() -> None:
    primary = [torch.tensor([1.0, 2.0])]
    auxiliary = [torch.tensor([-3.0, -4.0])]
    projected, diagnostics = project_auxiliary_gradient(
        primary, auxiliary, epsilon=1e-3
    )
    assert diagnostics["post_projection_dot"] < 0.0
    primary_norm = torch.linalg.vector_norm(primary[0])
    auxiliary_norm = torch.linalg.vector_norm(projected[0])
    q = 0.25
    scale = min(1.0, float(q * primary_norm / auxiliary_norm))
    combined = primary[0] + scale * projected[0]
    dot = torch.dot(primary[0], combined)
    assert dot >= (1.0 - q) * primary_norm.square() - 1e-6
