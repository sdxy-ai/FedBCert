from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    canonical_undirected,
    certificate_statistics,
    certified_hinge_loss,
    induced_edges,
    missing_degrees,
    row_normalized_shift,
    uncertainty_basis,
)


ROOT = Path(__file__).resolve().parents[1]


def graph_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    full = canonical_undirected(
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), 4
    )
    membership = np.asarray([0, 0, 1, 1])
    return full, induced_edges(full, membership)


def test_dual_feature_expert_has_zero_boundary_deviation() -> None:
    torch.manual_seed(2026)
    full, local = graph_fixture()
    local_shift, local_degree = row_normalized_shift(
        local, 4, device=torch.device("cpu")
    )
    full_shift, _ = row_normalized_shift(full, 4, device=torch.device("cpu"))
    basis = uncertainty_basis(
        local_shift,
        boundary_uncertainty(local_degree, missing_degrees(full, local, 4)),
        3,
    )
    model = PolynomialSpectralClassifier(
        5,
        16,
        3,
        3,
        0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        encoder_residual_scale=0.5,
    ).eval()
    features = torch.randn(4, 5)
    with torch.no_grad():
        # Make the invariant expert dominate the logits; it must still add
        # exactly zero local/full graph deviation.
        assert model.feature_classifier is not None
        model.feature_classifier.weight.mul_(50.0)
        invariant_logits = model.feature_classifier(
            model.feature_encoding(features)
        )
        local_spectral = model.spectral_logits(features, local_shift)
        full_spectral = model.spectral_logits(features, full_shift)
        local_logits = model(features, local_shift)
        full_logits = model(features, full_shift)
        stats = certificate_statistics(model, local_logits, basis)
    scale = model.spectral_fusion_scale
    assert torch.allclose(local_logits, scale * local_spectral + invariant_logits)
    assert torch.allclose(full_logits, scale * full_spectral + invariant_logits)
    assert torch.allclose(
        full_logits - local_logits,
        scale * (full_spectral - local_spectral),
        # Subtracting deliberately amplified invariant logits introduces a
        # few float32 ULPs even though the branch is algebraically identical.
        atol=5e-6,
        rtol=1e-6,
    )
    assert torch.all(
        (full_logits - local_logits).abs()
        <= stats.class_logit_bounds + 1e-6
    )


def test_certificate_objective_does_not_update_feature_expert() -> None:
    torch.manual_seed(13)
    model = PolynomialSpectralClassifier(
        7,
        8,
        3,
        3,
        0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        encoder_residual_scale=0.5,
    )
    features = torch.randn(6, 7)
    shift = torch.eye(6).to_sparse()
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    mask = torch.ones(6, dtype=torch.bool)
    basis = [torch.zeros(6), torch.full((6,), 0.2), torch.full((6,), 0.3)]
    _, certificate_logits, _, _ = model.training_logits(features, shift)
    loss = certified_hinge_loss(
        model, certificate_logits, labels, mask, basis, margin=0.05
    )
    gradients = torch.autograd.grad(
        loss, tuple(model.parameters()), allow_unused=True
    )
    named_gradients = dict(
        zip((name for name, _ in model.named_parameters()), gradients)
    )
    assert all(
        named_gradients[name] is None
        for name in named_gradients
        if name.startswith("feature_")
    )


def test_e10a_development_and_confirmation_seeds_are_isolated() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e10a_protocol.json").read_text(encoding="utf-8")
    )
    count = (
        len(config["datasets"])
        * len(config["num_clients"])
        * len(config["partitions"])
        * len(config["development_seeds"])
        * len(config["variants"])
    )
    assert count == config["expected_development_units"] == 24
    assert config["evaluation_scope"] == "validation_only"
    assert config["selection_uses_test_metrics"] is False
    assert set(config["development_seeds"]).isdisjoint(
        config["reserved_confirmation_seeds"]
    )
    assert all(seed >= 30 for seed in config["development_seeds"])
