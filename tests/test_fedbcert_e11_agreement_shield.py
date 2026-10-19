from __future__ import annotations

import json
from pathlib import Path

import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    prediction_certificate_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_agreement_shield_preserves_utility_predictions() -> None:
    torch.manual_seed(1101)
    node_count = 9
    model = PolynomialSpectralClassifier(
        7,
        16,
        4,
        3,
        0.0,
        encoder_type="shielded_sage_spectral",
        encoder_depth=2,
    ).eval()
    features = torch.randn(node_count, 7)
    shift = torch.eye(node_count).to_sparse()
    basis = [
        torch.zeros(node_count),
        torch.full((node_count,), 0.15),
        torch.full((node_count,), 0.25),
    ]
    with torch.no_grad():
        result = prediction_certificate_statistics(model, features, shift, basis)
    assert torch.equal(
        result.logits.argmax(dim=1), result.utility_logits.argmax(dim=1)
    )
    safe_agreement = (
        result.safe_logits.argmax(dim=1)
        == result.utility_logits.argmax(dim=1)
    )
    assert torch.all(~result.certificate.certified | safe_agreement)
    assert torch.equal(result.certificate.certified, result.shield)


def test_unshielded_utility_nodes_are_never_claimed_bounded() -> None:
    torch.manual_seed(1102)
    node_count = 6
    model = PolynomialSpectralClassifier(
        5,
        8,
        3,
        3,
        0.0,
        encoder_type="shielded_sage_spectral",
        encoder_depth=3,
    ).eval()
    features = torch.randn(node_count, 5)
    shift = torch.eye(node_count).to_sparse()
    # The maximal basis makes every nontrivial spectral margin uncertified.
    basis = [
        torch.zeros(node_count),
        torch.full((node_count,), 2.0),
        torch.full((node_count,), 2.0),
    ]
    with torch.no_grad():
        result = prediction_certificate_statistics(model, features, shift, basis)
    unshielded = ~result.shield
    assert bool(unshielded.any())
    assert torch.isinf(
        result.certificate.class_logit_bounds[unshielded]
    ).all()


def test_certificate_auxiliary_has_no_utility_gradient() -> None:
    torch.manual_seed(1103)
    model = PolynomialSpectralClassifier(
        5,
        8,
        3,
        3,
        0.0,
        encoder_type="shielded_sage_spectral",
        encoder_depth=2,
    )
    features = torch.randn(7, 5)
    shift = torch.eye(7).to_sparse()
    _, certificate_logits, _, _ = model.training_logits(features, shift)
    loss = certificate_logits.square().mean()
    gradients = torch.autograd.grad(
        loss, tuple(model.parameters()), allow_unused=True
    )
    named = dict(zip((name for name, _ in model.named_parameters()), gradients))
    assert all(
        gradient is None
        for name, gradient in named.items()
        if name.startswith("utility_")
    )


def test_e11_development_and_confirmation_seeds_are_isolated() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e11a_protocol.json").read_text(encoding="utf-8")
    )
    count = (
        len(config["datasets"])
        * len(config["num_clients"])
        * len(config["partitions"])
        * len(config["development_seeds"])
        * len(config["variants"])
    )
    assert count == config["expected_development_units"]
    assert config["evaluation_scope"] == "validation_only"
    assert config["selection_uses_test_metrics"] is False
    assert set(config["development_seeds"]).isdisjoint(
        config["reserved_confirmation_seeds"]
    )
    assert set(config["development_seeds"]).isdisjoint({21, 22, 23, 32, 33, 34})
