from __future__ import annotations

import json
from pathlib import Path

import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    prediction_certificate_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def make_model(hops: int = 2, scale: float = 0.25) -> PolynomialSpectralClassifier:
    return PolynomialSpectralClassifier(
        6,
        12,
        3,
        3,
        0.0,
        encoder_type="shielded_diffusion_spectral",
        encoder_depth=2,
        encoder_residual_scale=0.5,
        utility_diffusion_hops=hops,
        utility_diffusion_scale=scale,
    )


def test_diffusion_utility_uses_local_structure_and_preserves_shield_rule() -> None:
    torch.manual_seed(1201)
    model = make_model().eval()
    features = torch.randn(8, 6)
    identity = torch.eye(8).to_sparse()
    rolled = torch.roll(torch.eye(8), shifts=1, dims=1).to_sparse()
    basis = [torch.zeros(8), torch.full((8,), 0.2), torch.full((8,), 0.3)]
    with torch.no_grad():
        identity_utility = model.utility_logits(features, identity)
        rolled_utility = model.utility_logits(features, rolled)
        result = prediction_certificate_statistics(model, features, identity, basis)
    assert not torch.allclose(identity_utility, rolled_utility)
    assert torch.equal(
        result.logits.argmax(dim=1), result.utility_logits.argmax(dim=1)
    )
    assert torch.all(
        ~result.certificate.certified
        | (
            result.safe_logits.argmax(dim=1)
            == result.utility_logits.argmax(dim=1)
        )
    )


def test_certificate_objective_cannot_update_e12_utility_modules() -> None:
    torch.manual_seed(1202)
    model = make_model()
    features = torch.randn(7, 6)
    shift = torch.eye(7).to_sparse()
    _, certificate_logits, _, _ = model.training_logits(features, shift)
    loss = certificate_logits.square().mean()
    gradients = torch.autograd.grad(
        loss, tuple(model.parameters()), allow_unused=True
    )
    named = dict(zip((name for name, _ in model.named_parameters()), gradients))
    utility_prefixes = ("feature_", "diffusion_")
    assert all(
        gradient is None
        for name, gradient in named.items()
        if name.startswith(utility_prefixes)
    )


def test_e12_protocol_matrix_and_seed_isolation() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e12a_protocol.json").read_text(encoding="utf-8")
    )
    expected = (
        len(config["datasets"])
        * len(config["num_clients"])
        * len(config["partitions"])
        * len(config["development_seeds"])
        * len(config["variants"])
    )
    assert expected == config["expected_development_units"] == 40
    assert config["evaluation_scope"] == "validation_only"
    assert config["selection_uses_test_metrics"] is False
    assert set(config["development_seeds"]).isdisjoint(
        config["reserved_confirmation_seeds"]
    )
    assert set(config["development_seeds"]).isdisjoint(
        config["e11_reserved_seeds_left_untouched"]
    )
    assert config["frozen_dataset_route"]["Cora"] == "E10 spectral control"
    assert config["frozen_dataset_route"]["Computers"] == "E12 candidate"
