from __future__ import annotations

import json
from pathlib import Path

import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    external_teacher_agreement_statistics,
    teacher_target_certified_hinge_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def test_external_teacher_shield_preserves_every_teacher_class() -> None:
    torch.manual_seed(2201)
    model = PolynomialSpectralClassifier(
        6, 12, 4, 3, 0.0,
        encoder_type="linear_relu",
        encoder_depth=1,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    features = torch.randn(10, 6)
    shift = torch.eye(10).to_sparse()
    basis = [torch.zeros(10), torch.full((10,), 0.1), torch.full((10,), 0.2)]
    teacher = torch.randn(10, 4)
    result = external_teacher_agreement_statistics(
        model, features, shift, basis, teacher, certificate_kind="cone_hybrid"
    )
    assert torch.equal(result.logits.argmax(1), teacher.argmax(1))
    assert torch.all(
        ~result.shield
        | (result.safe_logits.argmax(1) == teacher.argmax(1))
    )
    assert torch.isinf(result.certificate.class_logit_bounds[~result.shield]).all()


def test_cad_does_not_backpropagate_into_teacher() -> None:
    torch.manual_seed(2202)
    model = PolynomialSpectralClassifier(
        5, 8, 3, 3, 0.0,
        encoder_type="linear_relu",
        certificate_margin_mode="pairwise_difference",
    )
    features = torch.randn(7, 5)
    shift = torch.eye(7).to_sparse()
    safe = model.spectral_logits(features, shift)
    teacher = torch.randn(7, 3, requires_grad=True)
    basis = [torch.zeros(7), torch.full((7,), 0.1), torch.full((7,), 0.2)]
    loss = teacher_target_certified_hinge_loss(
        model, safe, teacher, torch.ones(7, dtype=torch.bool), basis, 0.05
    )
    teacher_gradient = torch.autograd.grad(
        loss, teacher, allow_unused=True, retain_graph=True
    )[0]
    assert teacher_gradient is None
    model_gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    assert any(gradient is not None for gradient in model_gradients)


def test_e22_seed_and_matrix_contract() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e22_teacher_shield_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    expected = (
        len(config["datasets"])
        * len(config["num_clients"])
        * len(config["partitions"])
        * len(config["development_seeds"])
        * len(config["variants"])
    )
    assert expected == config["expected_development_units"] == 40
    assert set(config["development_seeds"]).isdisjoint(
        config["reserved_confirmation_seeds"]
    )
    assert set(config["development_seeds"]).isdisjoint(
        {21, 22, 23, 101, 102, 103, 104, 105}
    )
    assert "feature_kd_cad" in config["variants"]
    assert config["test_metrics_used_for_selection"] is False
