from __future__ import annotations

import json
from pathlib import Path

import torch

from openfgl.flcore.fedbcert import PolynomialSpectralClassifier


ROOT = Path(__file__).resolve().parents[1]


def test_e9a_matrix_and_test_isolation_are_frozen() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e9a_protocol.json").read_text(encoding="utf-8")
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


def test_e9a_residual_encoder_is_bounded_in_train_and_eval() -> None:
    torch.manual_seed(9)
    model = PolynomialSpectralClassifier(
        7,
        16,
        3,
        3,
        0.5,
        encoder_type="bounded_residual_mlp",
        encoder_depth=3,
        encoder_residual_scale=0.5,
    )
    features = torch.randn(32, 7)
    model.train()
    train_hidden = model.bounded_encoding(features)
    model.eval()
    eval_hidden = model.bounded_encoding(features)
    assert torch.all(train_hidden.norm(p=2, dim=1) <= 1.0 + 1e-6)
    assert torch.all(eval_hidden.norm(p=2, dim=1) <= 1.0 + 1e-6)


def test_e8_default_encoder_state_layout_is_unchanged() -> None:
    model = PolynomialSpectralClassifier(7, 16, 3, 3, 0.2)
    assert set(model.state_dict()) == {
        "theta",
        "encoder.weight",
        "encoder.bias",
        "classifier.weight",
        "classifier.bias",
    }
