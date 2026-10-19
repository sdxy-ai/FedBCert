from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_safe_cover_is_monotone_union() -> None:
    feature = torch.tensor([True, False, True, False, False])
    spectral = torch.tensor([False, True, True, False, True])
    spectral_route = (~feature) & spectral
    cover = feature | spectral_route
    assert torch.equal(cover, feature | spectral)
    assert int(cover.sum()) >= int(feature.sum())
    assert int(cover.sum()) >= int(spectral.sum())


def test_e23_is_no_selection_independent_confirmation() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e23_safe_cover_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["model_or_hyperparameter_selection_in_e23"] is False
    assert config["expected_teacher_units"] == 24
    assert config["expected_component_training_units"] == 72
    assert config["expected_cover_evaluation_units"] == 24
    assert set(config["confirmation_seeds"]).isdisjoint(
        config["untouched_e22_confirmation_seeds"]
    )
    assert config["fixed_cover_priority"] == [
        "feature_kd_cad", "spectral_cad", "FedLap+ utility teacher"
    ]
