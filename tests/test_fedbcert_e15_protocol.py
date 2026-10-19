from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e15_is_selection_free_and_disjoint_from_e14() -> None:
    e14 = json.loads(
        (ROOT / "configs" / "fedbcert_e14a_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    e15 = json.loads(
        (ROOT / "configs" / "fedbcert_e15_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert e15["model_selection"] is False
    assert e15["test_labels_used_for_selection"] is False
    assert e15["training_method"].startswith("standard FedAvg only")
    assert set(e15["confirmation_seeds"]).isdisjoint(e14["development_seeds"])
    assert set(e15["confirmation_seeds"]).isdisjoint(
        e15["untouched_e14b_seeds"]
    )
    assert e15["untouched_e14b_seeds"] == e14["reserved_confirmation_seeds"]


def test_e15_matrix_and_fixed_frontier_are_frozen() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "fedbcert_e15_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    variants = protocol["variants"]
    assert variants["feature_alpha0"] == {
        "encoder_type": "feature_only_mlp",
        "encoder_depth": 2,
        "alpha": 0.0,
    }
    assert [variants[f"dual_alpha{label}"]["alpha"] for label in (
        "010", "025", "050", "100"
    )] == [0.1, 0.25, 0.5, 1.0]
    expected = (
        len(protocol["datasets"])
        * len(protocol["num_clients"])
        * len(protocol["partitions"])
        * len(protocol["confirmation_seeds"])
        * len(variants)
    )
    assert protocol["num_clients"] == [3, 10]
    assert "spectral_only" not in variants
    assert protocol["expected_runner_units"] == expected == 360
    assert protocol["expected_metric_rows"] == expected
    assert protocol["fixed_hyperparameters"]["uncertainty_recurrence"] == (
        "retained_mass"
    )
    assert protocol["fixed_hyperparameters"]["certificate_margin_mode"] == (
        "pairwise_difference"
    )
