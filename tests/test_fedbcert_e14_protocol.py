from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_e14a_pure_feature_gate_and_matrix_are_frozen() -> None:
    protocol = load("fedbcert_e14a_protocol.json")
    variants = protocol["variants"]
    assert variants["feature_alpha0"]["alpha"] == 0.0
    assert variants["feature_alpha0"]["encoder_type"] == "feature_only_mlp"
    assert variants["feature_alpha0"]["include_local"] is True
    assert protocol["expected_runner_units"] == (
        len(protocol["datasets"])
        * len(protocol["development_seeds"])
        * len(variants)
    )
    assert protocol["expected_metric_rows"] == 156
    assert protocol["unified_hyperparameters"]["uncertainty_recurrence"] == "retained_mass"
    assert protocol["unified_hyperparameters"]["certificate_margin_mode"] == "pairwise_difference"
    assert protocol["gate"]["hierarchical_bootstrap_structural_lift_lower_min"] == 0.0
    assert protocol["gate"]["pure_feature_certified_ratio_min"] >= 0.999999


def test_e14b_is_independent_and_uses_a_single_selected_alpha() -> None:
    development = load("fedbcert_e14a_protocol.json")
    confirmation = load("fedbcert_e14b_protocol.json")
    assert set(development["development_seeds"]).isdisjoint(
        confirmation["confirmation_seeds"]
    )
    assert confirmation["confirmation_seeds"] == development["reserved_confirmation_seeds"]
    assert confirmation["num_clients"] == [3, 5, 10]
    assert confirmation["partitions"] == ["metis", "random"]
    assert confirmation["variants"] == [
        "feature_alpha0",
        "selected_dual",
        "spectral_only",
    ]
    assert confirmation["expected_runner_units"] == (
        len(confirmation["datasets"])
        * len(confirmation["num_clients"])
        * len(confirmation["partitions"])
        * len(confirmation["confirmation_seeds"])
        * len(confirmation["variants"])
    )
    assert confirmation["noninferiority"]["margin"] == 0.005
    assert confirmation["claims"]["no_dataset_specific_model_route"] is True
