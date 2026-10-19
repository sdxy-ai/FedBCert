from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e20_seed_separation_and_matrix_sizes() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e20_identity_null_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    development = set(protocol["development_seeds"])
    confirmation = set(protocol["reserved_confirmation_seeds"])
    assert development.isdisjoint(confirmation)
    assert protocol["expected_development_runner_units"] == (
        len(protocol["datasets"])
        * len(protocol["development_seeds"])
        * len(protocol["alphas"])
        * 2
    )
    matrix = protocol["confirmation_matrix"]
    assert confirmation == set(matrix["seeds"])
    assert matrix["expected_runner_units"] == (
        len(protocol["datasets"])
        * len(matrix["num_clients"])
        * len(matrix["partitions"])
        * len(matrix["seeds"])
        * len(matrix["variants"])
    )


def test_e20_development_is_validation_only_and_confirmation_is_gated() -> None:
    development = (
        ROOT / "scripts/run_fedbcert_e20a_identity_null_development.sh"
    ).read_text(encoding="utf-8")
    confirmation = (
        ROOT / "scripts/run_fedbcert_e20b_identity_null_confirmation.sh"
    ).read_text(encoding="utf-8")
    assert "--evaluation-scope validation_only" in development
    assert "87" not in development and "88" not in development and "89" not in development
    assert "dual_feature_identity_null_spectral" in development
    assert "observed identity" in development
    assert "proceed_to_e20b" in confirmation
    assert "--evaluation-scope full" in confirmation
    assert "87 88 89" in confirmation
    assert "observed identity" in confirmation


def test_e20_contract_removes_the_identity_capacity_confound() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e20_identity_null_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    contract = protocol["identity_null_contract"]
    assert contract["spectral_classifier_bias"] is False
    assert contract["identity_shift_spectral_logits"] == "exactly zero"
    assert protocol["selection_rule"]["minimum_positive_dataset_count"] >= 5
    assert protocol["confirmation_claims"]["macro_f1_hierarchical_ci_low_min"] == 0.0
