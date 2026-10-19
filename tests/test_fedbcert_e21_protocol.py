from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e21_seed_separation_and_matrix_sizes() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e21_staged_residual_protocol.json").read_text(
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


def test_e21_is_validation_only_staged_and_confirmation_gated() -> None:
    script = (ROOT / "scripts/run_fedbcert_e21a_staged_residual_development.sh").read_text(
        encoding="utf-8"
    )
    assert "--evaluation-scope validation_only" in script
    assert "--identity-null-training-mode staged_residual" in script
    assert "--feature-pretrain-rounds 100" in script
    assert "observed identity" in script
    assert all(str(seed) not in script for seed in (103, 104, 105))


def test_e21_contract_is_one_global_strict_structural_test() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e21_staged_residual_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    contract = protocol["training_contract"]
    assert contract["spectral_classifier_initialization"] == "exactly zero"
    assert contract["stage_2_dropout"] is False
    assert protocol["selection_rule"]["minimum_positive_dataset_count"] >= 5
    assert protocol["confirmation_claims"]["macro_f1_hierarchical_ci_low_min"] == 0.0
