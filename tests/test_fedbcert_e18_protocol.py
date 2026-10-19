from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e18_protocol_keeps_confirmation_seeds_separate() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e18_cone_protocol.json").read_text(encoding="utf-8")
    )
    development = set(protocol["development_seeds"])
    confirmation = set(protocol["reserved_confirmation_seeds"])
    assert development.isdisjoint(confirmation)
    assert confirmation == set(protocol["confirmation_matrix"]["seeds"])
    assert protocol["expected_development_runner_units"] == (
        len(protocol["datasets"])
        * len(protocol["development_seeds"])
        * len(protocol["variants"])
    )
    matrix = protocol["confirmation_matrix"]
    assert matrix["expected_runner_units"] == (
        len(protocol["datasets"])
        * len(matrix["num_clients"])
        * len(matrix["partitions"])
        * len(matrix["seeds"])
        * len(matrix["variants"])
    )


def test_e18_scripts_enforce_validation_then_confirmation() -> None:
    development = (ROOT / "scripts/run_fedbcert_e18a_cone_development.sh").read_text()
    confirmation = (ROOT / "scripts/run_fedbcert_e18b_cone_confirmation.sh").read_text()
    assert "--evaluation-scope validation_only" in development
    assert "73" not in development and "74" not in development and "75" not in development
    assert "--include-cone-certificate-audit 1" in development
    assert "proceed_to_e18b" in confirmation
    assert "--evaluation-scope full" in confirmation
    assert "--include-cone-certificate-audit 1" in confirmation


def test_e18c_is_a_frozen_no_selection_ogb_audit() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e18c_ogb_cone_scale_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    parent = json.loads(
        (ROOT / "configs/fedbcert_e18_cone_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    used = set(parent["development_seeds"]) | set(
        parent["reserved_confirmation_seeds"]
    )
    assert used.isdisjoint(protocol["seeds"])
    assert protocol["dataset"] == "ogbn-arxiv"
    assert protocol["variants"]["dual_alpha025"]["alpha"] == 0.25
    assert set(protocol["variants"]) == {"feature_alpha0", "dual_alpha025"}
    assert protocol["selected_certificate"] == "cone_hybrid"
    assert protocol["expected_runner_units"] == (
        len(protocol["seeds"]) * len(protocol["variants"])
    )
    assert protocol["test_labels_used_for_selection"] is False


def test_e18c_runner_enforces_parent_confirmation_and_cone_audit() -> None:
    runner = (ROOT / "scripts/run_fedbcert_e18c_ogb_cone_scale.sh").read_text()
    assert "confirmation_complete_claim_supported" in runner
    assert 'selected" != "dual_alpha025"' in runner
    assert 'alpha" != "0.25"' in runner
    assert "--dataset ogbn-arxiv" in runner
    assert "--evaluation-scope full" in runner
    assert "--include-cone-certificate-audit 1" in runner
    assert "76 77 78" in runner
