from __future__ import annotations

import json
from pathlib import Path

from scripts.run_boundary_certified_external_baseline import build_parser
from scripts.run_fedbcert_e8_certificate_stress import run_trial
from scripts.select_fedbcert_e8_q1_hyperparameters import select


ROOT = Path(__file__).resolve().parents[1]


def test_e8_frozen_matrix_has_ninety_units() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e8_q1_protocol.json").read_text(encoding="utf-8")
    )
    count = (
        len(config["datasets"])
        * len(config["num_clients"])
        * len(config["partitions"])
        * len(config["confirmation_seeds"])
    )
    assert count == config["expected_confirmation_units"] == 90
    assert config["claims"]["test_labels_used_for_training_or_selection"] is False


def test_e8_native_adapter_exposes_all_five_baselines() -> None:
    for baseline in ("fedsage_plus", "feddep", "fedpub", "fedtad", "fedgta"):
        args = build_parser().parse_args(
            ["--baseline", baseline, "--dataset", "Photo", "--partition", "random", "--seed", "21"]
        )
        assert args.baseline == baseline
        assert args.dataset == "Photo"


def test_e8_tuning_selects_validation_only(tmp_path: Path) -> None:
    baselines = ("fedsage_plus", "feddep", "fedpub", "fedtad", "fedgta")
    datasets = ("Cora", "CiteSeer", "PubMed", "Computers", "Photo")
    for baseline in baselines:
        for dataset in datasets:
            for tag, val, test in (("low", 0.4, 0.99), ("high", 0.6, 0.01)):
                folder = tmp_path / f"{baseline}_{dataset}_random_c3_seed17_{tag}"
                folder.mkdir()
                (folder / "run_complete.json").write_text(
                    json.dumps({
                        "pooled_val_macro_f1": val,
                        "pooled_test_macro_f1": test,
                        "learning_rate": 0.001 if tag == "low" else 0.01,
                        "weight_decay": 0.0,
                        "test_labels_used_for_selection": False,
                    }),
                    encoding="utf-8",
                )
    chosen = select(tmp_path)
    assert chosen["test_labels_used_for_selection"] is False
    assert all(
        chosen["selected"][baseline][dataset]["pooled_val_macro_f1"] == 0.6
        for baseline in baselines
        for dataset in datasets
    )


def test_e8_exhaustive_stress_has_no_violation() -> None:
    row, _ = run_trial(seed=31, order_count=2, remote_count=3)
    assert row["completion_count"] == 216
    assert row["exact_logit_bound_violations"] == 0
    assert row["certified_soundness_violations"] == 0
    assert row["maximum_exact_to_certificate_ratio"] <= 1.0 + 1e-6
