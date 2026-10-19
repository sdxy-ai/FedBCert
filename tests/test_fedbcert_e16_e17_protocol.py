from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from openfgl.flcore.fedsla_xn.data import _load_official_ogb_pyg_dataset


ROOT = Path(__file__).resolve().parents[1]


def load_e17_module():
    path = ROOT / "scripts" / "run_fedbcert_e17_exact_completion_oracle.py"
    specification = importlib.util.spec_from_file_location("fedbcert_e17", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_e16_development_and_confirmation_seeds_are_disjoint() -> None:
    config = json.loads(
        (ROOT / "configs" / "fedbcert_e16a_deployment_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    development = set(config["development_seeds"])
    confirmation = set(config["reserved_confirmation_seeds"])
    assert development.isdisjoint(confirmation)
    assert confirmation == set(config["confirmation_matrix"]["seeds"])
    assert config["expected_runner_units"] == (
        len(config["datasets"])
        * len(config["development_seeds"])
        * len(config["variants"])
    )
    matrix = config["confirmation_matrix"]
    assert matrix["expected_runner_units"] == (
        len(config["datasets"])
        * len(matrix["num_clients"])
        * len(matrix["partitions"])
        * len(matrix["seeds"])
    )


def test_e16_policy_has_a_feasible_graph_invariant_fallback() -> None:
    config = json.loads(
        (ROOT / "configs" / "fedbcert_e16a_deployment_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    fallback = config["variants"]["feature_alpha0"]
    assert fallback["encoder_type"] == "feature_only_mlp"
    assert fallback["alpha"] == 0.0
    assert "dataset-specific alpha routing" in config["prohibited_actions"]
    assert "reselecting alpha after confirmation" in config["prohibited_actions"]


def test_official_ogb_loader_scopes_pytorch26_compatibility_override() -> None:
    calls: list[dict] = []
    original_load = torch.load

    def sentinel_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return "processed"

    class FakeOfficialDataset:
        def __init__(self, *, name: str, root: str):
            self.name = name
            self.root = root
            self.payload = torch.load("processed.pt")

    try:
        torch.load = sentinel_load
        dataset = _load_official_ogb_pyg_dataset(
            FakeOfficialDataset,
            name="ogbn-arxiv",
            root="trusted-official-root",
        )
    finally:
        torch.load = original_load

    assert dataset.payload == "processed"
    assert calls == [{"weights_only": False}]
    assert torch.load is original_load


def test_e17_frozen_completion_count_is_exact() -> None:
    module = load_e17_module()
    completions = list(module.completion_edge_indices([1, 2, 1, 0], 3))
    assert len(completions) == 216
    assert all(edge_index.shape[0] == 2 for edge_index in completions)


def test_e17_one_trial_has_no_oracle_violation() -> None:
    module = load_e17_module()
    rows, trial = module.run_trial(
        seed=61,
        order_count=3,
        boundary_counts=[1, 2, 1, 0],
        remote_count=3,
        tolerance=1e-10,
    )
    assert trial["completion_count"] == 216
    assert len(rows) == 4
    assert sum(int(row["soundness_violations"]) for row in rows) == 0
    assert sum(int(row["pairwise_bound_violations"]) for row in rows) == 0
