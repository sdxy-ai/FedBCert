from __future__ import annotations

import json
from pathlib import Path

import torch

from openfgl.flcore.fedbcert.federated import (
    _identity_shift_like,
    _zero_uncertainty_basis_like,
)
from scripts.run_fedbcert_e22_teacher_shield import VARIANTS
from scripts.run_fedbcert_e24_invariant_cover import COMPONENTS, COVERS


ROOT = Path(__file__).resolve().parents[1]


def test_identity_propagation_has_exact_zero_completion_radius() -> None:
    indices = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]])
    values = torch.ones(indices.size(1))
    reference = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
    identity = _identity_shift_like(reference)
    assert torch.equal(identity.to_dense(), torch.eye(3))
    basis = [torch.arange(3, dtype=torch.float32), torch.ones(3), torch.full((3,), 2.0)]
    zero = _zero_uncertainty_basis_like(basis)
    assert len(zero) == len(basis)
    assert all(torch.count_nonzero(value).item() == 0 for value in zero)


def test_e24_is_frozen_no_selection_control() -> None:
    config = json.loads(
        (ROOT / "configs/fedbcert_e24_invariant_cover_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["status"] == "frozen_before_execution"
    assert config["selection_or_reselection"] is False
    assert config["confirmation_seeds"] == [128, 129, 130]
    assert config["expected_teacher_units"] == 24
    assert config["expected_component_training_units"] == 96
    assert config["expected_cover_evaluation_units"] == 24
    assert "spectral_identity_cad" in VARIANTS
    assert COVERS["identity_invariant_cover"] == (
        "feature_kd_cad",
        "spectral_identity_cad",
    )
    assert COVERS["observed_graph_cover"] == (
        "feature_kd_cad",
        "spectral_cad",
    )
    assert COMPONENTS["feature_ce"][2] == "cone_hybrid"
    assert COMPONENTS["feature_kd_cad"][2] == "cone_hybrid"
    assert COMPONENTS["spectral_identity_cad"][2] == "symmetric"
