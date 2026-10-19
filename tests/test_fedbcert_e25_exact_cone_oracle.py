from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fedbcert_e25_exact_cone_oracle import remote_scalar_interval


def test_remote_scalar_interval_matches_positive_ball_support() -> None:
    coefficients = torch.tensor([2.0, -0.5], dtype=torch.float64)
    direction = torch.tensor([3.0, -4.0], dtype=torch.float64)
    lower, upper = remote_scalar_interval(coefficients, direction)
    # support(direction)=[-4,3]; multiplying by the negative coefficient swaps endpoints.
    assert torch.allclose(lower, torch.tensor(-9.5, dtype=torch.float64))
    assert torch.allclose(upper, torch.tensor(8.0, dtype=torch.float64))


def test_e25_protocol_is_no_selection_exact_audit() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "fedbcert_e25_exact_cone_oracle_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["status"] == "frozen_before_execution"
    assert protocol["selection_or_training"] is False
    assert protocol["test_labels_used"] is False
    assert protocol["count_semantics"] == "exact"
    assert protocol["completion_family"]["expected_total_graph_completions"] == 8640
    assert protocol["certificate_variants"] == [
        "retained_symmetric",
        "cone_raw",
        "cone_hybrid",
    ]


if __name__ == "__main__":
    test_remote_scalar_interval_matches_positive_ball_support()
    test_e25_protocol_is_no_selection_exact_audit()
    print("2 E25 tests passed")
