import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fedbcert_on_demand_20261015 import (
    PROTOCOL, check_result, run_strategy, workload_targets,
)
from openfgl.flcore.fedbcert.certificate import PolynomialSpectralClassifier, row_normalized_shift


def test_protocol_is_bounded_and_honest():
    spec = json.loads(PROTOCOL.read_text())
    assert spec["seeds"] == list(range(28101, 28109))
    assert spec["cell"]["h"] == .9 and spec["cell"]["rho"] == .4
    assert [work["count"] for work in spec["workloads"]] == [2400, 32]
    assert spec["timing"]["repetitions"] == 3
    assert "NOT a new independent confirmation" in spec["scope"]
    assert spec["numerics"]["guard"] == 1e-6


def test_target_sample_is_fixed_without_label_or_prediction_input():
    nodes = torch.arange(2400)
    a = workload_targets(nodes, 32, 28101)
    assert torch.equal(a, workload_targets(nodes, 32, 28101))
    assert not torch.equal(a, workload_targets(nodes, 32, 28102))
    assert a.numel() == a.unique().numel() == 32
    assert torch.equal(a, a.sort().values)
    assert torch.equal(workload_targets(nodes, 2400, 28101), nodes)
    with pytest.raises(ValueError):
        workload_targets(nodes, 0, 28101)


@pytest.mark.parametrize("strategy", ["full_reference", "exposure_only", "ball_pairwise", "nonnegative_diameter", "cone"])
def test_routing_matches_fullgraph_on_toy(strategy):
    torch.set_num_threads(1)
    torch.manual_seed(45)
    model = PolynomialSpectralClassifier(3, 5, 2, 3, 0.2,
        encoder_type="dual_feature_spectral", encoder_depth=2,
        spectral_fusion_scale=.25, certificate_margin_mode="pairwise_difference").double().eval()
    features = torch.randn(6, 3, dtype=torch.float64)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    membership = torch.tensor([0, 0, 1, 1, 2, 2])
    allowed = dict(features=features, full_edge_index=edges, membership=membership)
    targets = torch.tensor([0, 2, 5])
    shift, _ = row_normalized_shift(edges, 6, device=torch.device("cpu"), dtype=torch.float64)
    with torch.no_grad():
        reference = model(features, shift)
    row, evidence = run_strategy(model, allowed, targets, strategy, 1e-6)
    # The reference nominal is only needed for descriptive disagreement.
    check_result(row, evidence, reference, reference, targets,
                 {"numerics": {"reference_logit_atol": 1e-10}})
    assert row["class_mismatches"] == 0
    assert row["fallback_max_abs_error"] < 1e-10
    assert row["certified_n"] + row["fallback_n"] == len(targets)
    assert row["total_wire_bytes"] == row["metadata_wire_bytes"] + row["aggregate_wire_bytes"]
    if strategy == "full_reference":
        assert row["metadata_wire_bytes"] == 0
        assert row["nominal_seconds"] == row["certificate_seconds"] == 0


def test_class_flip_is_never_excused_by_logit_tolerance():
    reference = torch.tensor([[1e-12, 0.]], dtype=torch.float64)
    result = dict(final_logits=torch.tensor([[0., 1e-12]], dtype=torch.float64),
        certified=torch.tensor([False]), fallback_nodes=torch.tensor([0]),
        fallback_logits=torch.tensor([[0., 1e-12]], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="mismatch"):
        check_result({}, result, reference, reference, torch.tensor([0]),
                     {"numerics": {"reference_logit_atol": 1e-10}})
