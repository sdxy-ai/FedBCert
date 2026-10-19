import json
from pathlib import Path

import torch

from openfgl.flcore.fedbcert import boundary_uncertainty
from scripts.run_fedbcert_e27_ledger_granularity import represented_count


ROOT = Path(__file__).resolve().parents[1]


def test_e27_protocol_is_frozen_and_selection_free() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e27_ledger_granularity_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["information_modes"] == [
        "exact",
        "bin2",
        "bin4",
        "bin8",
        "presence",
        "client_max",
        "universal",
    ]
    assert protocol["claims"]["no_training"] is True
    assert protocol["claims"]["no_model_checkpoint_or_certificate_selection"] is True
    assert protocol["claims"]["test_metrics_are_evaluation_only"] is True
    assert len(protocol["datasets"]) * len(protocol["partitions"]) * len(
        protocol["seeds"]
    ) == 30


def test_nested_safe_ledgers_never_underestimate_and_weaken_monotonically() -> None:
    exact = torch.tensor([0.0, 1.0, 2.0, 5.0, 8.0])
    retained_degree = torch.tensor([1.0, 2.0, 4.0, 3.0, 7.0])

    counts = []
    bits = []
    for mode in ("exact", "bin2", "bin4", "bin8"):
        represented, direct_budget, storage_bits = represented_count(
            mode, exact, float(exact.max().item())
        )
        assert direct_budget is None
        assert represented is not None
        assert torch.all(represented >= exact)
        counts.append(represented)
        bits.append(storage_bits)

    for finer, coarser in zip(counts, counts[1:]):
        assert torch.all(coarser >= finer)
        finer_budget = boundary_uncertainty(retained_degree, finer)
        coarser_budget = boundary_uncertainty(retained_degree, coarser)
        assert torch.all(coarser_budget >= finer_budget)
    assert bits == sorted(bits, reverse=True)

    _, presence_budget, presence_bits = represented_count(
        "presence", exact, float(exact.max().item())
    )
    _, universal_budget, universal_bits = represented_count(
        "universal", exact, float(exact.max().item())
    )
    client_max, _, client_bits = represented_count(
        "client_max", exact, float(exact.max().item())
    )
    assert client_max is not None and torch.all(client_max >= exact)
    assert torch.all(universal_budget >= presence_budget)
    assert presence_bits == exact.numel()
    assert universal_bits == 0
    assert client_bits > 0
