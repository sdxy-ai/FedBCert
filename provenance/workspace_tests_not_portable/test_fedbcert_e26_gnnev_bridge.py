from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fedbcert_e26_gnnev_bridge.py"
COMPAT = (
    ROOT
    / "results_boundary_certified"
    / "boundary_certified_spectral_e26_known_universe_verifier"
    / "official_bridges"
    / "gurobi12_compat"
)
GNNEV = COMPAT.parent / "gnnev"
for entry in (ROOT, GNNEV, COMPAT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
spec = importlib.util.spec_from_file_location("fedbcert_e26_bridge", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_common_predictor_is_exact() -> None:
    generator = torch.Generator().manual_seed(2601)
    features = module.positive_unit_rows(generator, 7, 3)
    official, polynomial, _, _ = module.build_models(generator, 3, 3)
    edges = module.directed_base_edges([[0, 1], [1, 2], [2, 3], [3, 0]], 7)
    official_logits = official(module.Data(x=features, edge_index=module.edge_tensor(edges)))
    local_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]).t()
    shift, _ = module.row_normalized_shift(
        local_edges, 4, device=torch.device("cpu"), dtype=torch.float64
    )
    polynomial_logits = polynomial(features[:4], shift)
    assert torch.allclose(official_logits[:4], polynomial_logits, atol=1e-10, rtol=0.0)


def test_known_set_is_strictly_more_informative() -> None:
    generator = torch.Generator().manual_seed(7)
    fixed = module.positive_unit_rows(generator, 3, 3)
    assert torch.all(fixed > 0)
    alternative = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert not any(torch.equal(alternative, row) for row in fixed)


if __name__ == "__main__":
    test_common_predictor_is_exact()
    test_known_set_is_strictly_more_informative()
    print("2 E26 tests passed")
