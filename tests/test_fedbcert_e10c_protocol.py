from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e10c_matrix_and_frozen_route() -> None:
    protocol = json.loads(
        (ROOT / "configs/fedbcert_e10c_protocol.json").read_text(encoding="utf-8")
    )
    units = (
        len(protocol["datasets"])
        * len(protocol["partitions"])
        * len(protocol["num_clients"])
        * len(protocol["seeds"])
    )
    assert units == protocol["expected_units"] == 30
    assert protocol["frozen_variant_by_dataset"] == {
        "Cora": "e8_linear_h64_d1",
        "CiteSeer": "e8_linear_h64_d1",
        "PubMed": "e8_linear_h64_d1",
        "Computers": "dual_h64_d2_s05_sep025",
        "Photo": "dual_h64_d2_s05_sep025",
    }
    assert protocol["claims"]["bundle_sha256_must_match_manifest"] is True
    assert protocol["claims"]["e10c_results_cannot_trigger_method_retuning"] is True
