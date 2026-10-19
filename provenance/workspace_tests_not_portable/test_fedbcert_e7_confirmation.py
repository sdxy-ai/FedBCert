from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_boundary_certified_spectral_e7_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("fedbcert_e7_confirmation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_expected_units_builds_full_cartesian_product() -> None:
    units = MODULE.expected_units(
        ["Cora", "CiteSeer"], ["metis", "random"], [11, 12, 13]
    )
    assert len(units) == 12
    assert ("Cora", "metis", 11) in units
    assert ("CiteSeer", "random", 13) in units


def test_summary_reports_sample_standard_deviation() -> None:
    result = MODULE.summarize([0.7, 0.8, 0.9])
    assert abs(result["mean"] - 0.8) < 1e-12
    assert abs(result["sample_standard_deviation"] - 0.1) < 1e-12
    low, high = result["bootstrap_95_mean_interval"]
    assert low <= result["mean"] <= high


def test_external_method_order_is_frozen() -> None:
    assert MODULE.INTERNAL_METHODS == ("fedavg", "fedbcert_ba")
    assert MODULE.EXTERNAL_METHODS == ("fedsage_plus", "feddep")
