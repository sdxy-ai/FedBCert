"""Checks for the public saved-record entry point, not a graph experiment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_saved_on_demand_records():
    from verify_on_demand_reference import verify
    report = verify()
    comparison = report["cone_comparisons"]["all_test"]["full_reference"]
    assert round(100 * comparison["relative_payload_reduction"]["mean"], 2) == 22.09
    assert round(100 * (comparison["cone_to_baseline_cpu_ratio"]["mean"] - 1), 1) == 23.8
