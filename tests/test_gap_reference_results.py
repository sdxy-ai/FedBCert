"""Compact-record regression tests; no model execution, timing or training."""
import copy
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gap_check", ROOT / "scripts/verify_gap_reference_results.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture
def records():
    ref = ROOT / "reference_results/gaps"
    return (checker.read(ref / "g1/gate_audit.json"),
            checker.read(ref / "g2/execution_manifest.json"),
            checker.read(ref / "g2/summary.json"),
            {checker.read(p)["case"]: checker.read(p) for p in (ref / "g2").glob("*_timing.json")})


def test_valid_records(records):
    assert checker.verify_records(*records)["formal_measurements"] == 1200


def test_missing_case_rejected(records):
    records[3].pop(next(iter(records[3])))
    with pytest.raises(ValueError, match="case inventory"):
        checker.verify_records(*records)


def test_unavailable_is_not_zero_precision(records):
    records[0]["precision"] = 0
    with pytest.raises(ValueError, match="numeric performance"):
        checker.verify_records(*records)


@pytest.mark.parametrize("stage", range(5))
def test_changed_statistic_rejected(records, stage):
    row = records[2]["results"][0]
    row["stages"][stage]["median_seconds"] += .001
    records[3][row["case"]] = copy.deepcopy(row)
    with pytest.raises(ValueError, match="statistic mismatch"):
        checker.verify_records(*records)


def test_failed_semantics_rejected(records):
    row = records[2]["results"][0]
    row["stages"][0]["checks"]["input_model_unchanged"] = False
    records[3][row["case"]] = copy.deepcopy(row)
    with pytest.raises(ValueError, match="semantic-check"):
        checker.verify_records(*records)
