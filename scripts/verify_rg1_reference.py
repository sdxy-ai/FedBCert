#!/usr/bin/env python3
"""Recompute the frozen RG-1.0 statistics from saved measurements only."""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fedbcert_rg1_analysis import analyze

PINS = {
    "all_records.json": "d461beaa360c16b20a8ff72fe0a3722114bbf4d006a64ce60bfa857f01b3e2c6",
    "case_metrics.json": "2375b48eec4f513267dc1129d5dc8ec9d51575978d19b0e4c7a6eddeaf1def70",
    "analysis.json": "83bbf3fa50623fa844b313989d0f08c798c06e69a30879fc9232c386bc02a34f",
}


def compare(actual, expected, path="analysis"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"key mismatch: {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"length mismatch: {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}/{i}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"numeric mismatch: {path}")
    elif actual != expected:
        raise ValueError(f"value mismatch: {path}")


def verify(root=ROOT):
    data = Path(root) / "rg1_data"
    loaded = {}
    for name, expected in PINS.items():
        raw = (data / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"saved record checksum mismatch: {name}")
        loaded[name] = json.loads(raw)
    result = analyze(loaded["all_records.json"], loaded["case_metrics.json"])
    compare(result, loaded["analysis.json"])
    if result["case_count"] != 32 or result["record_count"] != 1536 or len(result["measurements"]) != 512:
        raise ValueError("incomplete formal matrix")
    print("PASS: 32 cases, 1,536 formal measurements, 512 medians and all frozen descriptive statistics.")
    print("Saved-data analysis only: no model training, inference or new measurement.")
    return result


if __name__ == "__main__":
    verify()
