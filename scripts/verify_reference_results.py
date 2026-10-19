#!/usr/bin/env python3
"""Check the compact reference outputs against the headline paper values."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_results"


def load_json(relative: str) -> dict:
    return json.loads((REFERENCE / relative).read_text(encoding="utf-8"))


def load_rows(relative: str) -> list[dict[str, str]]:
    with (REFERENCE / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(value: float, target: float, tolerance: float = 1e-4) -> None:
    if abs(value - target) > tolerance:
        raise AssertionError(f"{value} differs from expected {target}")


def verify_checksums() -> None:
    manifest = REFERENCE / "SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def main() -> None:
    verify_checksums()
    main_confirmation = load_json("main_confirmation/confirmation_summary.json")
    assert main_confirmation["complete"] is True
    assert main_confirmation["certificate_violations"] == 0
    assert main_confirmation["selected_alpha"] == 0.25

    main_row = load_rows("main_confirmation/paper_table_e18b_overall.csv")[0]
    close(float(main_row["cone_hybrid_certified_ratio"]), 0.9131)
    close(float(main_row["hybrid_minus_symmetric_certified_ratio"]), 0.1128)

    exact = load_json("exact_worst_case/analysis_summary.json")
    assert exact["complete"] is True
    assert exact["soundness_violations"] == 0
    close(float(exact["cone_hybrid_recall"]), 0.8205)
    close(float(exact["cone_hybrid_precision"]), 1.0)

    ledger_rows = load_rows(
        "ledger_granularity/paper_table_e27_ledger_granularity_overall.csv"
    )
    ledger = {row["certificate_information"]: row for row in ledger_rows}
    close(float(ledger["exact"]["mean_certified_ratio"]), 0.8501)
    close(
        float(ledger["exact"]["mean_certified_ratio"])
        - float(ledger["presence"]["mean_certified_ratio"]),
        0.1851,
    )

    cover_rows = load_rows("safe_cover/paper_table_e24_overall.csv")
    cover = {row["method"]: row for row in cover_rows}
    close(float(cover["observed_graph_cover"]["test_certified_ratio"]), 0.8388)

    ogb = load_json("ogbn_arxiv/analysis_summary.json")
    assert ogb["complete"] is True
    assert ogb["node_count"] == 169343
    assert ogb["certificate_violations"] == 0

    gnnev = load_json("gnnev_bridge/bridge_summary.json")
    assert gnnev["complete"] is True
    assert gnnev["gnnev_enumeration_agreement_count"] == 40
    assert gnnev["gnnev_unknown_results"] == 0

    print("Reference-result checks passed.")


if __name__ == "__main__":
    main()
