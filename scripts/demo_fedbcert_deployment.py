#!/usr/bin/env python3
"""Deterministic interface illustration; no training or research evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openfgl.flcore.fedbcert.certificate import PolynomialSpectralClassifier
from openfgl.flcore.fedbcert.deployment import BoundaryLedgerReceipt, FrozenLocalDeployment


def illustrative_inputs():
    # Values are chosen solely to illustrate interface states, not to estimate
    # coverage, validate a risk task, or establish graph utility.
    with torch.random.fork_rng():
        torch.manual_seed(20261004)
        model = PolynomialSpectralClassifier(
            2, 2, 2, 3, 0.0, certificate_margin_mode="pairwise_difference"
        ).double().eval()
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.encoder.bias.zero_()
        model.theta.copy_(torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64))
        model.classifier.weight.copy_(torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64))
        model.classifier.bias.zero_()
    features = torch.tensor([[1., 0.], [1., 0.], [1., 0.8]], dtype=torch.float64)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    return model, features, edges


def build_demo() -> dict:
    model, features, edges = illustrative_inputs()
    snapshot = FrozenLocalDeployment(
        model, features, edges, view_version="illustration-local-v1",
        preprocessing_version="identity-preprocessing-v1",
        node_ids=("local-strong", "boundary-strong", "boundary-weak"),
    )
    receipt = BoundaryLedgerReceipt(
        bounds=torch.tensor([0, 1, 3]), ledger_version="illustration-ledger-v1",
        source="illustrative_external_authority_not_a_real_ledger",
        view_sha256=snapshot.view_sha256,
        preprocessing_version=snapshot.preprocessing_version,
        upper_bound_attested=True,
    )
    cases = {
        "valid_upper_bound_receipt": snapshot.certify(receipt),
        "missing_ledger": snapshot.certify(),
        "unattested_ledger": snapshot.certify(replace(receipt, upper_bound_attested=False)),
        "stale_view_receipt": snapshot.certify(replace(receipt, view_sha256="stale-view")),
        "stale_preprocessing_receipt": snapshot.certify(replace(receipt, preprocessing_version="old-v0")),
        "invalid_negative_count": snapshot.certify(replace(receipt, bounds=torch.tensor([0, -1, 3]))),
        "explicit_universal_without_ledger": snapshot.certify(mode="universal"),
    }
    for case in cases.values():
        if case["nominal_logits"] != cases["valid_upper_bound_receipt"]["nominal_logits"]:
            raise AssertionError("a certificate mode changed the cached nominal output")
    return {
        "purpose": "engineering_interface_illustration_not_an_application_benchmark",
        "does_not_train_or_use_labels_or_remote_inputs": True,
        "frozen_output_identical_across_cases": True,
        "class_names_for_illustration_only": {"0": "待核查", "1": "其他"},
        "cases": cases,
        "state_counts": {
            name: {state: sum(row["state"] == state for row in case["records"])
                   for state in ("Certified", "Deferred", "Unavailable")}
            for name, case in cases.items()
        },
    }


def write_new_report(path: Path, report: dict) -> None:
    """Exclusive creation; no --force and no replacement of old reports."""
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(rendered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new JSON file; an existing path is never overwritten")
    args = parser.parse_args(argv)
    report = build_demo()
    if args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        write_new_report(args.output, report)
        print(json.dumps({
            "output": str(args.output), "state_counts": report["state_counts"],
            "frozen_output_identical_across_cases": report["frozen_output_identical_across_cases"],
        }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
