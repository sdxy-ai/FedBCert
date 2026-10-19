"""Fast schema/integrity tests; no historical experiment is executed."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from audit_fedbcert_on_demand_inputs_20261015 import (
    CERTIFICATES, SPLITS, sha256, validate_bundle, validate_evidence,
    verify_hash, write_exclusive_json,
)


def fixture_bundle():
    masks, start = {}, 0
    for name, count in SPLITS.items():
        masks[name] = torch.zeros(6000, dtype=torch.bool)
        masks[name][start:start + count] = True
        start += count
    missing = torch.zeros(6000, dtype=torch.float32)
    missing[0] = missing[2000] = 1
    return {"features": torch.zeros((6000, 32)), "labels": torch.tensor([0] * 3000 + [1] * 3000),
            "membership": torch.arange(3).repeat_interleave(2000),
            "full_edge_index": torch.tensor([[0, 2000], [2000, 0]]), "missing_degree": missing,
            "masks": masks, "metadata": {"seed": 28101, "h": .9, "rho": .4,
                                          "nodes": 6000, "edges": 1, "split_counts": SPLITS}}


def fixture_evidence(data):
    nodes = data["masks"]["test_sealed"].nonzero().squeeze(1)
    logits = torch.tensor([[1., 0.]], dtype=torch.float64).repeat(2400, 1)
    return {"nodes": nodes, "labels": data["labels"][nodes], "nominal_logits": logits,
            "completion_logits": logits.clone(), "flip": torch.zeros(2400, dtype=torch.bool),
            "reliability": {"numeric_tolerance": 1e-6},
            "certificates": {name: torch.ones(2400, dtype=torch.bool) for name in CERTIFICATES},
            "slacks": {name: torch.ones(2400, dtype=torch.float64) for name in CERTIFICATES}}


class OnDemandInputAuditTests(unittest.TestCase):
    def test_valid_minimal_bundle(self):
        self.assertEqual(validate_bundle(fixture_bundle(), 28101)["nodes"], 6000)

    def test_ledger_tampering_is_rejected(self):
        data = fixture_bundle()
        data["missing_degree"][0] = 0
        with self.assertRaisesRegex(ValueError, "ledger"):
            validate_bundle(data, 28101)

    def test_overlapping_masks_are_rejected(self):
        data = fixture_bundle()
        data["masks"]["train"][0] = False
        data["masks"]["train"][2000] = True
        with self.assertRaisesRegex(ValueError, "splits overlap"):
            validate_bundle(data, 28101)

    def test_wrong_primary_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wrong fixed primary"):
            validate_bundle(fixture_bundle(), 28102)

    def test_evidence_acceptance_guard_and_flip(self):
        data = fixture_bundle()
        evidence = fixture_evidence(data)
        validate_evidence(data, evidence)
        bad = copy.deepcopy(evidence)
        bad["slacks"]["cone_hybrid"][0] = 1e-6
        with self.assertRaisesRegex(ValueError, "mask/guard"):
            validate_evidence(data, bad)
        bad = copy.deepcopy(evidence)
        bad["completion_logits"][0] = torch.tensor([0., 1.])
        bad["flip"][0] = True
        with self.assertRaisesRegex(ValueError, "certified flip"):
            validate_evidence(data, bad)

    def test_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            write_exclusive_json(path, {"ok": True})
            digest = sha256(path)
            self.assertEqual(verify_hash(path, digest), digest)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_hash(path, "0" * 64)

    def test_exclusive_output_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            write_exclusive_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_exclusive_json(path, {"ok": False})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
