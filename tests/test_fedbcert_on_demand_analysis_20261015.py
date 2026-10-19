"""Synthetic-record tests only: no historical targets or replay are loaded."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_fedbcert_on_demand_20261015 import (
    REPETITIONS, SEEDS, STRATEGIES, WORKLOADS, analyze_records, ideal_break_even,
    markdown_report, sha256, validate_records, verify_index, write_analysis,
)


def records_fixture():
    payloads = dict(zip(STRATEGIES, (10000, 9000, 8000, 7000, 6000)))
    times = dict(zip(STRATEGIES, (.10, .12, .14, .16, .17)))
    records = []
    for seed in SEEDS:
        for workload, n in WORKLOADS.items():
            for strategy in STRATEGIES:
                certified = 0 if strategy == "full_reference" else n // 2
                metadata = 0 if strategy == "full_reference" else 100
                total = payloads[strategy]
                for rep in REPETITIONS:
                    records.append({"seed": seed, "graph_id": f"h0.90_rho0.40_seed{seed}",
                                    "workload": workload, "strategy": strategy, "repetition": rep,
                                    "target_n": n, "certified_n": certified, "fallback_n": n - certified,
                                    "metadata_wire_bytes": metadata, "aggregate_request_bytes": 100,
                                    "aggregate_response_bytes": total - metadata - 100,
                                    "aggregate_wire_bytes": total - metadata, "total_wire_bytes": total,
                                    "remote_vectors": 100, "request_response_packets": 200,
                                    "cpu_seconds": times[strategy] * (1., .9, 1.1)[rep],
                                    "service_setup_seconds": .001, "nominal_seconds": .002,
                                    "metadata_seconds": .001, "certificate_seconds": .003, "fallback_seconds": .004,
                                    "class_mismatches": 0, "certified_class_mismatches": 0,
                                    "fallback_max_abs_error": 0., "nominal_vs_full_class_mismatches": 1})
    return records


class OnDemandAnalysisTests(unittest.TestCase):
    def test_complete_matrix_and_median_cpu(self):
        grouped = validate_records(records_fixture())
        self.assertEqual(len(grouped), 80)
        self.assertAlmostEqual(grouped[28101, "all_test", "cone"]["median_cpu_seconds"], .17)

    def test_primary_gate_and_seed_ratios(self):
        result, rows = analyze_records(records_fixture())
        self.assertTrue(result["engineering_feasibility_passed"])
        self.assertEqual(len(rows), 80)
        value = result["cone_comparisons"]["all_test"]["exposure_only"]
        self.assertAlmostEqual(value["relative_payload_reduction"]["mean"], 1 / 3)
        self.assertEqual(value["positive_payload_saving_seeds"], 8)
        self.assertGreater(value["cone_to_baseline_cpu_ratio"]["mean"], 1)

    def test_missing_or_duplicate_matrix_is_rejected(self):
        rows = records_fixture()
        with self.assertRaisesRegex(ValueError, "240"):
            validate_records(rows[:-1])
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records(rows)

    def test_changed_rep_wire_is_rejected(self):
        rows = records_fixture()
        rows[1]["aggregate_response_bytes"] += 1
        rows[1]["aggregate_wire_bytes"] += 1
        rows[1]["total_wire_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "repetition changed"):
            validate_records(rows)

    def test_class_or_logit_mismatch_is_hard_failure(self):
        rows = records_fixture()
        rows[0]["class_mismatches"] = 1
        with self.assertRaisesRegex(ValueError, "class mismatch"):
            validate_records(rows)
        rows = records_fixture()
        rows[0]["fallback_max_abs_error"] = 1.01e-10
        with self.assertRaisesRegex(ValueError, "logit error"):
            validate_records(rows)

    def test_warmup_is_rejected(self):
        rows = records_fixture()
        rows[0]["warmup"] = True
        with self.assertRaisesRegex(ValueError, "warmup"):
            validate_records(rows)

    def test_negative_gate_does_not_replace_seeds(self):
        rows = records_fixture()
        for row in rows:
            if row["strategy"] == "cone" and row["seed"] in SEEDS[:3]:
                extra = 3800
                row["aggregate_response_bytes"] += extra
                row["aggregate_wire_bytes"] += extra
                row["total_wire_bytes"] += extra
        result, _ = analyze_records(rows)
        self.assertFalse(result["engineering_feasibility_passed"])
        self.assertFalse(result["engineering_gates"]["cone_vs_exposure_positive_in_at_least_6_of_8"])
        self.assertIn("停止", markdown_report(result))

    def test_bootstrap_is_deterministic(self):
        a, _ = analyze_records(records_fixture())
        b, _ = analyze_records(records_fixture())
        self.assertEqual(a, b)

    def test_break_even_direction_and_missing_cases(self):
        value = ideal_break_even(1000, 500, .1, .2)
        self.assertAlmostEqual(value["bandwidth_bps"], 40000)
        self.assertIn("below", value["regime"])
        self.assertIsNone(ideal_break_even(500, 1000, .1, .2)["bandwidth_bps"])
        self.assertIsNone(ideal_break_even(1000, 500, .2, .1)["bandwidth_bps"])

    def test_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.txt"
            with path.open("x") as handle:
                handle.write("fixed")
            verify_index(root, {"record.txt": sha256(path)})
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_index(root, {"record.txt": "0" * 64})

    def test_outputs_are_exclusive(self):
        result, rows = analyze_records(records_fixture())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            write_analysis(output, result, rows)
            self.assertEqual({p.name for p in output.iterdir()},
                             {"summary.json", "per_seed.csv", "workload_summary.csv", "RESULTS_ZH.md"})
            with self.assertRaises(FileExistsError):
                write_analysis(output, result, rows)


if __name__ == "__main__":
    unittest.main()
