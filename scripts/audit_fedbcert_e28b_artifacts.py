#!/usr/bin/env python
"""Post-run artifact integrity audit; cannot change E28b scientific decisions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from fedbcert_e28b_common import PROTOCOL, sha256, tensor_digest, write_json
from run_fedbcert_e28b_controlled import verify_manifest


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / protocol["artifacts"]["output_root"]
    manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    verify_manifest(manifest)
    summary_path = output / "analysis/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for relative, expected in summary["input_sha256"].items():
        assert sha256(ROOT / relative) == expected
    graph_reports = []
    for graph_id in sorted(set(unit["graph_id"] for unit in manifest["matrix"])):
        folder = output / "development" / graph_id
        graph_manifest = json.loads((folder / "graph_manifest.json").read_text(encoding="utf-8"))
        assert sha256(folder / "sealed_graph_bundle.pt") == graph_manifest["bundle_sha256"]
        data = torch.load(folder / "sealed_graph_bundle.pt", map_location="cpu", weights_only=False)
        masks = data["masks"]
        assert (torch.stack(list(masks.values())).sum(0) == 1).all()
        edges, member = data["full_edge_index"], data["membership"]
        cut = member[edges[0]] != member[edges[1]]
        ledger = torch.bincount(edges[0, cut], minlength=len(member)).float()
        assert torch.equal(ledger, data["missing_degree"])
        initial_digest, reference_nodes, reference_groups = None, None, None
        for name in ("observed_p", "identity", "strong_mlp"):
            model_folder = folder / name
            record = json.loads((model_folder / "validation_metrics.json").read_text(encoding="utf-8"))
            evidence = torch.load(model_folder / "validation_evidence.pt", map_location="cpu", weights_only=False)
            initial = torch.load(model_folder / "initial_state.pt", map_location="cpu", weights_only=True)
            selected = torch.load(model_folder / "selected_state.pt", map_location="cpu", weights_only=True)
            assert tensor_digest(initial) == record["initial_tensor_sha256"]
            assert tensor_digest(selected) == record["selected_tensor_sha256"]
            trajectory = json.loads((model_folder / "validation_trajectory.json").read_text(encoding="utf-8"))
            assert len(trajectory) == protocol["training"]["rounds"]
            expected_round = max(trajectory, key=lambda row: row["validation_macro_f1"])["round"]
            assert record["selected_round"] == expected_round  # max preserves earliest tie.
            nodes = evidence["nodes"]
            assert len(nodes) == len(set(nodes.tolist())) == int(masks["validation"].sum())
            assert masks["validation"][nodes].all()
            assert not masks["test_sealed"][nodes].any() and not masks["audit_unused"][nodes].any()
            assert torch.equal(evidence["labels"], data["labels"][nodes])
            assert torch.equal(evidence["flip"], evidence["nominal_logits"].argmax(1) != evidence["completion_logits"].argmax(1))
            if name == "observed_p":
                initial_digest, reference_nodes, reference_groups = record["initial_tensor_sha256"], nodes, evidence["groups"]
                certs = evidence["certificates"]
                assert torch.equal(certs["cone_hybrid"], certs["cone_only"] | certs["retained_pairwise"])
            else:
                assert torch.equal(nodes, reference_nodes)
                assert all(torch.equal(evidence["groups"][k], reference_groups[k]) for k in reference_groups)
                assert evidence["certificates"]["structural_invariance"].all()
                assert not evidence["flip"].any()
                if name == "identity":
                    assert record["initial_tensor_sha256"] == initial_digest
            correct = evidence["nominal_logits"].argmax(1) == evidence["labels"]
            for row in record["rows"]:
                group = evidence["groups"][row["group"]]
                certified = evidence["certificates"][row["certificate"]]
                assert row["n"] == int(group.sum())
                assert row["certified_n"] == int((group & certified).sum())
                assert row["certified_correct_n"] == int((group & certified & correct).sum())
                assert row["certified_flip_n"] == int((group & certified & evidence["flip"]).sum()) == 0
            assert record["reliability"]["bound_violations"] == 0
        graph_reports.append({key: graph_manifest[key] for key in (
            "seed", "h", "rho", "nodes", "edges", "mean_degree", "actual_homophily", "actual_cut_ratio", "split_counts")})
    assert len(graph_reports) == 8
    assert set(g["seed"] for g in graph_reports) == set(protocol["seeds"]["development"])
    verify_manifest(manifest)
    report = dict(experiment="E28b", phase="post_execution_artifact_integrity", graph_units=8,
                  training_runs=24, validation_nodes_per_graph=1200, all_checks_passed=True,
                  test_evaluated=False, confirmation_started=False,
                  summary_sha256=sha256(summary_path), graphs=graph_reports,
                  checks=["frozen/protected file hashes", "bundle and initial/selected state hashes",
                          "exact generated boundary ledger", "disjoint stratified masks",
                          "earliest best validation checkpoint", "validation-only evidence",
                          "P/I identical initialization", "shared observed-P exposure groups",
                          "cone/symmetric union and zero certified flips", "raw group counts reproduce metrics"])
    write_json(output / "analysis/artifact_integrity_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "graphs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
