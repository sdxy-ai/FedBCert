#!/usr/bin/env python
"""Recompute raw confirmation evidence and checkpoint integrity, without selection."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import torch
from fedbcert_e28b_common import macro_f1, resource_usage, sha256, tensor_digest, write_json
from fedbcert_e28b_confirmation_common import check_index, load_protocol, require_test_unlock
from run_fedbcert_e28b_confirmation import verified_manifest


def audit_raw_rows(record, evidence):
    labels, prediction = evidence["labels"], evidence["nominal_logits"].argmax(1)
    correct = labels == prediction
    if not torch.equal(evidence["flip"], prediction != evidence["completion_logits"].argmax(1)):
        raise RuntimeError("flip flags disagree with logits")
    soft_change = (evidence["nominal_logits"].softmax(1) - evidence["completion_logits"].softmax(1)).abs().max(1).values
    if not torch.equal(soft_change, evidence["softmax_change"]):
        raise RuntimeError("softmax diagnostic mismatch")
    for row in record["rows"]:
        group, cert = evidence["groups"][row["group"]], evidence["certificates"][row["certificate"]]
        counts = dict(n=int(group.sum()), correct_n=int((group & correct).sum()),
                      certified_n=int((group & cert).sum()), certified_correct_n=int((group & cert & correct).sum()),
                      certified_flip_n=int((group & cert & evidence["flip"]).sum()),
                      category_flip_n=int((group & evidence["flip"]).sum()),
                      softmax_change_ge_005_n=int((group & (soft_change >= .05)).sum()))
        if any(row[key] != value for key, value in counts.items()) or counts["certified_flip_n"]:
            raise RuntimeError("raw subgroup counts disagree or certified flip")
        expected = macro_f1(labels[group].numpy(), prediction[group].numpy())
        if (expected is None) != (row["macro_f1"] is None) or (expected is not None and abs(expected - row["macro_f1"]) > 1e-12):
            raise RuntimeError("raw subgroup Macro-F1 disagreement")
        for key, count in (("certified_ratio", counts["certified_n"]), ("certified_accuracy", counts["certified_correct_n"]),
                           ("category_flip_fraction", counts["category_flip_n"]), ("softmax_change_fraction", counts["softmax_change_ge_005_n"])):
            expected = count / counts["n"] if counts["n"] else None
            if (expected is None) != (row[key] is None) or (expected is not None and abs(expected - row[key]) > 1e-12):
                raise RuntimeError("raw subgroup ratio disagreement")


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    spec, base = load_protocol()
    output = ROOT / spec["artifacts"]["output_root"]
    manifest = verified_manifest(spec, output)
    commit = require_test_unlock(output, spec)
    completion = json.loads((output / "test_evaluation/completion.json").read_text(encoding="utf-8"))
    if completion["completed_test_runs"] != 288 or completion["checkpoint_commit_sha256"] != sha256(output / "checkpoint_commit.json"):
        raise RuntimeError("test matrix not complete")
    check_index(output, completion["records_sha256"])
    reports, test_hashes = [], {}
    for graph_id in sorted(set(unit["graph_id"] for unit in manifest["matrix"])):
        folder = output / "training" / graph_id
        metadata = json.loads((folder / "graph_manifest.json").read_text(encoding="utf-8"))
        if sha256(folder / "sealed_graph_bundle.pt") != metadata["bundle_sha256"]:
            raise RuntimeError("graph bundle mismatch")
        data = torch.load(folder / "sealed_graph_bundle.pt", map_location="cpu", weights_only=False)
        member, edges, masks = data["membership"], data["full_edge_index"], data["masks"]
        if not bool((torch.stack(list(masks.values())).sum(0) == 1).all()):
            raise RuntimeError("overlapping/missing masks")
        cut = member[edges[0]] != member[edges[1]]
        expected = torch.bincount(edges[0, cut], minlength=len(member)).float()
        if not torch.equal(expected, data["missing_degree"]):
            raise RuntimeError("generated boundary ledger mismatch")
        reference, initial_p = None, None
        for name in ("observed_p", "identity", "strong_mlp"):
            train_folder = folder / name
            record_train = json.loads((train_folder / "training_record.json").read_text(encoding="utf-8"))
            initial = torch.load(train_folder / "initial_state.pt", map_location="cpu", weights_only=True)
            selected = torch.load(train_folder / "selected_state.pt", map_location="cpu", weights_only=True)
            trajectory = json.loads((train_folder / "validation_trajectory.json").read_text(encoding="utf-8"))
            if len(trajectory) != 100 or record_train["selected_round"] != max(trajectory, key=lambda r: r["validation_macro_f1"])["round"]:
                raise RuntimeError("selection budget/earliest validation maximum mismatch")
            if record_train["test_targets_used"] or tensor_digest(initial) != record_train["initial_tensor_sha256"] or tensor_digest(selected) != record_train["selected_tensor_sha256"]:
                raise RuntimeError("training target/state integrity mismatch")
            caps = spec["resources"]
            if record_train["training_seconds"] > caps["seconds_per_training_run"] or any(record_train[k] > caps[k] for k in ("cuda_peak_gib", "process_peak_rss_gib")):
                raise RuntimeError("individual training resource cap")
            test_folder = output / "test_evaluation" / graph_id / name
            path = test_folder / "test_metrics.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            evidence_path = test_folder / "test_evidence.pt"
            if sha256(evidence_path) != record["evidence_sha256"] or record["checkpoint_sha256"] != sha256(train_folder / "selected_state.pt"):
                raise RuntimeError("test artifact hash mismatch")
            if record["split"] != "test" or record["phase"] != "independent_confirmation":
                raise RuntimeError("test phase mismatch")
            evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
            nodes = evidence["nodes"]
            if len(nodes) != 2400 or len(set(nodes.tolist())) != 2400 or not masks["test_sealed"][nodes].all():
                raise RuntimeError("test node isolation mismatch")
            if not torch.equal(evidence["labels"], data["labels"][nodes]):
                raise RuntimeError("test target mismatch")
            audit_raw_rows(record, evidence)
            if name == "observed_p":
                reference, initial_p = evidence, record_train["initial_tensor_sha256"]
                cert = evidence["certificates"]
                if not torch.equal(cert["cone_hybrid"], cert["cone_only"] | cert["retained_pairwise"]):
                    raise RuntimeError("union mismatch")
                if metadata["rho"] == 0 and (evidence["groups"]["exposed"].any() or evidence["flip"].any()):
                    raise RuntimeError("zero-cut exposure/invariance mismatch")
            else:
                if not torch.equal(nodes, reference["nodes"]) or any(not torch.equal(value, reference["groups"][key]) for key, value in evidence["groups"].items()):
                    raise RuntimeError("reference exposure mismatch")
                if not evidence["certificates"]["structural_invariance"].all() or evidence["flip"].any():
                    raise RuntimeError("invariant control mismatch")
                if name == "identity" and record_train["initial_tensor_sha256"] != initial_p:
                    raise RuntimeError("P/I initial state pairing mismatch")
            if record["reliability"]["bound_violations"] or record["reliability"]["certified_flip_nodes"]:
                raise RuntimeError("reliability violation")
            test_hashes[str(path.relative_to(output))] = sha256(path)
            test_hashes[str(evidence_path.relative_to(output))] = sha256(evidence_path)
        reports.append({k: metadata[k] for k in ("seed", "h", "rho", "nodes", "edges", "mean_degree", "actual_homophily", "actual_cut_ratio", "split_counts")})
    if len(reports) != 96:
        raise RuntimeError("missing graph units")
    caps = spec["resources"]
    if commit["training_elapsed_seconds"] > caps["seconds_total_training"] or completion["test_elapsed_seconds"] > caps["seconds_total_test_audit"]:
        raise RuntimeError("phase walltime cap")
    for report in (commit, completion):
        if any(report[k] > caps[k] for k in ("cuda_peak_gib", "process_peak_rss_gib")):
            raise RuntimeError("phase memory cap")
    check_index(output, commit["artifact_sha256"])
    from run_fedbcert_e28b_controlled import verify_manifest
    verify_manifest(manifest)
    elapsed, usage = time.monotonic() - started, resource_usage(torch.device("cpu"))
    if completion["test_elapsed_seconds"] + elapsed > caps["seconds_total_test_audit"] or any(usage[k] > caps[k] for k in usage):
        raise RuntimeError("combined test/artifact-audit resource cap")
    write_json(output / "artifact_integrity_audit.json", dict(
        experiment=spec["experiment"], all_checks_passed=True, graph_units=96, training_runs=288,
        test_nodes_per_graph=2400, test_evaluated=True, training_unchanged_after_commit=True,
        checkpoint_commit_sha256=sha256(output / "checkpoint_commit.json"),
        test_completion_sha256=sha256(output / "test_evaluation/completion.json"),
        artifact_audit_seconds=elapsed, **usage,
        test_artifact_sha256=test_hashes, graphs=reports,
        checks=["freeze/protected hashes", "all-checkpoint commit and test file hashes", "earliest validation-selected checkpoint",
                "no test targets in training", "original test-only evidence", "raw Macro-F1 and count/ratio recomputation",
                "exact generated ledger", "matched initialization and shared exposure", "union and zero certified flips/bound violations", "resource caps"]))
    print("Confirmation artifact audit passed: 96 graphs, 288 training/test units, original 2400-node test masks.")


if __name__ == "__main__":
    main()
