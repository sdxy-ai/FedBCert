"""Read-only input audit for the fixed E28b on-demand inference replay.

No training, checkpoint selection, query policy, cost experiment, or paper edit.
Only a new exclusive input_audit.json is written. Trusted historical torch
bundles are loaded only after their committed file hashes have been checked.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from fedbcert_e28b_common import make_clients, make_model, tensor_digest
from openfgl.flcore.fedbcert.certificate import row_normalized_shift

SEEDS = tuple(range(28101, 28109))
SOURCE = Path("results_boundary_certified/boundary_certified_spectral_e28b_confirmation")
OUTPUT = ROOT / "results_boundary_certified/fedbcert_on_demand_20261015/input_audit.json"
SPLITS = {"train": 1800, "audit_unused": 600, "validation": 1200, "test_sealed": 2400}
CERTIFICATES = {"additive_classwise", "retained_pairwise", "cone_only", "cone_hybrid"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = sha256(path)
    require(actual == expected, f"hash mismatch: {path}")
    return actual


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_exclusive_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def tensor_schema(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: tensor_schema(item) for key, item in value.items()}
    return type(value).__name__


def validate_bundle(data, seed):
    required = {"features", "labels", "membership", "full_edge_index", "missing_degree", "masks", "metadata"}
    require(set(data) == required, "unexpected graph bundle schema")
    require(data["features"].shape == (6000, 32) and data["features"].dtype == torch.float32,
            "feature schema changed")
    require(bool(torch.isfinite(data["features"]).all()), "nonfinite features")
    for name in ("labels", "membership"):
        require(data[name].shape == (6000,) and data[name].dtype == torch.int64, f"{name} schema changed")
    require(torch.equal(torch.bincount(data["labels"]), torch.tensor([3000, 3000])), "class counts changed")
    require(torch.equal(torch.bincount(data["membership"]), torch.tensor([2000, 2000, 2000])),
            "client membership counts changed")
    require(set(data["masks"]) == set(SPLITS), "split names changed")
    cover = torch.zeros(6000, dtype=torch.int64)
    for name, count in SPLITS.items():
        mask = data["masks"][name]
        require(mask.dtype == torch.bool and mask.shape == (6000,), f"invalid {name} mask")
        require(int(mask.sum()) == count, f"{name} node count changed")
        cover += mask.long()
    require(bool((cover == 1).all()), "splits overlap or fail to cover graph")
    edges = data["full_edge_index"]
    require(edges.dtype == torch.int64 and edges.ndim == 2 and edges.shape[0] == 2, "edge schema changed")
    require(bool(((edges >= 0) & (edges < 6000)).all()), "edge endpoint out of range")
    require(not bool((edges[0] == edges[1]).any()), "unexpected raw self-loop")
    keys = edges[0] * 6000 + edges[1]
    require(torch.unique(keys).numel() == keys.numel(), "duplicate raw edge")
    require(torch.equal(keys.sort().values, (edges[1] * 6000 + edges[0]).sort().values),
            "full graph is not undirected")
    cross = data["membership"][edges[0]] != data["membership"][edges[1]]
    exact = torch.bincount(edges[0, cross], minlength=6000)
    missing = data["missing_degree"]
    require(missing.dtype == torch.float32 and missing.shape == (6000,), "ledger schema changed")
    require(torch.equal(missing.double(), exact.double()), "ledger under/over-count or noninteger value")
    meta = data["metadata"]
    require(meta["seed"] == seed and meta["h"] == .9 and meta["rho"] == .4, "wrong fixed primary cell")
    require(meta["nodes"] == 6000 and meta["edges"] * 2 == edges.shape[1], "graph metadata count mismatch")
    require(meta["split_counts"] == SPLITS, "metadata split counts mismatch")
    return {"nodes": 6000, "undirected_edges": edges.shape[1] // 2,
            "cross_client_undirected_edges": int(cross.sum()) // 2,
            "clients": 3, "nodes_per_client": 2000, "split_counts": SPLITS}


def validate_evidence(data, evidence):
    nodes = evidence["nodes"]
    require(nodes.dtype == torch.int64 and nodes.shape == (2400,), "test node schema changed")
    require(torch.unique(nodes).numel() == 2400 and bool(data["masks"]["test_sealed"][nodes].all()),
            "test evidence contains duplicate/non-test nodes")
    require(torch.equal(evidence["labels"], data["labels"][nodes]), "historical evidence label mismatch")
    require(set(evidence["certificates"]) == CERTIFICATES and set(evidence["slacks"]) == CERTIFICATES,
            "certificate evidence schema changed")
    for name in ("nominal_logits", "completion_logits"):
        values = evidence[name]
        require(values.shape == (2400, 2) and values.dtype == torch.float64 and bool(torch.isfinite(values).all()),
                f"invalid historical {name}")
    tol = float(evidence["reliability"]["numeric_tolerance"])
    require(tol == 1e-6, "historical numeric guard changed")
    flips = evidence["nominal_logits"].argmax(1) != evidence["completion_logits"].argmax(1)
    require(torch.equal(flips, evidence["flip"]), "historical flip flags mismatch")
    for name in CERTIFICATES:
        cert, slack = evidence["certificates"][name], evidence["slacks"][name]
        require(cert.shape == (2400,) and cert.dtype == torch.bool, "certificate mask schema changed")
        require(slack.shape == (2400,) and bool(torch.isfinite(slack).all()), "invalid certificate slack")
        require(torch.equal(cert, slack > tol), f"historical {name} mask/guard mismatch")
        require(not bool((cert & flips).any()), f"historical certified flip: {name}")
    require(torch.equal(evidence["certificates"]["cone_hybrid"],
                        evidence["certificates"]["cone_only"] | evidence["certificates"]["retained_pairwise"]),
            "historical certificate union mismatch")


@torch.no_grad()
def replay_predictions(base, data, state, evidence):
    model = make_model(base, "observed_p", dtype=torch.float64).eval()
    model.load_state_dict(state, strict=True)
    require(sum(parameter.numel() for parameter in model.parameters()) == 8775, "model parameter count changed")
    clients = make_clients(base, data, "cpu", dtype=torch.float64)
    nominal = torch.empty((6000, 2), dtype=torch.float64)
    schemas = []
    for client in clients:
        require(client.shift.is_coalesced(), "local shift is not coalesced")
        require(bool((client.shift.values() >= 0).all()), "negative propagation weight")
        require(torch.allclose(torch.sparse.sum(client.shift, dim=1).to_dense(), torch.ones(2000, dtype=torch.float64),
                               rtol=0, atol=1e-14), "local shift is not row stochastic")
        nominal[client.global_nodes] = model(client.features, client.shift)
        schemas.append({"client_id": client.client_id, "global_nodes": tensor_schema(client.global_nodes),
                        "features": tensor_schema(client.features), "shift_shape": list(client.shift.shape),
                        "shift_nnz_including_self_loops": client.shift._nnz(),
                        "missing_degree": tensor_schema(client.missing_degree)})
    full_shift, _ = row_normalized_shift(data["full_edge_index"], 6000,
                                         device=torch.device("cpu"), dtype=torch.float64)
    completed = model(data["features"].double(), full_shift)
    nodes = evidence["nodes"]
    errors = {"nominal_max_abs": float((nominal[nodes] - evidence["nominal_logits"]).abs().max()),
              "full_graph_max_abs": float((completed[nodes] - evidence["completion_logits"]).abs().max())}
    require(max(errors.values()) <= 1e-10, f"historical prediction replay mismatch: {errors}")
    require(torch.equal(completed[nodes].argmax(1), evidence["completion_logits"].argmax(1)),
            "historical full-graph class replay mismatch")
    require(torch.equal(nominal[nodes].argmax(1), evidence["nominal_logits"].argmax(1)),
            "historical nominal class replay mismatch")
    return errors, schemas


def audit_inputs(root=ROOT):
    root = Path(root)
    source = root / SOURCE
    verified = {}

    def record(path, expected=None):
        path = Path(path)
        digest = verify_hash(path, expected) if expected is not None else sha256(path)
        verified[str(path.relative_to(root))] = digest
        return digest

    protocol_path = root / "configs/fedbcert_e28b_confirmation_protocol.json"
    spec = read_json(protocol_path)
    base_path = root / spec["base_protocol"]
    record(base_path, spec["base_protocol_sha256"])
    base = read_json(base_path)
    require(tuple(spec["confirmation"]["seeds"]) == SEEDS, "confirmation seed namespace changed")
    require(spec["confirmation"]["primary_cell"] == {"h": .9, "rho": .4}, "primary cell changed")
    freeze_path, commit_path = source / "freeze_manifest.json", source / "checkpoint_commit.json"
    freeze, commit = read_json(freeze_path), read_json(commit_path)
    record(protocol_path, commit["protocol_sha256"])
    record(freeze_path, commit["freeze_manifest_sha256"])
    record(commit_path)
    require(commit["all_training_complete"] is True and commit["training_runs"] == 288 and commit["graph_units"] == 96,
            "incomplete checkpoint commit")
    require(commit["test_evaluated_before_commit"] is False, "historical test isolation failed")
    require(len(commit["matrix"]) == 288, "committed training matrix changed")
    source_names = ("configs/fedbcert_e28b_controlled_protocol_v2.json",
                    "configs/fedbcert_e28b_confirmation_protocol.json", "scripts/fedbcert_e28b_common.py",
                    "scripts/fedbcert_e28b_confirmation_common.py")
    for name in source_names:
        record(root / name, freeze["source_sha256"][name])
    for name in ("openfgl/flcore/fedbcert/certificate.py", "openfgl/flcore/fedbcert/federated.py"):
        record(root / name, freeze["protected_sha256"][name])
    completion_path = source / "test_evaluation/completion.json"
    completion = read_json(completion_path)
    record(completion_path)
    require(completion["completed_test_runs"] == 288 and completion["checkpoints_mutated"] is False,
            "incomplete historical test evaluation")
    require(completion["checkpoint_commit_sha256"] == sha256(commit_path), "historical completion commit changed")
    cases = []
    for seed in SEEDS:
        graph_id = f"h0.90_rho0.40_seed{seed}"
        train = source / "training" / graph_id
        for relative in ("sealed_graph_bundle.pt", "graph_manifest.json"):
            path = train / relative
            record(path, commit["artifact_sha256"][str(path.relative_to(source))])
        for name in ("observed_p", "identity", "strong_mlp"):
            for relative in ("initial_state.pt", "selected_state.pt", "validation_trajectory.json", "training_record.json"):
                path = train / name / relative
                record(path, commit["artifact_sha256"][str(path.relative_to(source))])
        data_path = train / "sealed_graph_bundle.pt"
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        summary = validate_bundle(data, seed)
        manifest = read_json(train / "graph_manifest.json")
        require(manifest["bundle_sha256"] == sha256(data_path), "graph manifest bundle hash mismatch")
        for key, value in data["metadata"].items():
            require(manifest[key] == value, f"graph manifest metadata mismatch: {key}")
        selected_path = train / "observed_p/selected_state.pt"
        state = torch.load(selected_path, map_location="cpu", weights_only=True)
        initial = torch.load(train / "observed_p/initial_state.pt", map_location="cpu", weights_only=True)
        identity_initial = torch.load(train / "identity/initial_state.pt", map_location="cpu", weights_only=True)
        training = read_json(train / "observed_p/training_record.json")
        require(training["checkpoint_sha256"] == sha256(selected_path), "training checkpoint hash mismatch")
        require(training["selected_tensor_sha256"] == tensor_digest(state), "selected tensor digest changed")
        require(training["initial_tensor_sha256"] == tensor_digest(initial) == tensor_digest(identity_initial),
                "P/I initial state pairing mismatch")
        require(training["trainable_parameters"] == 8775 and training["test_targets_used"] is False,
                "training protocol record changed")
        with torch.random.fork_rng():
            torch.manual_seed(data["metadata"]["derived_seeds"]["model"])
            regenerated = make_model(base, "observed_p")
        require(tensor_digest(regenerated.state_dict()) == tensor_digest(initial), "initial model replay changed")
        trajectory = read_json(train / "observed_p/validation_trajectory.json")
        best = max(trajectory, key=lambda entry: entry["validation_macro_f1"])
        require(len(trajectory) == 100 and best["round"] == training["selected_round"], "checkpoint selection record mismatch")
        test = source / "test_evaluation" / graph_id / "observed_p"
        metrics_path, evidence_path = test / "test_metrics.json", test / "test_evidence.pt"
        record(metrics_path, completion["records_sha256"][str(metrics_path.relative_to(source))])
        metrics = read_json(metrics_path)
        record(evidence_path, metrics["evidence_sha256"])
        require(metrics["checkpoint_sha256"] == sha256(selected_path) and metrics["commit_sha256"] == sha256(commit_path),
                "historical test checkpoint chain mismatch")
        evidence = torch.load(evidence_path, map_location="cpu", weights_only=False)
        validate_evidence(data, evidence)
        before = tensor_digest(state)
        errors, clients = replay_predictions(base, data, state, evidence)
        require(tensor_digest(state) == before, "audit mutated selected state")
        cases.append({"seed": seed, "graph_id": graph_id, "bundle": str(data_path.relative_to(root)),
                      "selected_state": str(selected_path.relative_to(root)),
                      "historical_test_evidence": str(evidence_path.relative_to(root)),
                      "graph": summary, "selected_round": training["selected_round"],
                      "parameter_count": 8775, "initial_tensor_sha256": tensor_digest(initial),
                      "selected_tensor_sha256": before, "prediction_replay": errors,
                      "bundle_schema": tensor_schema(data), "state_schema": tensor_schema(state),
                      "evidence_schema": tensor_schema(evidence), "client_schema": clients})
    for relative, digest in verified.items():
        verify_hash(root / relative, digest)
    return {"schema_version": 1, "date": "2026-10-15", "passed": True,
            "scope": "read-only audit and historical forward replay of eight fixed synthetic primary cases",
            "not_performed": ["training", "checkpoint selection", "query/cost experiment", "new efficacy analysis", "paper modification"],
            "observed_model": {"encoder": "dual_feature_spectral", "parameters": 8775, "input_dim": 32,
                               "hidden_dim": 64, "classes": 2, "order_count": 3, "alpha": .25,
                               "residual_scale": .5, "evaluation_dtype": "float64", "training_state_dtype": "float32"},
            "source_hashes_unchanged_after_audit": True, "verified_file_count": len(verified),
            "verified_file_sha256": verified, "cases": cases,
            "limitations": ["synthetic historical checkpoints, not new independent application confirmation",
                            "floating-point replay is not outward-rounded verification",
                            "historical evidence contains labels; no query-policy or inference payload is produced here"]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = audit_inputs()
    write_exclusive_json(OUTPUT, report)
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "verified_files": report["verified_file_count"], "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
