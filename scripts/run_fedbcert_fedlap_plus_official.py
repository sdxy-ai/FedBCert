#!/usr/bin/env python3
"""Run pinned FedLap+ (NeurIPS 2025) code on one immutable E8 split.

The adapter preserves the official LanczosLaplace/Arnoldi model, loss and
gradient aggregation.  It only installs the frozen FedBCert membership and
label masks, applies the paper's predeclared dataset hyperparameters, and
selects one checkpoint with pooled validation Macro-F1.  Test labels are read
once, after checkpoint restoration.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEDLAP_COMMIT = "652fba0bb2e9a660a31ce7e8abcf728c225d188c"
ADAPTER_REVISION = "fedlap-plus-e8-v1-paper-hparams"
CONFIG_NAMES = {
    "Cora": "config_Cora.yml",
    "CiteSeer": "config_CiteSeer.yml",
    "PubMed": "config_Pubmed.yml",
    "Photo": "config_Photo.yml",
}

# Table 4 in the NeurIPS 2025 FedLap paper.  Computers is deliberately absent:
# it is available in the code repository but not in the paper's experiments.
PAPER_HYPERPARAMETERS = {
    "Cora": {
        "lr": 0.003, "weight_decay": 0.0005, "rounds": 100,
        "regularizer_coef": 1.0, "structural_dim": 512, "arnoldi_rank": 100,
    },
    "CiteSeer": {
        "lr": 0.002, "weight_decay": 0.0005, "rounds": 100,
        "regularizer_coef": 1.0, "structural_dim": 1024, "arnoldi_rank": 100,
    },
    "PubMed": {
        "lr": 0.001, "weight_decay": 0.0003, "rounds": 150,
        "regularizer_coef": 1.0, "structural_dim": 256, "arnoldi_rank": 100,
    },
    "Photo": {
        "lr": 0.001, "weight_decay": 0.0005, "rounds": 150,
        "regularizer_coef": 0.1, "structural_dim": 512, "arnoldi_rank": 75,
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(CONFIG_NAMES), required=True)
    p.add_argument("--partition", choices=["metis", "random"], required=True)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--bundle-root",
        default="results_boundary_certified/boundary_certified_spectral_e8_q1/official_bridges/bundles",
    )
    p.add_argument("--fedlap-root", default="third_party/fedbcert_e8/FedLap")
    p.add_argument(
        "--output-root",
        default="results_boundary_certified/boundary_certified_spectral_e8_q1/official_bridges/fedlap_plus_core",
    )
    p.add_argument("--rounds", type=int, default=None, help="Smoke-test override only.")
    p.add_argument(
        "--export-aligned-logits",
        action="store_true",
        help=(
            "Export label-free, all-node logits from the validation-selected "
            "checkpoint in immutable-bundle node order."
        ),
    )
    p.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help=(
            "Do not read or emit test metrics. Intended for E22 teacher "
            "materialization, where the final wrapper owns the sole test pass."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def prepare_import_root(fedlap_root: Path, output_dir: Path) -> tuple[Path, list[str]]:
    """Create an isolated, semantics-neutral Python 3.9 compatibility shadow."""
    shadow_root = output_dir / "python39_compat"
    shadow_src = shadow_root / "src"
    if shadow_src.exists():
        shutil.rmtree(shadow_src)
    shutil.copytree(fedlap_root / "src", shadow_src)
    patched: list[str] = []
    if sys.version_info < (3, 10):
        for relative in ["src/utils/data.py", "src/utils/graph.py", "src/GNN/GNN_client.py"]:
            path = shadow_root / relative
            source = path.read_text(encoding="utf-8")
            if not source.startswith("from __future__ import annotations"):
                path.write_text("from __future__ import annotations\n" + source, encoding="utf-8")
            patched.append(relative + ":postponed-annotations")
    return shadow_root, patched


def clone_state(client: Any) -> dict[str, Any]:
    return {
        "weights": copy.deepcopy(client.state_dict()),
        # Official hop2vec/FedLap spectral coordinates are optimized outside
        # the classifier state_dict and must be checkpointed explicitly.
        "structural_features": client.classifier.get_x().detach().cpu().clone(),
    }


def restore_state(client: Any, state: dict[str, Any]) -> None:
    client.load_state_dict(state["weights"])
    with torch.no_grad():
        client.classifier.get_x().copy_(
            state["structural_features"].to(client.classifier.get_x().device)
        )


@torch.no_grad()
def evaluate(clients: list[Any], mask_name: str) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    client_macro_f1: list[float] = []
    client_counts: list[int] = []
    for client in clients:
        client.eval()
        mask = getattr(client.graph, mask_name).to(client.graph.y.device)
        logits = client.classifier.get_embeddings()
        pred = logits[mask].argmax(dim=1).detach().cpu().numpy()
        truth = client.graph.y[mask].detach().cpu().numpy()
        if truth.size == 0:
            client_macro_f1.append(float("nan"))
            client_counts.append(0)
            continue
        predictions.append(pred)
        truths.append(truth)
        client_macro_f1.append(float(f1_score(truth, pred, average="macro", zero_division=0)))
        client_counts.append(int(truth.size))
    if not truths:
        raise RuntimeError(f"no nodes available for {mask_name}")
    pooled_pred = np.concatenate(predictions)
    pooled_truth = np.concatenate(truths)
    finite = [value for value in client_macro_f1 if np.isfinite(value)]
    return {
        "pooled_macro_f1": float(f1_score(pooled_truth, pooled_pred, average="macro", zero_division=0)),
        "pooled_accuracy": float(np.mean(pooled_truth == pooled_pred)),
        "worst_client_macro_f1": float(min(finite)),
        "client_macro_f1": client_macro_f1,
        "client_counts": client_counts,
        "node_count": int(pooled_truth.size),
    }


@torch.no_grad()
def aligned_client_logits(
    clients: list[Any], node_count: int, class_count: int
) -> torch.Tensor:
    """Assemble personalized client outputs in immutable global-node order."""
    output = torch.empty((int(node_count), int(class_count)), dtype=torch.float32)
    assigned = torch.zeros(int(node_count), dtype=torch.bool)
    for client in clients:
        client.eval()
        node_ids = client.graph.node_ids.detach().cpu().long()
        logits = client.classifier.get_embeddings().detach().cpu().float()
        if logits.ndim != 2 or logits.shape != (node_ids.numel(), int(class_count)):
            raise RuntimeError("FedLap+ logits are not aligned with client node_ids")
        if bool((node_ids < 0).any()) or bool((node_ids >= int(node_count)).any()):
            raise RuntimeError("FedLap+ client emitted an out-of-range global node id")
        if bool(assigned[node_ids].any()):
            raise RuntimeError("FedLap+ clients duplicated a global node id")
        output[node_ids] = logits
        assigned[node_ids] = True
    if not bool(assigned.all()):
        missing = int((~assigned).sum().item())
        raise RuntimeError(f"FedLap+ aligned logit export missed {missing} nodes")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("FedLap+ aligned logit export contains non-finite values")
    return output


def main() -> None:
    args = parser().parse_args()
    if args.num_clients < 2:
        raise ValueError("FedLap+ confirmation requires at least two clients")

    fedlap_root = (REPOSITORY_ROOT / args.fedlap_root).resolve()
    if not (fedlap_root / ".git").exists():
        raise FileNotFoundError(
            f"missing official FedLap checkout: {fedlap_root}; run "
            "bash scripts/setup_fedlap.sh first"
        )
    actual_commit = git_head(fedlap_root)
    if actual_commit != FEDLAP_COMMIT:
        raise RuntimeError(f"FedLap commit mismatch: {actual_commit} != {FEDLAP_COMMIT}")

    run_name = f"{args.dataset}_{args.partition}_c{args.num_clients}_seed{args.seed}"
    bundle_dir = (REPOSITORY_ROOT / args.bundle_root / run_name).resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing immutable bundle {manifest_path}; prepare it with "
            "scripts/prepare_fedbcert_e8_official_bundle.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_path = bundle_dir / manifest["bundle"]
    if file_sha256(bundle_path) != manifest["bundle_sha256"]:
        raise RuntimeError("immutable split bundle SHA-256 mismatch")

    output_dir = (REPOSITORY_ROOT / args.output_root / run_name).resolve()
    result_path = output_dir / "fedlap_plus_official_metrics.json"
    completion_path = output_dir / "run_complete.json"
    logits_path = output_dir / "utility_logits.pt"
    if completion_path.exists() and result_path.exists() and not args.overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        requested_is_smoke = args.rounds is not None
        existing_matches_mode = bool(existing.get("smoke_round_override")) == requested_is_smoke
        existing_matches_test_scope = (
            int(existing.get("test_evaluation_count", 1))
            == int(not args.skip_test_evaluation)
        )
        requested_artifact_available = (
            not args.export_aligned_logits or logits_path.exists()
        )
        if (
            existing.get("adapter_revision") == ADAPTER_REVISION
            and existing_matches_mode
            and existing_matches_test_scope
            and requested_artifact_available
        ):
            print(f"[SKIP] complete: {output_dir}")
            return
        print(f"[RERUN] legacy or smoke/formal mode-mismatched result: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = fedlap_root / "config" / CONFIG_NAMES[args.dataset]
    official_log_root = output_dir / "official_logs"
    official_log_root.mkdir(parents=True, exist_ok=True)
    os.environ["CONFIGPATH"] = str(config_path)
    os.environ["RESULTPATH"] = str(official_log_root) + os.sep
    os.environ["seed"] = str(args.seed)
    os.chdir(fedlap_root)
    import_root, compatibility_patches = prepare_import_root(fedlap_root, output_dir)
    sys.path.insert(0, str(import_root))

    from src import config, device, sum_lod  # type: ignore
    from src.GNN.GNN_server import GNNServer  # type: ignore
    from src.utils.graph import Graph  # type: ignore
    from src.utils.graph_partitioning import create_subgraps  # type: ignore

    paper = PAPER_HYPERPARAMETERS[args.dataset]
    original_values = {
        "lr": float(config.model.lr),
        "weight_decay": float(config.model.weight_decay),
        "rounds": int(config.model.iterations),
        "regularizer_coef": float(config.spectral.regularizer_coef),
        "structural_dim": int(config.structure_model.num_structural_features),
        "spectral_len": int(config.spectral.spectral_len),
        "lanczos_iter": int(config.spectral.lanczos_iter),
        "matrix": str(config.spectral.matrix),
        "method": str(config.spectral.method),
    }
    config.model.lr = paper["lr"]
    config.model.weight_decay = paper["weight_decay"]
    config.model.iterations = paper["rounds"]
    config.spectral.regularizer_coef = paper["regularizer_coef"]
    config.structure_model.num_structural_features = paper["structural_dim"]
    config.spectral.spectral_len = paper["arnoldi_rank"]
    config.spectral.lanczos_iter = paper["arnoldi_rank"]
    config.spectral.matrix = "lap"
    config.spectral.method = "arnoldi"

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    expected_keys = {
        "x", "y", "edge_index", "membership", "train_mask",
        "calibration_mask", "validation_mask", "test_mask",
    }
    if set(bundle) != expected_keys:
        raise RuntimeError(f"unexpected bundle keys: {sorted(bundle)}")
    membership = bundle["membership"].long()
    if int(membership.min()) != 0 or int(membership.max()) + 1 != args.num_clients:
        raise RuntimeError("bundle membership/client-count mismatch")
    if manifest["dataset"] != args.dataset or manifest["partition"] != args.partition:
        raise RuntimeError("bundle dataset/partition mismatch")
    if int(manifest["num_clients"]) != args.num_clients or int(manifest["seed"]) != args.seed:
        raise RuntimeError("bundle manifest does not match requested run")

    graph = Graph(
        x=bundle["x"].float().to(device),
        y=bundle["y"].long().to(device),
        edge_index=bundle["edge_index"].long().to(device),
        node_ids=torch.arange(bundle["x"].size(0), device=device),
        keep_sfvs=True,
        train_mask=bundle["train_mask"].bool(),
        val_mask=bundle["validation_mask"].bool(),
        test_mask=bundle["test_mask"].bool(),
        num_classes=int(manifest["class_count"]),
    )
    groups = {
        client_id: torch.where(membership == client_id)[0].to(device)
        for client_id in range(args.num_clients)
    }
    subgraphs = create_subgraps(graph, groups)
    if sum(subgraph.num_nodes for subgraph in subgraphs) != graph.num_nodes:
        raise RuntimeError("official subgraph conversion lost or duplicated nodes")

    server = GNNServer(graph)
    for subgraph in subgraphs:
        server.add_client(subgraph)
    start = time.perf_counter()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    smodel_type = "LanczosLaplace"
    fmodel_type = str(config.model.fmodel_type)
    structure_type = "hop2vec"
    rank = int(paper["arnoldi_rank"])
    print(
        f"[{run_name}] initializing official FedLap+ "
        f"{smodel_type}+{fmodel_type}, Arnoldi rank={rank}", flush=True,
    )
    initialization_start = time.perf_counter()
    server.initialize_FL(
        smodel_type=smodel_type,
        fmodel_type=fmodel_type,
        data_type="f+s",
        spectral_len=rank,
        structure_type=structure_type,
        num_structural_features=int(paper["structural_dim"]),
        log=False,
    )
    initialization_seconds = time.perf_counter() - initialization_start
    print(f"[{run_name}] initialization complete in {initialization_seconds:.1f}s", flush=True)
    server.share_weights()

    rounds = int(paper["rounds"] if args.rounds is None else args.rounds)
    node_total = sum(client.num_nodes() for client in server.clients)
    coefficients = [client.num_nodes() / node_total for client in server.clients]
    best_score = (-float("inf"), -float("inf"))
    best_round = 0
    best_states: list[dict[str, Any]] | None = None
    history: list[dict[str, Any]] = []

    for round_id in range(1, rounds + 1):
        server.reset_trainings()
        server.set_train_mode()
        train_results = server.train_clients(eval_=False)
        client_grads = server.get_grads()
        server.share_grads(sum_lod(client_grads, coefficients))
        server.update_models()
        validation = evaluate(server.clients, "val_mask")
        score = (validation["pooled_macro_f1"], validation["pooled_accuracy"])
        if score > best_score:
            best_score = score
            best_round = round_id
            best_states = [clone_state(client) for client in server.clients]
        history.append({
            "round": round_id,
            "train_loss_node_weighted": float(sum(
                coef * item["Train Loss"] for coef, item in zip(coefficients, train_results)
            )),
            "validation_pooled_macro_f1": validation["pooled_macro_f1"],
            "validation_pooled_accuracy": validation["pooled_accuracy"],
            "validation_worst_client_macro_f1": validation["worst_client_macro_f1"],
        })
        print(
            f"[{run_name}] {round_id:03d}/{rounds} "
            f"val_macro_f1={validation['pooled_macro_f1']:.4f} "
            f"best={best_score[0]:.4f}@{best_round}", flush=True,
        )

    if best_states is None:
        raise RuntimeError("no FedLap+ checkpoint was selected")
    for client, state in zip(server.clients, best_states):
        restore_state(client, state)
    exported_logits = None
    if args.export_aligned_logits:
        exported_logits = aligned_client_logits(
            server.clients,
            int(bundle["x"].size(0)),
            int(manifest["class_count"]),
        )
        torch.save(
            {
                "logits": exported_logits,
                "bundle_sha256": manifest["bundle_sha256"],
                "dataset": args.dataset,
                "partition": args.partition,
                "num_clients": int(args.num_clients),
                "seed": int(args.seed),
                "selected_round": int(best_round),
                "source_commit": actual_commit,
                "labels_included": False,
            },
            logits_path,
        )
    validation = evaluate(server.clients, "val_mask")
    test = (
        None
        if args.skip_test_evaluation
        else evaluate(server.clients, "test_mask")  # first and only test evaluation
    )
    wall_seconds = time.perf_counter() - start
    peak_cuda_mb = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_cuda_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))

    history_path = output_dir / "round_validation_metrics.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    record = {
        "protocol": "FedBCert E8 FedLap+ official-code matched confirmation",
        "adapter_revision": ADAPTER_REVISION,
        "method": "FedLap+-official-Arnoldi",
        "official_repository": "https://github.com/JavadAliakbari/FedLap",
        "official_paper": "https://papers.nips.cc/paper_files/paper/2025/hash/aaf93e15b3475281dcbcb61ae3172992-Abstract-Conference.html",
        "source_commit": actual_commit,
        "runtime_compatibility_patches": compatibility_patches,
        "runtime_compatibility_scope": "postponed evaluation of Python annotations on Python 3.9 only",
        "dataset": args.dataset,
        "partition": args.partition,
        "num_clients": args.num_clients,
        "seed": args.seed,
        "bundle_sha256": manifest["bundle_sha256"],
        "exact_fedbcert_bundle_consumed": True,
        "paper_hyperparameters_applied": True,
        "paper_hyperparameters": paper,
        "checked_in_yaml_values_before_paper_override": original_values,
        "selection_metric": "pooled validation Macro-F1; pooled accuracy tie-break",
        "test_labels_used_for_training_or_selection": False,
        "test_evaluation_count": int(not args.skip_test_evaluation),
        "selected_round": best_round,
        "rounds": rounds,
        "smoke_round_override": args.rounds is not None,
        "official_config": str(config_path.relative_to(fedlap_root)),
        "variant": {
            "data_type": "f+s",
            "feature_model": fmodel_type,
            "structure_model": smodel_type,
            "structure_type": structure_type,
            "spectral_matrix": "lap",
            "spectral_method": "arnoldi",
            "aggregation": "official gradient aggregation weighted by client node count",
            "intrinsic_laplacian_regularizer": True,
        },
        "information_interface": {
            "matches_fedbcert": False,
            "reason": (
                "official FedLap+ simulation constructs global Laplacian Arnoldi factors; "
                "the decentralized protocol exchanges cross-client sparse matrix-vector "
                "products and assumes cross-client endpoint knowledge"
            ),
            "deterministic_boundary_completion_certificate": False,
        },
        "calibration_nodes_unused": int(bundle["calibration_mask"].sum()),
        "validation": validation,
        "test": test,
        "initialization_seconds": initialization_seconds,
        "wall_seconds": wall_seconds,
        "peak_cuda_memory_mb": peak_cuda_mb,
        "history_file": history_path.name,
        "aligned_logit_export": (
            {
                "file": logits_path.name,
                "sha256": file_sha256(logits_path),
                "node_count": int(exported_logits.size(0)),
                "class_count": int(exported_logits.size(1)),
                "labels_included": False,
            }
            if exported_logits is not None
            else None
        ),
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completion_path.write_text(
        json.dumps({"complete": True, "result": result_path.name}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
