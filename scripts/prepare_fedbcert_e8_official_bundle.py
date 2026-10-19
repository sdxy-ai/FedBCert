#!/usr/bin/env python3
"""Export an immutable E8 split bundle for official baseline repositories."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from openfgl.flcore.fedbcert import canonical_undirected
from openfgl.flcore.fedbeso import balanced_random_membership, metis_membership
from openfgl.flcore.fedbcert.data import (
    load_cross_network_dataset,
    stratified_four_way_masks,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["Cora", "CiteSeer", "PubMed", "Computers", "Photo"], required=True)
    p.add_argument("--partition", choices=["metis", "random"], required=True)
    p.add_argument("--num-clients", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-root", default="data/fedbcert")
    p.add_argument("--output-root", default="results_boundary_certified/boundary_certified_spectral_e8_q1/official_bridges/bundles")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parser().parse_args()
    data = load_cross_network_dataset(args.dataset, args.data_root)
    x = data.x.detach().cpu().float()
    y = data.y.detach().cpu().reshape(-1).long()
    edge_index = canonical_undirected(data.edge_index, int(data.num_nodes)).cpu()
    if args.partition == "metis":
        membership = metis_membership(edge_index, int(data.num_nodes), args.num_clients, args.seed)
    else:
        membership = balanced_random_membership(int(data.num_nodes), args.num_clients, args.seed)
    masks = stratified_four_way_masks(
        y,
        train_ratio=0.30,
        calibration_ratio=0.10,
        val_ratio=0.20,
        test_ratio=0.40,
        seed=args.seed,
    )
    name = f"{args.dataset}_{args.partition}_c{args.num_clients}_seed{args.seed}"
    output = Path(args.output_root) / name
    existing_manifest_path = output / "manifest.json"
    existing_bundle_path = output / "split_bundle.pt"
    if existing_manifest_path.exists() and existing_bundle_path.exists():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        identity = {
            "dataset": args.dataset,
            "partition": args.partition,
            "num_clients": args.num_clients,
            "seed": args.seed,
        }
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"existing bundle identity mismatch: {output}")
        if sha256(existing_bundle_path) != existing.get("bundle_sha256"):
            raise RuntimeError(f"existing immutable bundle hash mismatch: {output}")
        print(f"[SKIP] verified immutable bundle: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / "split_bundle.pt"
    torch.save(
        {
            "x": x,
            "y": y,
            "edge_index": edge_index,
            "membership": torch.as_tensor(membership, dtype=torch.long),
            "train_mask": masks[0].cpu(),
            "calibration_mask": masks[1].cpu(),
            "validation_mask": masks[2].cpu(),
            "test_mask": masks[3].cpu(),
        },
        bundle,
    )
    digest = sha256(bundle)
    manifest = {
        "protocol": "FedBCert E8 official baseline immutable split bundle",
        "dataset": args.dataset,
        "partition": args.partition,
        "num_clients": args.num_clients,
        "seed": args.seed,
        "node_count": int(data.num_nodes),
        "feature_dim": int(x.size(1)),
        "class_count": int(y.max().item()) + 1,
        "bundle": bundle.name,
        "bundle_sha256": digest,
        "test_labels_used_for_selection": False,
        "required_selection_metric": "pooled validation Macro-F1",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savetxt(output / "membership.csv", np.column_stack([np.arange(len(membership)), membership]), delimiter=",", header="global_node_id,client_id", comments="", fmt="%d")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
