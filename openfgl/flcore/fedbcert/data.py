from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    Planetoid,
    Twitch,
    WikipediaNetwork,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_undirected,
)


def _load_official_ogb_pyg_dataset(
    dataset_class,
    *,
    name: str,
    root: str,
):
    """Load an official OGB PyG cache under PyTorch >=2.6.

    Older OGB releases call ``torch.load(processed_path)`` without spelling
    out ``weights_only=False``.  PyTorch 2.6 changed that default to ``True``,
    which rejects the legitimate PyG ``Data`` objects stored in OGB's own
    processed cache.  Restrict the compatibility override to construction of
    the explicitly requested official OGB dataset; model states and immutable
    experiment bundles elsewhere keep their stricter ``weights_only=True``
    loading path.

    The processed file is a pickle and must therefore come from the trusted
    official OGB download/cache selected by the caller.
    """
    original_load = torch.load

    def trusted_ogb_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    with patch.object(torch, "load", trusted_ogb_load):
        return dataset_class(name=name, root=root)


SCENARIOS: Dict[str, Tuple[str, ...]] = {
    "webkb3": ("Texas", "Cornell", "Wisconsin"),
    "twitch6": (
        "Twitch-DE",
        "Twitch-EN",
        "Twitch-ES",
        "Twitch-FR",
        "Twitch-PT",
        "Twitch-RU",
    ),
    "mix6": (
        "Cora",
        "CiteSeer",
        "PubMed",
        "Texas",
        "Cornell",
        "Wisconsin",
    ),
    "mix10": (
        "Cora",
        "CiteSeer",
        "PubMed",
        "Computers",
        "Photo",
        "Chameleon",
        "Squirrel",
        "Actor",
        "Texas",
        "Cornell",
    ),
}

# Stable identities used only by explicitly dataset-keyed experiments.
# The Mix10 ordering is preserved exactly, which keeps its historical random
# streams unchanged. Wisconsin follows at 10; Twitch uses its own disjoint
# scenario-local ordering because Twitch and mixed datasets never co-occur.
CANONICAL_DATASET_RNG_KEYS: Dict[str, int] = {
    dataset_name: index
    for index, dataset_name in enumerate(SCENARIOS["mix10"])
}
CANONICAL_DATASET_RNG_KEYS["Wisconsin"] = 10
CANONICAL_DATASET_RNG_KEYS.update(
    {
        dataset_name: index
        for index, dataset_name in enumerate(SCENARIOS["twitch6"])
    }
)


def dataset_rng_key(dataset_name: str) -> int:
    name = str(dataset_name)
    if name in CANONICAL_DATASET_RNG_KEYS:
        return int(CANONICAL_DATASET_RNG_KEYS[name])
    # Keep unknown future datasets deterministic and outside reserved keys.
    return 1_000 + int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:4],
        byteorder="big",
        signed=False,
    )


WEBKB_SOURCE_URLS: Dict[str, Dict[str, str]] = {
    name: {
        "nodes": (
            "https://raw.githubusercontent.com/"
            "graphdml-uiuc-jlu/geom-gcn/master/"
            f"new_data/{name.lower()}/out1_node_feature_label.txt"
        ),
        "edges": (
            "https://raw.githubusercontent.com/"
            "graphdml-uiuc-jlu/geom-gcn/master/"
            f"new_data/{name.lower()}/out1_graph_edges.txt"
        ),
    }
    for name in ("Texas", "Cornell", "Wisconsin")
}


@dataclass
class ClientGraph:
    client_id: int
    dataset_name: str
    data: Data
    train_mask: torch.Tensor
    calibration_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    homophily: float
    heterophily: float
    label_high_frequency_ratio: float
    split_digest: str

    @property
    def num_classes(self) -> int:
        return int(self.data.y.max().item()) + 1

    @property
    def num_features(self) -> int:
        return int(self.data.x.size(1))

    @property
    def train_count(self) -> int:
        return int(self.train_mask.sum().item())


def scenario_datasets(name: str) -> Tuple[str, ...]:
    key = str(name).strip().lower()
    if key not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario={name}; choices={sorted(SCENARIOS)}"
        )
    return SCENARIOS[key]


def _webkb_raw_dir(
    dataset_root: str,
    name: str,
) -> Path:
    # Keep the same on-disk layout previously created by PyG WebKB:
    # data/fedbcert/Cornell/cornell/raw/
    return Path(dataset_root) / name.lower() / "raw"


def _nonempty_file(path: Path) -> bool:
    return bool(path.is_file() and path.stat().st_size > 0)


def _manual_webkb_error(
    name: str,
    node_path: Path,
    edge_path: Path,
) -> RuntimeError:
    urls = WEBKB_SOURCE_URLS[name]
    return RuntimeError(
        "\n".join(
            [
                f"Manual WebKB files are incomplete for {name}.",
                "The reproducibility loader intentionally does not download the ten "
                "Geom-GCN split files because it creates its own four-way split.",
                "",
                "Required local files:",
                f"  {node_path}",
                f"  {edge_path}",
                "",
                "Open these two addresses in the Windows browser and use "
                "'Save page as' without changing the filenames:",
                f"  {urls['nodes']}",
                f"  {urls['edges']}",
                "",
                "Zero-byte files are treated as incomplete and must be replaced.",
            ]
        )
    )


def _parse_webkb_nodes(
    path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    lines = [
        line.rstrip("\r\n")
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"Empty WebKB node file: {path}")

    rows = lines[1:]
    node_ids: List[int] = []
    features: List[List[float]] = []
    labels: List[int] = []

    for line_number, line in enumerate(rows, start=2):
        parts = line.split("\t")
        if len(parts) != 3:
            raise RuntimeError(
                f"Malformed WebKB node line {line_number} in {path}: "
                "expected tab-separated node_id, feature, label. "
                "Use browser 'Save page as'; do not copy visible text."
            )
        node_id = int(parts[0])
        vector = [
            float(value)
            for value in parts[1].split(",")
        ]
        label = int(parts[2])
        node_ids.append(node_id)
        features.append(vector)
        labels.append(label)

    if len(set(node_ids)) != len(node_ids):
        raise RuntimeError(
            f"Duplicate node ids in {path}"
        )
    feature_dims = {
        len(vector)
        for vector in features
    }
    if len(feature_dims) != 1:
        raise RuntimeError(
            f"Inconsistent feature dimensions in {path}: "
            f"{sorted(feature_dims)}"
        )

    id_to_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }
    return (
        torch.tensor(
            features,
            dtype=torch.float32,
        ),
        torch.tensor(
            labels,
            dtype=torch.long,
        ),
        id_to_index,
    )


def _parse_webkb_edges(
    path: Path,
    id_to_index: Dict[int, int],
) -> torch.Tensor:
    lines = [
        line.rstrip("\r\n")
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"Empty WebKB edge file: {path}")

    edges: List[List[int]] = []
    for line_number, line in enumerate(
        lines[1:],
        start=2,
    ):
        parts = line.split("\t")
        if len(parts) != 2:
            raise RuntimeError(
                f"Malformed WebKB edge line {line_number} in {path}: "
                "expected two tab-separated node ids."
            )
        source_id = int(parts[0])
        target_id = int(parts[1])
        if (
            source_id not in id_to_index
            or target_id not in id_to_index
        ):
            raise RuntimeError(
                f"Unknown node id at edge line {line_number} in {path}: "
                f"{source_id}, {target_id}"
            )
        edges.append(
            [
                id_to_index[source_id],
                id_to_index[target_id],
            ]
        )

    if not edges:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
        )
    return torch.tensor(
        edges,
        dtype=torch.long,
    ).t().contiguous()


def load_manual_webkb(
    name: str,
    dataset_root: str,
) -> Data:
    if name not in WEBKB_SOURCE_URLS:
        raise ValueError(
            f"Unsupported WebKB name: {name}"
        )

    raw_dir = _webkb_raw_dir(
        dataset_root,
        name,
    )
    node_path = (
        raw_dir
        / "out1_node_feature_label.txt"
    )
    edge_path = (
        raw_dir
        / "out1_graph_edges.txt"
    )

    if not (
        _nonempty_file(node_path)
        and _nonempty_file(edge_path)
    ):
        raise _manual_webkb_error(
            name,
            node_path,
            edge_path,
        )

    x, y, id_to_index = (
        _parse_webkb_nodes(node_path)
    )
    edge_index = _parse_webkb_edges(
        edge_path,
        id_to_index,
    )
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )
    return NormalizeFeatures()(data)


def load_cross_network_dataset(
    name: str,
    root: str,
) -> Data:
    transform = NormalizeFeatures()
    normalized = str(name).strip()
    lower = normalized.lower()
    dataset_root = str(
        Path(root)
        / normalized.replace("/", "_")
    )

    if normalized in {
        "Cora",
        "CiteSeer",
        "PubMed",
    }:
        cache_path = (
            Path(dataset_root)
            / "linqs_data.pt"
        )
        if not cache_path.is_file():
            raise RuntimeError(
                "Missing offline LINQS cache: "
                f"{cache_path}. Run "
                "python scripts/"
                "prepare_linqs_data.py "
                "--download"
            )
        try:
            payload = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(
                cache_path,
                map_location="cpu",
            )
        data = payload["data"]
        data = transform(data)
    elif normalized in {
        "Computers",
        "Photo",
    }:
        dataset = Amazon(
            root=dataset_root,
            name=normalized,
            transform=transform,
        )
        data = dataset[0]
    elif normalized == "CS":
        dataset = Coauthor(
            root=dataset_root,
            name="CS",
            transform=transform,
        )
        data = dataset[0]
    elif normalized == "ogbn-arxiv":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as error:
            raise RuntimeError(
                "ogbn-arxiv requires the optional 'ogb' package"
            ) from error
        dataset = _load_official_ogb_pyg_dataset(
            PygNodePropPredDataset,
            name="ogbn-arxiv",
            root=dataset_root,
        )
        data = dataset[0]
        split = dataset.get_idx_split()
        for split_name, attribute in (
            ("train", "official_train_mask"),
            ("valid", "official_validation_mask"),
            ("test", "official_test_mask"),
        ):
            mask = torch.zeros(int(data.num_nodes), dtype=torch.bool)
            mask[split[split_name].reshape(-1).long()] = True
            setattr(data, attribute, mask)
    elif normalized in {
        "Chameleon",
        "Squirrel",
    }:
        dataset = WikipediaNetwork(
            root=dataset_root,
            name=lower,
            geom_gcn_preprocess=True,
            transform=transform,
        )
        data = dataset[0]
    elif normalized == "Actor":
        dataset = Actor(
            root=dataset_root,
            transform=transform,
        )
        data = dataset[0]
    elif normalized in {
        "Texas",
        "Cornell",
        "Wisconsin",
    }:
        data = load_manual_webkb(
            normalized,
            dataset_root,
        )
    elif lower.startswith("twitch-"):
        language = normalized.split("-", 1)[1]
        dataset = Twitch(
            root=dataset_root,
            name=language,
            transform=transform,
        )
        data = dataset[0]
    else:
        raise ValueError(
            "Unsupported cross-network dataset: "
            f"{name}"
        )

    data.x = data.x.float()
    data.y = data.y.reshape(-1).long()
    edge_index = to_undirected(
        data.edge_index
    )
    edge_index, _ = remove_self_loops(
        edge_index
    )
    data.edge_index = coalesce(
        edge_index,
        num_nodes=int(data.num_nodes),
    )
    for key in (
        "edge_attr",
        "edge_weight",
        "edge_type",
    ):
        if key in data:
            del data[key]
    return data


def _allocate_counts(
    class_count: int,
    ratios: Sequence[float],
) -> List[int]:
    if class_count <= 0:
        return [0 for _ in ratios]
    raw = [
        float(value) * class_count
        for value in ratios
    ]
    counts = [
        int(value)
        for value in raw
    ]
    remaining = int(
        class_count - sum(counts)
    )
    order = sorted(
        range(len(ratios)),
        key=lambda index: (
            raw[index] - counts[index]
        ),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1

    if class_count >= len(ratios):
        for index in range(len(counts)):
            if counts[index] == 0:
                donor = max(
                    range(len(counts)),
                    key=lambda item: counts[item],
                )
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return counts


def stratified_four_way_masks(
    labels: torch.Tensor,
    *,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    ratios = (
        float(train_ratio),
        float(calibration_ratio),
        float(val_ratio),
        float(test_ratio),
    )
    if any(value <= 0.0 for value in ratios):
        raise ValueError(
            "all four split ratios must be positive"
        )
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(
            "split ratios must sum to one"
        )

    labels = (
        labels.detach()
        .cpu()
        .reshape(-1)
        .long()
    )
    masks = [
        torch.zeros(
            labels.numel(),
            dtype=torch.bool,
        )
        for _ in range(4)
    ]
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    for class_id in torch.unique(
        labels
    ).tolist():
        indices = torch.where(
            labels == int(class_id)
        )[0]
        permutation = torch.randperm(
            indices.numel(),
            generator=generator,
        )
        indices = indices[permutation]
        counts = _allocate_counts(
            int(indices.numel()),
            ratios,
        )
        offset = 0
        for split_index, count in enumerate(
            counts
        ):
            selected = indices[
                offset : offset + count
            ]
            masks[split_index][selected] = True
            offset += count

    coverage = (
        masks[0]
        | masks[1]
        | masks[2]
        | masks[3]
    )
    if not bool(coverage.all()):
        missing = torch.where(~coverage)[0]
        masks[-1][missing] = True
    for left in range(4):
        for right in range(
            left + 1,
            4,
        ):
            if bool(
                (
                    masks[left]
                    & masks[right]
                ).any()
            ):
                raise RuntimeError(
                    "split masks overlap"
                )
    return tuple(masks)  # type: ignore[return-value]


def edge_homophily(data: Data) -> float:
    if data.edge_index.numel() == 0:
        return 0.0
    source, target = data.edge_index
    return float(
        (
            data.y[source]
            == data.y[target]
        )
        .float()
        .mean()
        .item()
    )


def label_high_frequency_ratio(
    data: Data,
) -> float:
    if data.edge_index.numel() == 0:
        return 0.0
    one_hot = torch.nn.functional.one_hot(
        data.y,
        num_classes=(
            int(data.y.max().item()) + 1
        ),
    ).float()
    source, target = data.edge_index
    numerator = (
        (
            one_hot[source]
            - one_hot[target]
        )
        .pow(2)
        .sum(dim=1)
        .mean()
    )
    denominator = (
        one_hot.pow(2)
        .sum(dim=1)
        .mean()
        .clamp_min(1e-12)
    )
    return float(
        (numerator / denominator).item()
    )


def split_digest(
    *masks: torch.Tensor,
) -> str:
    hasher = hashlib.sha256()
    for mask in masks:
        indices = torch.where(
            mask.detach().cpu()
        )[0].numpy().tobytes()
        hasher.update(indices)
        hasher.update(b"|")
    return hasher.hexdigest()[:16]


def build_scenario_clients(
    scenario: str,
    *,
    data_root: str,
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng_key_mode: str = "client",
) -> List[ClientGraph]:
    resolved_rng_key_mode = str(rng_key_mode).strip().lower()
    if resolved_rng_key_mode not in {"client", "dataset"}:
        raise ValueError("rng_key_mode must be client or dataset")
    clients: List[ClientGraph] = []
    for client_id, dataset_name in enumerate(
        scenario_datasets(scenario)
    ):
        data = load_cross_network_dataset(
            dataset_name,
            data_root,
        )
        if resolved_rng_key_mode == "dataset":
            rng_key = dataset_rng_key(dataset_name)
        else:
            rng_key = int(client_id)
        masks = stratified_four_way_masks(
            data.y,
            train_ratio=train_ratio,
            calibration_ratio=(
                calibration_ratio
            ),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=(
                int(seed) * 100003
                + rng_key * 1009
                + 17
            ),
        )
        homophily = edge_homophily(data)
        clients.append(
            ClientGraph(
                client_id=client_id,
                dataset_name=dataset_name,
                data=data.cpu(),
                train_mask=masks[0],
                calibration_mask=masks[1],
                val_mask=masks[2],
                test_mask=masks[3],
                homophily=homophily,
                heterophily=(
                    1.0 - homophily
                ),
                label_high_frequency_ratio=(
                    label_high_frequency_ratio(
                        data
                    )
                ),
                split_digest=(
                    split_digest(*masks)
                ),
            )
        )
    return clients


def save_split_manifest(
    clients: Sequence[ClientGraph],
    path: Path,
) -> None:
    rows = []
    for client in clients:
        rows.append(
            {
                "client_id": int(
                    client.client_id
                ),
                "dataset": (
                    client.dataset_name
                ),
                "num_nodes": int(
                    client.data.num_nodes
                ),
                "num_edges": int(
                    client.data.edge_index.size(
                        1
                    )
                ),
                "num_features": int(
                    client.num_features
                ),
                "num_classes": int(
                    client.num_classes
                ),
                "train_count": int(
                    client.train_mask.sum()
                ),
                "calibration_count": int(
                    client.calibration_mask.sum()
                ),
                "val_count": int(
                    client.val_mask.sum()
                ),
                "test_count": int(
                    client.test_mask.sum()
                ),
                "homophily": float(
                    client.homophily
                ),
                "heterophily": float(
                    client.heterophily
                ),
                "label_high_frequency_ratio": float(
                    client.label_high_frequency_ratio
                ),
                "split_digest": (
                    client.split_digest
                ),
            }
        )
    path.write_text(
        json.dumps(
            rows,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
