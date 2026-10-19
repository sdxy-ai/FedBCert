#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.request import urlopen

import torch
from torch_geometric.data import Data


SOURCES = {
    "Cora": {
        "url": (
            "https://linqs-data.soe.ucsc.edu/"
            "public/lbc/cora.tgz"
        ),
        "archive": "cora.tgz",
        "node_glob": "cora.content",
        "edge_glob": "cora.cites",
        "expected": (2708, 1433, 7),
        "variant": "LINQS raw Cora",
    },
    "CiteSeer": {
        "url": (
            "https://linqs-data.soe.ucsc.edu/"
            "public/lbc/citeseer.tgz"
        ),
        "archive": "citeseer.tgz",
        "node_glob": "citeseer.content",
        "edge_glob": "citeseer.cites",
        "expected": (3312, 3703, 6),
        "variant": "LINQS raw CiteSeer",
    },
    "PubMed": {
        "url": (
            "https://linqs-data.soe.ucsc.edu/"
            "public/Pubmed-Diabetes.tgz"
        ),
        "archive": "Pubmed-Diabetes.tgz",
        "node_glob": (
            "Pubmed-Diabetes.NODE.paper.tab"
        ),
        "edge_glob": (
            "Pubmed-Diabetes.DIRECTED.cites.tab"
        ),
        "expected": (19717, 500, 3),
        "variant": "LINQS PubMed Diabetes",
    },
}


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {filename} under {root}; "
            f"found {matches}"
        )
    return matches[0]


def download(url: str, archive: Path) -> None:
    archive.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    print(f"[DOWNLOAD] {url}", flush=True)
    with urlopen(url, timeout=60) as source, archive.open("wb") as target:
        shutil.copyfileobj(source, target)
    if (
        not archive.is_file()
        or archive.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Download is empty: {archive}"
        )


def extract(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )
    with tarfile.open(
        archive,
        mode="r:gz",
    ) as handle:
        handle.extractall(destination)


def parse_cora_family(
    node_path: Path,
    edge_path: Path,
) -> Data:
    node_ids: List[str] = []
    features: List[List[float]] = []
    string_labels: List[str] = []

    with node_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 3:
                raise RuntimeError(
                    f"Malformed node line "
                    f"{line_number}: {node_path}"
                )
            node_ids.append(fields[0])
            features.append(
                [
                    float(value)
                    for value in fields[1:-1]
                ]
            )
            string_labels.append(fields[-1])

    if len(set(node_ids)) != len(node_ids):
        raise RuntimeError(
            f"Duplicate node IDs in {node_path}"
        )

    dimensions = {
        len(row)
        for row in features
    }
    if len(dimensions) != 1:
        raise RuntimeError(
            f"Inconsistent feature dimensions: "
            f"{sorted(dimensions)}"
        )

    label_names = sorted(set(string_labels))
    label_map = {
        value: index
        for index, value in enumerate(label_names)
    }
    id_map = {
        value: index
        for index, value in enumerate(node_ids)
    }

    edges: List[Tuple[int, int]] = []
    skipped = 0
    with edge_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            source_id = fields[0]
            target_id = fields[1]
            if (
                source_id not in id_map
                or target_id not in id_map
            ):
                skipped += 1
                continue
            edges.append(
                (
                    id_map[source_id],
                    id_map[target_id],
                )
            )

    print(
        f"[INFO] {node_path.name}: "
        f"filtered_unknown_edges={skipped}"
    )

    edge_index = (
        torch.tensor(
            edges,
            dtype=torch.long,
        )
        .t()
        .contiguous()
        if edges
        else torch.empty(
            (2, 0),
            dtype=torch.long,
        )
    )

    return Data(
        x=torch.tensor(
            features,
            dtype=torch.float32,
        ),
        y=torch.tensor(
            [
                label_map[value]
                for value in string_labels
            ],
            dtype=torch.long,
        ),
        edge_index=edge_index,
    )


def feature_pairs(
    fields: Iterable[str],
) -> Iterable[Tuple[str, float]]:
    for field in fields:
        if "=" not in field:
            continue
        key, raw_value = field.split(
            "=",
            maxsplit=1,
        )
        if not key.startswith("w-"):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        yield key, value


def parse_pubmed_nodes_first_pass(
    node_path: Path,
) -> Tuple[List[str], List[int], List[str]]:
    paper_ids: List[str] = []
    labels: List[int] = []
    feature_names = set()

    with node_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        next(handle, None)
        next(handle, None)

        for line_number, line in enumerate(
            handle,
            start=3,
        ):
            fields = line.rstrip(
                "\r\n"
            ).split("\t")

            if len(fields) < 3:
                raise RuntimeError(
                    f"Malformed PubMed node line "
                    f"{line_number}"
                )

            paper_ids.append(fields[0])

            if "=" not in fields[1]:
                raise RuntimeError(
                    f"Missing PubMed label at "
                    f"line {line_number}"
                )

            labels.append(
                int(
                    fields[1].split(
                        "=",
                        maxsplit=1,
                    )[1]
                )
                - 1
            )

            for key, _ in feature_pairs(
                fields[2:]
            ):
                feature_names.add(key)

    return (
        paper_ids,
        labels,
        sorted(feature_names),
    )


def parse_pubmed(
    node_path: Path,
    edge_path: Path,
) -> Data:
    (
        paper_ids,
        labels,
        feature_names,
    ) = parse_pubmed_nodes_first_pass(
        node_path
    )

    if len(set(paper_ids)) != len(paper_ids):
        raise RuntimeError(
            "Duplicate PubMed paper IDs"
        )

    paper_map = {
        value: index
        for index, value in enumerate(paper_ids)
    }
    feature_map = {
        value: index
        for index, value in enumerate(feature_names)
    }

    x = torch.zeros(
        (
            len(paper_ids),
            len(feature_names),
        ),
        dtype=torch.float32,
    )

    with node_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        next(handle, None)
        next(handle, None)

        for row_index, line in enumerate(
            handle
        ):
            fields = line.rstrip(
                "\r\n"
            ).split("\t")

            for key, value in feature_pairs(
                fields[2:]
            ):
                x[
                    row_index,
                    feature_map[key],
                ] = value

    edges: List[Tuple[int, int]] = []
    skipped = 0

    with edge_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        next(handle, None)
        next(handle, None)

        for line in handle:
            fields = line.rstrip(
                "\r\n"
            ).split("\t")

            if len(fields) < 4:
                continue

            try:
                source_id = fields[1].split(
                    ":",
                    maxsplit=1,
                )[1]
                target_id = fields[3].split(
                    ":",
                    maxsplit=1,
                )[1]
            except (
                IndexError,
                ValueError,
            ):
                continue

            if (
                source_id not in paper_map
                or target_id not in paper_map
            ):
                skipped += 1
                continue

            edges.append(
                (
                    paper_map[source_id],
                    paper_map[target_id],
                )
            )

    print(
        "[INFO] PubMed: "
        f"filtered_unknown_edges={skipped}"
    )

    edge_index = (
        torch.tensor(
            edges,
            dtype=torch.long,
        )
        .t()
        .contiguous()
        if edges
        else torch.empty(
            (2, 0),
            dtype=torch.long,
        )
    )

    return Data(
        x=x,
        y=torch.tensor(
            labels,
            dtype=torch.long,
        ),
        edge_index=edge_index,
    )


def remove_broken_target(path: Path) -> None:
    if path.is_symlink():
        print(f"[REMOVE SYMLINK] {path}")
        path.unlink()
    elif path.exists() and not path.is_dir():
        raise RuntimeError(
            f"Expected directory: {path}"
        )


def validate(name: str, data: Data) -> None:
    expected_nodes, expected_features, expected_classes = (
        SOURCES[name]["expected"]
    )

    actual = (
        int(data.num_nodes),
        int(data.num_node_features),
        int(data.y.max().item()) + 1,
    )
    expected = (
        expected_nodes,
        expected_features,
        expected_classes,
    )

    if actual != expected:
        raise RuntimeError(
            f"{name} validation failed: "
            f"actual={actual}, expected={expected}"
        )

    if data.edge_index.numel() == 0:
        raise RuntimeError(
            f"{name} has no edges"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="data/fedbcert",
    )
    parser.add_argument(
        "--download",
        action="store_true",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SOURCES),
        default=tuple(SOURCES),
        help="LINQS datasets to prepare (default: all three)",
    )
    parser.add_argument(
        "--archive-dir",
        default=(
            "data/fedbcert/"
            "_linqs_archives"
        ),
    )
    parser.add_argument(
        "--extract-dir",
        default=(
            "data/fedbcert/"
            "_linqs_extracted"
        ),
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    archive_root = Path(args.archive_dir)
    extract_root = Path(args.extract_dir)

    data_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in args.datasets:
        spec = SOURCES[name]
        archive = (
            archive_root
            / str(spec["archive"])
        )
        extracted = extract_root / name

        if args.download:
            download(
                str(spec["url"]),
                archive,
            )
        elif not archive.is_file():
            raise RuntimeError(
                f"Missing {archive}. Run with "
                "--download or copy the archive "
                "there manually."
            )

        print(
            f"\n[EXTRACT] {name}: {archive}"
        )

        extract(archive, extracted)

        node_path = find_one(
            extracted,
            str(spec["node_glob"]),
        )
        edge_path = find_one(
            extracted,
            str(spec["edge_glob"]),
        )

        print(f"[PARSE] nodes={node_path}")
        print(f"[PARSE] edges={edge_path}")

        if name in {"Cora", "CiteSeer"}:
            data = parse_cora_family(
                node_path,
                edge_path,
            )
        else:
            data = parse_pubmed(
                node_path,
                edge_path,
            )

        validate(name, data)

        target_dir = data_root / name
        remove_broken_target(target_dir)

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = (
            target_dir
            / "linqs_data.pt"
        )

        torch.save(
            {
                "data": data,
                "dataset": name,
                "source_url": str(
                    spec["url"]
                ),
                "source_variant": str(
                    spec["variant"]
                ),
            },
            target,
        )

        print(
            f"[SAVED] {target} "
            f"nodes={data.num_nodes} "
            f"edges={data.num_edges} "
            f"features="
            f"{data.num_node_features} "
            f"classes="
            f"{int(data.y.max().item()) + 1}"
        )

    print(
        "\nPASS: all LINQS caches prepared"
    )


if __name__ == "__main__":
    main()
