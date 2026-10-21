"""Single-process aggregate-only RPC replay, not a real network deployment.

The service owns a private partitioned graph fixture. Policies see only their
local view and explicitly requested metadata. Every cross-owner dependency in
an exact fallback is serialized, decoded, and charged before the receiver uses
its vector. No full-graph hidden matrix or full-graph logits are precomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
import time
from typing import Iterable, Literal

import numpy as np
import torch

from .certificate import PolynomialSpectralClassifier, canonical_undirected, row_normalized_shift
from .model_contract import model_fingerprint, validate_stock_model


_HEADER = struct.Struct("<8sB32sIqqI")
_DIM = struct.Struct("<I")
_VERSION = 1
_REQUEST = b"FBCREQ01"
_RESPONSE = b"FBCRES01"
_META_REQUEST = b"FBCMRQ01"
_META_RESPONSE = b"FBCMRS01"


@dataclass(frozen=True)
class LocalView:
    nodes: torch.Tensor
    features: torch.Tensor
    shift: torch.Tensor
    retained_degree: torch.Tensor


@dataclass(frozen=True)
class MetadataReceipt:
    kind: str
    nodes: torch.Tensor
    values: torch.Tensor
    request_bytes: int
    response_bytes: int
    payload_bytes: int
    snapshot_id: str
    request_payload: bytes
    response_payload: bytes

    @property
    def wire_bytes(self) -> int:
        return self.request_bytes + self.response_bytes


@dataclass(frozen=True)
class NominalResult:
    nodes: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor
    feature_logits: torch.Tensor | None
    encoded_rows: int
    inference_seconds: float


@dataclass(frozen=True)
class FallbackResult:
    nodes: torch.Tensor
    logits: torch.Tensor
    accounting: dict
    snapshot_id: str


def _tensor_hash(value: torch.Tensor) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.digest()


def state_fingerprint(model: PolynomialSpectralClassifier) -> str:
    """Bind the shared validated stock inference contract, not just tensors."""
    return model_fingerprint(model)


def _pack_header(magic: bytes, snapshot: str, stage: int, source: int, receiver: int, count: int) -> bytes:
    return _HEADER.pack(magic, _VERSION, bytes.fromhex(snapshot), stage, source, receiver, count)


def _read_header(
    payload: bytes, magic: bytes, snapshot: str, stage: int, source: int, receiver: int,
) -> int:
    if len(payload) < _HEADER.size:
        raise ValueError("truncated RPC header")
    actual = _HEADER.unpack_from(payload)
    expected = (magic, _VERSION, bytes.fromhex(snapshot), stage, source, receiver)
    if actual[:6] != expected:
        raise ValueError("RPC snapshot, stage, or owner header mismatch")
    return int(actual[6])


def _node_bytes(nodes: Iterable[int]) -> bytes:
    return np.asarray(list(nodes), dtype="<u8").tobytes()


def _decode_request(payload: bytes, snapshot: str, stage: int, source: int, receiver: int) -> list[int]:
    count = _read_header(payload, _REQUEST, snapshot, stage, source, receiver)
    if len(payload) != _HEADER.size + 8 * count:
        raise ValueError("RPC request length mismatch")
    nodes = np.frombuffer(payload, dtype="<u8", count=count, offset=_HEADER.size).tolist()
    if nodes != sorted(set(nodes)):
        raise ValueError("RPC node IDs must be sorted and unique")
    return [int(node) for node in nodes]


def decode_vector_response(
    payload: bytes, *, snapshot_id: str, stage: int, source_owner: int,
    receiver_owner: int, expected_nodes: Iterable[int], width: int,
) -> torch.Tensor:
    """Validate wire identity and return a copied CPU float64 vector matrix."""
    nodes = list(expected_nodes)
    count = _read_header(payload, _RESPONSE, snapshot_id, stage, source_owner, receiver_owner)
    if len(payload) < _HEADER.size + _DIM.size:
        raise ValueError("truncated vector dimension")
    dimension = _DIM.unpack_from(payload, _HEADER.size)[0]
    expected_length = _HEADER.size + _DIM.size + 8 * count + 8 * count * dimension
    if count != len(nodes) or dimension != width or len(payload) != expected_length:
        raise ValueError("RPC response length, node count, or width mismatch")
    offset = _HEADER.size + _DIM.size
    received_nodes = np.frombuffer(payload, dtype="<u8", count=count, offset=offset).tolist()
    if received_nodes != nodes or received_nodes != sorted(set(received_nodes)):
        raise ValueError("RPC response node IDs mismatch")
    array = np.frombuffer(payload, dtype="<f8", count=count * width, offset=offset + 8 * count)
    if not np.isfinite(array).all():
        raise ValueError("RPC response has non-finite vectors")
    return torch.from_numpy(array.copy().reshape(count, width))


def decode_metadata_response(
    payload: bytes, *, snapshot_id: str, owner: int, kind: str,
    expected_nodes: Iterable[int],
) -> torch.Tensor:
    """Validate and decode the authoritative fixture's requested summary."""
    if kind not in {"upper_count", "presence"}:
        raise ValueError("unknown metadata kind")
    nodes = list(expected_nodes)
    count = _read_header(payload, _META_RESPONSE, snapshot_id, 0, -1, int(owner))
    value_bytes = 4 * count if kind == "upper_count" else (count + 7) // 8
    if len(payload) != _HEADER.size + 1 + 8 * count + value_bytes or count != len(nodes):
        raise ValueError("metadata response length mismatch")
    expected_kind = 0 if kind == "upper_count" else 1
    if payload[_HEADER.size] != expected_kind:
        raise ValueError("metadata kind header mismatch")
    offset = _HEADER.size + 1
    received_nodes = np.frombuffer(payload, dtype="<u8", count=count, offset=offset).tolist()
    if received_nodes != nodes or received_nodes != sorted(set(received_nodes)):
        raise ValueError("metadata node IDs mismatch")
    offset += 8 * count
    if kind == "upper_count":
        values = np.frombuffer(payload, dtype="<u4", count=count, offset=offset).astype(np.int64)
    else:
        values = np.unpackbits(np.frombuffer(payload, dtype=np.uint8, offset=offset), bitorder="little")
        if bool(values[count:].any()):
            raise ValueError("nonzero metadata bit padding")
        values = values[:count].astype(np.bool_)
    return torch.from_numpy(values.copy())


def potential_exposure(shift: torch.Tensor, boundary_flags: torch.Tensor, order_count: int) -> torch.Tensor:
    """Reachability of changed local rows for powers zero through K-1.

    A changed row first affects its own node at power one, so K=3 includes
    boundary nodes and their one-hop retained predecessors, not two hops.
    """
    if not isinstance(order_count, int) or isinstance(order_count, bool) or order_count < 1:
        raise ValueError("order_count must be a positive integer")
    if shift.shape != (boundary_flags.numel(), boundary_flags.numel()) or boundary_flags.ndim != 1:
        raise ValueError("shift and boundary flags must be node aligned")
    if boundary_flags.dtype != torch.bool or boundary_flags.device != shift.device:
        raise ValueError("boundary_flags must be boolean and on the shift device")
    if shift.layout not in {torch.sparse_coo, torch.sparse_csr}:
        raise ValueError("shift must be sparse")
    coo = shift.to_sparse_coo().coalesce()
    if not torch.isfinite(coo.values()).all() or bool((coo.values() < 0).any()):
        raise ValueError("shift must be finite and nonnegative")
    exposed = torch.zeros_like(boundary_flags)
    if order_count == 1:
        return exposed
    exposed = boundary_flags.clone()
    for _ in range(order_count - 2):
        reached = torch.sparse.mm(shift, exposed.to(shift.dtype)[:, None]).squeeze(1) > 0
        exposed |= reached
    return exposed


class AggregateReplayService:
    """Partition-owner aggregate replay with immutable model/data identity.

    Float32 is supported for regression checks; the deployment-style replay
    runner uses CPU float64. Vector messages always encode float64 on the wire.
    Metadata is authoritative only inside this synthetic replay fixture.
    """

    def __init__(
        self, model: PolynomialSpectralClassifier, features: torch.Tensor,
        edge_index: torch.Tensor, membership: torch.Tensor | np.ndarray,
    ) -> None:
        validate_stock_model(model)
        if features.ndim != 2 or features.size(0) == 0:
            raise ValueError("features must be a nonempty matrix")
        if features.device.type != "cpu" or features.dtype not in {torch.float32, torch.float64}:
            raise ValueError("aggregate replay requires CPU float32 or float64 features")
        if features.dtype != model.theta.dtype or any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("model and features must use the same CPU dtype")
        if any(p.dtype != features.dtype for p in model.parameters()):
            raise ValueError("model parameter dtype mismatch")
        if not bool(torch.isfinite(features).all()) or any(
            not bool(torch.isfinite(p).all()) for p in model.parameters()
        ):
            raise ValueError("features and model must be finite")
        if edge_index.ndim != 2 or edge_index.size(0) != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a 2-by-E int64 tensor")
        if edge_index.device.type != "cpu":
            raise ValueError("edge_index must be on CPU")
        member = torch.as_tensor(membership)
        if member.dtype not in {torch.int32, torch.int64} or member.device.type != "cpu":
            raise ValueError("membership must contain CPU integer owners")
        if member.shape != (features.size(0),) or bool((member < 0).any()):
            raise ValueError("membership must assign a nonnegative owner to every node")
        if edge_index.numel() and bool(((edge_index < 0) | (edge_index >= features.size(0))).any()):
            raise ValueError("edge_index is out of range")
        self._model = model
        self._features = features.detach()
        self._input_edges = edge_index.detach()
        self._membership = member
        self._dtype = features.dtype
        self._count = features.size(0)
        self._edges = canonical_undirected(edge_index, self._count)
        adjacency: list[set[int]] = [{v} for v in range(self._count)]
        for u, v in self._edges.t().tolist():
            adjacency[int(v)].add(int(u))
        self._neighbors = [tuple(sorted(values)) for values in adjacency]
        self._owners = [int(owner) for owner in member.tolist()]
        self._owner_nodes = {
            int(owner): torch.where(member == owner)[0] for owner in member.unique(sorted=True).tolist()
        }
        self._width = int(model.classifier.in_features)
        self._classes = int(model.classifier.out_features)
        self._order = int(model.order_count)
        if self._order < 1 or model.theta.numel() != self._order:
            raise ValueError("polynomial order mismatch")
        self.snapshot_model_hash = self._model_hash()
        self.snapshot_data_hash = self._data_hash()
        self.snapshot_id = hashlib.sha256(
            bytes.fromhex(self.snapshot_model_hash) + bytes.fromhex(self.snapshot_data_hash)
        ).hexdigest()
        self._nominal_cache: dict[int, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = {}

    def _model_hash(self) -> str:
        return state_fingerprint(self._model)

    def _data_hash(self) -> str:
        digest = hashlib.sha256()
        for value in (self._features, self._input_edges, self._membership):
            digest.update(_tensor_hash(value))
        return digest.hexdigest()

    def _check_snapshot(self) -> None:
        if any(module.training for module in self._model.modules()):
            raise ValueError("model left eval mode")
        if self._model_hash() != self.snapshot_model_hash or self._data_hash() != self.snapshot_data_hash:
            raise ValueError("model or data changed after snapshot creation")

    def _nodes_for_owner(self, owner: int) -> torch.Tensor:
        if int(owner) not in self._owner_nodes:
            raise ValueError("unknown owner")
        return self._owner_nodes[int(owner)]

    def _local_view(self, owner: int) -> LocalView:
        nodes = self._nodes_for_owner(owner)
        index = torch.full((self._count,), -1, dtype=torch.long)
        index[nodes] = torch.arange(nodes.numel())
        keep = (self._membership[self._edges[0]] == owner) & (self._membership[self._edges[1]] == owner)
        local_edges = index[self._edges[:, keep]]
        shift, degree = row_normalized_shift(
            local_edges, nodes.numel(), device=torch.device("cpu"), dtype=self._dtype,
        )
        return LocalView(nodes.clone(), self._features[nodes].clone(), shift, degree)

    def local_view(self, owner: int) -> LocalView:
        self._check_snapshot()
        return self._local_view(int(owner))

    def metadata(self, owner: int, kind: Literal["upper_count", "presence"]) -> MetadataReceipt:
        self._check_snapshot()
        owner = int(owner)
        if kind not in {"upper_count", "presence"}:
            raise ValueError("unknown metadata kind")
        nodes = self._nodes_for_owner(owner)
        counts = np.asarray([
            sum(self._owners[u] != owner for u in self._neighbors[v]) for v in nodes.tolist()
        ], dtype=np.int64)
        kind_byte = bytes([0 if kind == "upper_count" else 1])
        request = _pack_header(_META_REQUEST, self.snapshot_id, 0, -1, owner, 0) + kind_byte
        _read_header(request, _META_REQUEST, self.snapshot_id, 0, -1, owner)
        if kind == "upper_count":
            if bool((counts > np.iinfo(np.uint32).max).any()):
                raise ValueError("count does not fit uint32")
            values = counts.astype("<u4").tobytes()
        else:
            # Presence callers never receive exact counts.
            values = np.packbits(counts > 0, bitorder="little").tobytes()
        response = (
            _pack_header(_META_RESPONSE, self.snapshot_id, 0, -1, owner, nodes.numel())
            + kind_byte + _node_bytes(nodes.tolist()) + values
        )
        decoded = decode_metadata_response(
            response, snapshot_id=self.snapshot_id, owner=owner, kind=kind,
            expected_nodes=nodes.tolist(),
        )
        return MetadataReceipt(
            kind, nodes.clone(), decoded, len(request), len(response), len(values),
            self.snapshot_id, request, response,
        )

    @torch.no_grad()
    def nominal(self, owner: int) -> NominalResult:
        self._check_snapshot()
        owner = int(owner)
        started = time.perf_counter()
        view = self._local_view(owner)
        encoded_rows = 0
        if owner not in self._nominal_cache:
            hidden = self._model.bounded_encoding(view.features)
            spectral = self._model.classifier(self._model.filter(hidden, view.shift))
            if self._model.encoder_type == "dual_feature_spectral":
                feature_logits = self._model.feature_classifier(self._model.feature_encoding(view.features))
                logits = feature_logits + self._model.spectral_fusion_scale * spectral
            else:
                feature_logits = None
                logits = spectral
            if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(logits).all()):
                raise ValueError("non-finite nominal inference")
            self._nominal_cache[owner] = (hidden.clone(), None if feature_logits is None else feature_logits.clone(), logits.clone())
            encoded_rows = view.nodes.numel()
        hidden, feature_logits, logits = self._nominal_cache[owner]
        return NominalResult(
            view.nodes, logits.clone(), hidden.clone(),
            None if feature_logits is None else feature_logits.clone(),
            int(encoded_rows), time.perf_counter() - started,
        )

    def _exchange(
        self, nodes: list[int], stage: int, source: int, receiver: int,
        source_values: dict[int, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], dict]:
        request = _pack_header(_REQUEST, self.snapshot_id, stage, source, receiver, len(nodes)) + _node_bytes(nodes)
        requested = _decode_request(request, self.snapshot_id, stage, source, receiver)
        if any(node >= self._count or self._owners[node] != source for node in requested):
            raise ValueError("RPC requested a node not owned by its source")
        vectors = torch.stack([source_values[node] for node in requested]).to(torch.float64)
        response = (
            _pack_header(_RESPONSE, self.snapshot_id, stage, source, receiver, len(requested))
            + _DIM.pack(self._width) + _node_bytes(requested)
            + vectors.numpy().astype("<f8", copy=False).tobytes()
        )
        decoded = decode_vector_response(
            response, snapshot_id=self.snapshot_id, stage=stage, source_owner=source,
            receiver_owner=receiver, expected_nodes=requested, width=self._width,
        ).to(self._dtype)
        values = {node: decoded[index] for index, node in enumerate(requested)}
        accounting = {
            "source_owner": source, "receiver_owner": receiver,
            "previous_order": stage, "node_ids": requested,
            "remote_vectors": len(requested), "request_response_packets": 2,
            "request_bytes": len(request), "response_bytes": len(response),
            "payload_bytes": 8 * len(requested) * self._width,
            "wire_bytes": len(request) + len(response),
        }
        return values, accounting

    @torch.no_grad()
    def exact(self, targets: Iterable[int] | torch.Tensor) -> FallbackResult:
        """Fetch only required aggregates; discard fallback caches per batch.

        The private backward plan represents recursive owner requests. At
        previous order j, a vector is sent once per (node, receiving owner, j),
        even if several receiving rows need it. Request-response packet count
        counts individual packets, i.e. two per batched RPC. Payload bytes
        count numeric vectors only; wire bytes include headers and node IDs.
        """
        self._check_snapshot()
        started = time.perf_counter()
        raw = torch.as_tensor(list(targets) if not isinstance(targets, torch.Tensor) else targets)
        if raw.ndim != 1 or (raw.numel() and raw.dtype not in {torch.int32, torch.int64}):
            raise ValueError("targets must be one-dimensional integer global node IDs")
        nodes = raw.to(dtype=torch.long, device="cpu").unique(sorted=True)
        if bool(((nodes < 0) | (nodes >= self._count)).any()):
            raise ValueError("target node ID out of range")
        accounting = {
            "request_bytes": 0, "response_bytes": 0, "payload_bytes": 0,
            "wire_bytes": 0, "remote_vectors": 0, "request_response_packets": 0,
            "cross_client_stages": 0, "encoded_rows": 0,
            "reused_encoded_rows": 0, "propagated_rows": 0,
            "inference_seconds": 0.0, "stages": [],
        }
        if nodes.numel() == 0:
            return FallbackResult(nodes, torch.empty((0, self._classes), dtype=self._dtype), accounting, self.snapshot_id)
        target_set = set(nodes.tolist())
        needed = [set(target_set) for _ in range(self._order)]
        for order in range(self._order - 1, 0, -1):
            for node in needed[order]:
                needed[order - 1].update(self._neighbors[node])
        states: list[dict[int, torch.Tensor]] = [{} for _ in range(self._order)]
        for owner in sorted({self._owners[node] for node in needed[0]}):
            required = sorted(node for node in needed[0] if self._owners[node] == owner)
            cached = self._nominal_cache.get(owner)
            if cached is not None:
                positions = torch.searchsorted(self._owner_nodes[owner], torch.tensor(required))
                for index, node in enumerate(required):
                    states[0][node] = cached[0][positions[index]]
                accounting["reused_encoded_rows"] += len(required)
            else:
                hidden = self._model.bounded_encoding(self._features[required])
                if not bool(torch.isfinite(hidden).all()):
                    raise ValueError("non-finite fallback encoding")
                states[0].update({node: hidden[index] for index, node in enumerate(required)})
                accounting["encoded_rows"] += len(required)
        for order in range(1, self._order):
            groups: dict[tuple[int, int], set[int]] = {}
            for node in needed[order]:
                receiver = self._owners[node]
                for source_node in self._neighbors[node]:
                    source = self._owners[source_node]
                    if source != receiver:
                        groups.setdefault((source, receiver), set()).add(source_node)
            received: dict[tuple[int, int], torch.Tensor] = {}
            stage = {"order": order, "batches": [], "remote_vectors": 0, "wire_bytes": 0}
            for (source, receiver), source_nodes in sorted(groups.items()):
                values, details = self._exchange(sorted(source_nodes), order - 1, source, receiver, states[order - 1])
                received.update({(receiver, node): vector for node, vector in values.items()})
                stage["batches"].append(details)
                stage["remote_vectors"] += details["remote_vectors"]
                stage["wire_bytes"] += details["wire_bytes"]
                for key in ("request_bytes", "response_bytes", "payload_bytes", "wire_bytes", "remote_vectors", "request_response_packets"):
                    accounting[key] += details[key]
            if groups:
                accounting["cross_client_stages"] += 1
            for owner in sorted({self._owners[node] for node in needed[order]}):
                local_nodes = sorted(node for node in needed[order] if self._owners[node] == owner)
                source_nodes = sorted({u for node in local_nodes for u in self._neighbors[node]})
                source_position = {node: index for index, node in enumerate(source_nodes)}
                vectors = torch.stack([
                    states[order - 1][node] if self._owners[node] == owner else received[(owner, node)]
                    for node in source_nodes
                ])
                rows, columns, weights = [], [], []
                for local_row, node in enumerate(local_nodes):
                    for source_node in self._neighbors[node]:
                        rows.append(local_row)
                        columns.append(source_position[source_node])
                        weights.append(1.0 / len(self._neighbors[node]))
                local_shift = torch.sparse_coo_tensor(
                    torch.tensor([rows, columns], dtype=torch.long),
                    torch.tensor(weights, dtype=self._dtype),
                    size=(len(local_nodes), len(source_nodes)),
                ).coalesce()
                propagated = torch.sparse.mm(local_shift, vectors)
                states[order].update({node: propagated[index] for index, node in enumerate(local_nodes)})
                accounting["propagated_rows"] += len(local_nodes)
            accounting["stages"].append(stage)
        filtered = self._model.theta[0] * torch.stack([states[0][node] for node in nodes.tolist()])
        for order in range(1, self._order):
            filtered += self._model.theta[order] * torch.stack([states[order][node] for node in nodes.tolist()])
        spectral = self._model.classifier(filtered)
        if self._model.encoder_type == "dual_feature_spectral":
            feature_rows: dict[int, torch.Tensor] = {}
            for owner in sorted({self._owners[node] for node in target_set}):
                local_nodes = sorted(node for node in target_set if self._owners[node] == owner)
                cached = self._nominal_cache.get(owner)
                if cached is not None:
                    positions = torch.searchsorted(self._owner_nodes[owner], torch.tensor(local_nodes))
                    values = cached[1][positions]
                else:
                    values = self._model.feature_classifier(self._model.feature_encoding(self._features[local_nodes]))
                feature_rows.update({node: values[index] for index, node in enumerate(local_nodes)})
            feature_logits = torch.stack([feature_rows[node] for node in nodes.tolist()])
            logits = feature_logits + self._model.spectral_fusion_scale * spectral
        else:
            logits = spectral
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("non-finite exact fallback logits")
        accounting["inference_seconds"] = time.perf_counter() - started
        return FallbackResult(nodes, logits, accounting, self.snapshot_id)
