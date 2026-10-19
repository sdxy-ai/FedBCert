from __future__ import annotations

import numpy as np
import pytest
import torch

from openfgl.flcore.fedbcert.certificate import PolynomialSpectralClassifier, row_normalized_shift
from openfgl.flcore.fedbcert import on_demand
from openfgl.flcore.fedbcert.on_demand import (
    AggregateReplayService, decode_metadata_response, decode_vector_response,
    potential_exposure, state_fingerprint,
)


def _fixture(kind="dual_feature_spectral", dtype=torch.float64, order=4):
    torch.manual_seed(271)
    model = PolynomialSpectralClassifier(
        input_dim=3, hidden_dim=5, num_classes=3, order_count=order,
        dropout=0.2, encoder_type=kind,
        encoder_depth=1 if kind == "linear_relu" else 2,
        spectral_fusion_scale=1.0 if kind == "linear_relu" else 0.25,
        certificate_margin_mode="pairwise_difference",
    ).to(dtype).eval()
    with torch.no_grad():
        model.theta.copy_(torch.randn(order, dtype=dtype))
    features = torch.randn(8, 3, dtype=dtype)
    edges = torch.tensor([
        [0, 0, 1, 2, 3, 4, 5, 6, 1],
        [1, 2, 3, 4, 4, 5, 7, 7, 6],
    ], dtype=torch.long)
    membership = torch.tensor([0, 0, 1, 1, 2, 2, 0, 2])
    return model, features, edges, membership


def _central(model, features, edges):
    shift, _ = row_normalized_shift(
        edges, features.size(0), device=torch.device("cpu"), dtype=features.dtype,
    )
    with torch.no_grad():
        return model(features, shift)


@pytest.mark.parametrize("kind", ["linear_relu", "dual_feature_spectral"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_matches_central_for_signed_polynomials_and_subsets(kind, dtype):
    model, features, edges, membership = _fixture(kind, dtype)
    service = AggregateReplayService(model, features, edges, membership)
    expected = _central(model, features, edges)
    for targets in ([0], [3, 0, 3, 5], list(range(features.size(0)))):
        result = service.exact(targets)
        tolerance = 2e-6 if dtype == torch.float32 else 2e-13
        assert torch.allclose(result.logits, expected[result.nodes], atol=tolerance, rtol=tolerance)
        assert result.nodes.tolist() == sorted(set(targets))
        assert result.accounting["encoded_rows"] > 0
        assert result.accounting["wire_bytes"] == result.accounting["request_bytes"] + result.accounting["response_bytes"]


def test_remote_remote_dependencies_and_full_degree_normalization():
    model, features, edges, membership = _fixture(order=3)
    service = AggregateReplayService(model, features, edges, membership)
    result = service.exact([0])
    assert torch.allclose(result.logits, _central(model, features, edges)[[0]], atol=1e-13, rtol=1e-13)
    # Target 0 (owner 0) needs node 2 (owner 1), which at its previous order
    # needs node 4 (owner 2). That remote-remote message must be transmitted.
    stage_zero = result.accounting["stages"][0]["batches"]
    assert any(
        batch["source_owner"] == 2 and batch["receiver_owner"] == 1 and 4 in batch["node_ids"]
        for batch in stage_zero
    )
    assert result.accounting["cross_client_stages"] == 2
    nominal = service.nominal(0)
    position = nominal.nodes.tolist().index(0)
    assert not torch.allclose(nominal.logits[position], result.logits[0], atol=1e-8, rtol=1e-8)


def test_nominal_matches_forward_local_hidden_cache_reuse_and_return_isolation():
    model, features, edges, membership = _fixture()
    service = AggregateReplayService(model, features, edges, membership)
    for owner in membership.unique().tolist():
        view = service.local_view(owner)
        nominal = service.nominal(owner)
        with torch.no_grad():
            assert torch.equal(nominal.logits, model(view.features, view.shift))
            assert torch.equal(nominal.hidden, model.bounded_encoding(view.features))
        assert nominal.encoded_rows == view.nodes.numel()
        assert service.nominal(owner).encoded_rows == 0
        view.features.zero_()
        nominal.hidden.zero_()
        nominal.logits.zero_()
        if nominal.feature_logits is not None:
            nominal.feature_logits.zero_()
    exact = service.exact([0, 3])
    assert exact.accounting["encoded_rows"] == 0
    assert exact.accounting["reused_encoded_rows"] > 0
    assert torch.allclose(exact.logits, _central(model, features, edges)[exact.nodes], atol=1e-13, rtol=1e-13)


def test_message_deduplication_and_subset_work_reduction():
    model, features, edges, membership = _fixture()
    service = AggregateReplayService(model, features, edges, membership)
    subset = service.exact([0])
    full = service.exact(range(features.size(0)))
    for result in (subset, full):
        keys = []
        groups = []
        numeric_payload = 0
        request_bytes = 0
        response_bytes = 0
        for stage in result.accounting["stages"]:
            for batch in stage["batches"]:
                groups.append((batch["source_owner"], batch["receiver_owner"], batch["previous_order"]))
                for node in batch["node_ids"]:
                    keys.append((node, batch["receiver_owner"], batch["previous_order"]))
                numeric_payload += 8 * len(batch["node_ids"]) * model.classifier.in_features
                request_bytes += len(batch["node_ids"]) * 8 + on_demand._HEADER.size
                response_bytes += len(batch["node_ids"]) * (8 + 8 * model.classifier.in_features) + on_demand._HEADER.size + 4
        assert len(keys) == len(set(keys)) == result.accounting["remote_vectors"]
        assert len(groups) == len(set(groups))
        assert result.accounting["request_response_packets"] == 2 * len(groups)
        assert result.accounting["payload_bytes"] == numeric_payload
        assert result.accounting["request_bytes"] == request_bytes
        assert result.accounting["response_bytes"] == response_bytes
    assert subset.accounting["remote_vectors"] < full.accounting["remote_vectors"]
    assert subset.accounting["propagated_rows"] < full.accounting["propagated_rows"]


def test_fallback_does_not_reuse_hidden_or_propagation_between_batches():
    model, features, edges, membership = _fixture()
    service = AggregateReplayService(model, features, edges, membership)
    first, second = service.exact([0]), service.exact([0])
    assert first.accounting["encoded_rows"] == second.accounting["encoded_rows"] > 0
    assert first.accounting["reused_encoded_rows"] == second.accounting["reused_encoded_rows"] == 0
    assert first.accounting["wire_bytes"] == second.accounting["wire_bytes"]
    assert torch.equal(first.logits, second.logits)


def test_receivers_use_decoded_messages_not_a_hidden_full_graph_oracle(monkeypatch):
    model, features, edges, membership = _fixture()
    reference = AggregateReplayService(model, features, edges, membership).exact([0]).logits
    original = on_demand.decode_vector_response
    seen = []

    def replace_decoded_values(payload, **kwargs):
        values = original(payload, **kwargs)
        seen.append(values.shape)
        return torch.zeros_like(values)

    monkeypatch.setattr(on_demand, "decode_vector_response", replace_decoded_values)
    corrupted = AggregateReplayService(model, features, edges, membership).exact([0]).logits
    assert seen
    assert not torch.allclose(corrupted, reference, atol=1e-8, rtol=1e-8)


def test_metadata_actual_wire_lengths_and_presence_only_summary():
    model, features, edges, membership = _fixture()
    service = AggregateReplayService(model, features, edges, membership)
    for owner in membership.unique().tolist():
        view = service.local_view(owner)
        count = service.metadata(owner, "upper_count")
        presence = service.metadata(owner, "presence")
        assert count.values.dtype == torch.long
        assert presence.values.dtype == torch.bool
        assert torch.equal(presence.values, count.values > 0)
        assert count.payload_bytes == 4 * view.nodes.numel()
        assert presence.payload_bytes == (view.nodes.numel() + 7) // 8
        assert count.request_bytes == len(count.request_payload)
        assert count.response_bytes == len(count.response_payload)
        assert count.wire_bytes == len(count.request_payload) + len(count.response_payload)
        assert len(presence.response_payload) == on_demand._HEADER.size + 1 + 8 * view.nodes.numel() + presence.payload_bytes
        # True external-degree counts, not full degrees or endpoint lists.
        _, full_degree = row_normalized_shift(edges, features.size(0), device=torch.device("cpu"), dtype=features.dtype)
        assert torch.equal(count.values.to(features.dtype), full_degree[view.nodes] - view.retained_degree)
    with pytest.raises(ValueError, match="metadata kind"):
        service.metadata(0, "exact_endpoint_list")


def test_snapshot_and_vector_header_checks(monkeypatch):
    model, features, edges, membership = _fixture()
    service = AggregateReplayService(model, features, edges, membership)
    capture = []
    original = on_demand.decode_vector_response

    def record(payload, **kwargs):
        capture.append((payload, kwargs.copy()))
        return original(payload, **kwargs)

    monkeypatch.setattr(on_demand, "decode_vector_response", record)
    service.exact([0])
    payload, kwargs = capture[0]
    broken = bytearray(payload)
    broken[10] ^= 1  # snapshot digest in the actual serialized response
    with pytest.raises(ValueError, match="header mismatch"):
        decode_vector_response(bytes(broken), **kwargs)
    bad = dict(kwargs, stage=kwargs["stage"] + 1)
    with pytest.raises(ValueError, match="header mismatch"):
        decode_vector_response(payload, **bad)
    with pytest.raises(ValueError, match="length"):
        decode_vector_response(payload[:-1], **kwargs)
    metadata = service.metadata(0, "upper_count")
    with pytest.raises(ValueError, match="header mismatch"):
        decode_metadata_response(metadata.response_payload, snapshot_id="00" * 32, owner=0, kind="upper_count", expected_nodes=metadata.nodes.tolist())


@pytest.mark.parametrize("what", ["model", "features", "edges", "membership", "mode"])
def test_mutated_snapshot_rejected(what):
    model, features, edges, membership = _fixture()
    service = AggregateReplayService(model, features, edges, membership)
    with torch.no_grad():
        if what == "model":
            model.theta[0].add_(0.1)
        elif what == "features":
            features[0, 0].add_(0.1)
        elif what == "edges":
            edges[0, 0] = 7
        elif what == "membership":
            membership[0] = 2
        else:
            model.train()
    with pytest.raises(ValueError, match="snapshot|changed|eval"):
        service.exact([0])
    with pytest.raises(ValueError, match="snapshot|changed|eval"):
        service.nominal(0)


def test_empty_targets_zero_cross_client_case_and_invalid_ids():
    model, features, edges, _ = _fixture()
    membership = np.zeros(features.size(0), dtype=np.int64)
    service = AggregateReplayService(model, features, edges, membership)
    empty = service.exact([])
    assert empty.logits.shape == (0, model.classifier.out_features)
    assert empty.accounting["wire_bytes"] == empty.accounting["encoded_rows"] == 0
    assert empty.accounting["stages"] == []
    actual = service.exact([0, 2])
    assert actual.accounting["remote_vectors"] == actual.accounting["wire_bytes"] == 0
    assert actual.accounting["request_response_packets"] == actual.accounting["cross_client_stages"] == 0
    assert not service.metadata(0, "presence").values.any()
    for targets in ([-1], [8], [0.5]):
        with pytest.raises(ValueError, match="target"):
            service.exact(targets)


def test_potential_exposure_k1_k2_k3_and_direction():
    shift = torch.tensor([
        [0.5, 0.5, 0.0, 0.0],
        [0.25, 0.5, 0.25, 0.0],
        [0.0, 0.25, 0.5, 0.25],
        [0.0, 0.0, 0.5, 0.5],
    ], dtype=torch.float64).to_sparse_coo().coalesce()
    flags = torch.tensor([True, False, False, False])
    assert potential_exposure(shift, flags, 1).tolist() == [False, False, False, False]
    assert potential_exposure(shift, flags, 2).tolist() == [True, False, False, False]
    assert potential_exposure(shift, flags, 3).tolist() == [True, True, False, False]
    assert potential_exposure(shift, flags, 4).tolist() == [True, True, True, False]


def test_state_fingerprint_matches_service_and_detects_config_changes():
    model, features, edges, membership = _fixture()
    before = state_fingerprint(model)
    service = AggregateReplayService(model, features, edges, membership)
    assert service.snapshot_model_hash == before
    model.spectral_fusion_scale += 0.01
    assert state_fingerprint(model) != before
    with pytest.raises(ValueError, match="changed"):
        service.local_view(0)
