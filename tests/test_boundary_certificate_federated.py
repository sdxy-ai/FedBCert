from __future__ import annotations

import numpy as np
import torch

from openfgl.flcore.fedbeso import mixed_membership

from openfgl.flcore.fedbcert import (
    build_federated_clients,
    boundary_severity_from_aggregates,
    build_boundary_ledger_views,
    canonical_undirected,
    missing_degrees,
    pairwise_masked_sum,
    induced_edges,
    project_auxiliary_gradient,
    weighted_average_states,
)


def test_weighted_state_average() -> None:
    states = [
        {"weight": torch.tensor([1.0, 3.0])},
        {"weight": torch.tensor([5.0, 7.0])},
    ]
    averaged = weighted_average_states(states, [1, 3])
    assert torch.allclose(averaged["weight"], torch.tensor([4.0, 6.0]))


def test_client_views_have_only_local_node_indices() -> None:
    full = canonical_undirected(
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), 4
    )
    membership = np.asarray([0, 0, 1, 1])
    local = induced_edges(full, membership)
    missing = missing_degrees(full, local, 4)
    features = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    train = torch.tensor([True, False, True, False])
    validation = torch.tensor([False, True, False, True])
    test = validation.clone()
    clients = build_federated_clients(
        full,
        membership,
        features,
        labels,
        train,
        validation,
        test,
        missing,
        3,
        torch.device("cpu"),
    )
    assert len(clients) == 2
    assert clients[0].features.size(0) == 2
    assert clients[1].features.size(0) == 2
    assert set(clients[0].global_nodes.tolist()).isdisjoint(
        clients[1].global_nodes.tolist()
    )
    for client in clients:
        assert tuple(client.shift.shape) == (2, 2)
        assert sorted(client.missing_degree.tolist()) == [0.0, 1.0]


def test_auxiliary_projection_preserves_primary_descent_halfspace() -> None:
    primary = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([-1.0, 2.0])]
    projected, diagnostics = project_auxiliary_gradient(primary, auxiliary)
    assert diagnostics["conflict"]
    assert diagnostics["pre_projection_dot"] == -1.0
    assert diagnostics["post_projection_dot"] >= -1e-12
    assert torch.allclose(projected[0], torch.tensor([0.0, 2.0]))


def test_nonconflicting_auxiliary_gradient_is_unchanged() -> None:
    primary = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([1.0, 2.0])]
    projected, diagnostics = project_auxiliary_gradient(primary, auxiliary)
    assert not diagnostics["conflict"]
    assert torch.equal(projected[0], auxiliary[0])


def test_boundary_severity_uses_only_aggregate_counts() -> None:
    assert boundary_severity_from_aggregates(80, 40) == 0.2


def test_pairwise_masked_sum_recovers_only_the_aggregate() -> None:
    aggregate, audit = pairwise_masked_sum(
        [[10, 3], [20, 5], [30, 8]], seed=17
    )
    assert aggregate == [60, 16]
    assert audit["masks_cancel_exactly"]
    assert not audit["individual_unmasked_values_visible_to_server"]
    assert not audit["differential_privacy_claimed"]


def test_boundary_ledger_has_only_local_edge_indices() -> None:
    full = canonical_undirected(
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), 4
    )
    membership = np.asarray([0, 0, 1, 1])
    local = induced_edges(full, membership)
    missing = missing_degrees(full, local, 4)
    ledgers = build_boundary_ledger_views(full, membership, missing)
    assert len(ledgers) == 2
    assert [ledger.missing_degree_sum for ledger in ledgers] == [1, 1]
    for ledger in ledgers:
        assert int(ledger.local_edge_index.min()) >= 0
        assert int(ledger.local_edge_index.max()) < ledger.global_nodes.numel()


def test_mixed_membership_is_deterministic_and_preserves_client_sizes() -> None:
    full = canonical_undirected(
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), 4
    )
    base = mixed_membership(full, 4, 2, seed=9, mixing=0.0)
    first = mixed_membership(full, 4, 2, seed=9, mixing=0.5)
    second = mixed_membership(full, 4, 2, seed=9, mixing=0.5)
    assert np.array_equal(first, second)
    assert sorted(np.bincount(first).tolist()) == sorted(np.bincount(base).tolist())
