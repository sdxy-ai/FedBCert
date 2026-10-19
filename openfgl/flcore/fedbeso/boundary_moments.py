from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp


@dataclass(frozen=True)
class DirectedBoundaryBlock:
    receiver: int
    sender: int
    receiver_nodes: np.ndarray
    sender_nodes: np.ndarray
    row_mass: np.ndarray
    column_mass: np.ndarray
    total_mass: float


def propagation_from_laplacian(laplacian: sp.spmatrix) -> sp.csr_matrix:
    """Return the symmetric normalized propagation matrix P = I - L."""
    size = int(laplacian.shape[0])
    return (sp.eye(size, format="csr") - laplacian.tocsr()).tocsr()


def directed_boundary_blocks(
    propagation: sp.spmatrix,
    membership: np.ndarray,
) -> dict[tuple[int, int], DirectedBoundaryBlock]:
    propagation = propagation.tocsr()
    clients = np.unique(membership).astype(int)
    nodes = {client: np.flatnonzero(membership == client) for client in clients}
    output: dict[tuple[int, int], DirectedBoundaryBlock] = {}
    for receiver in clients:
        for sender in clients:
            if receiver == sender:
                continue
            block = propagation[nodes[receiver]][:, nodes[sender]].tocsr()
            row_mass = np.asarray(block.sum(axis=1)).reshape(-1)
            column_mass = np.asarray(block.sum(axis=0)).reshape(-1)
            total_mass = float(row_mass.sum())
            if total_mass <= 1e-15:
                continue
            output[(receiver, sender)] = DirectedBoundaryBlock(
                receiver=receiver,
                sender=sender,
                receiver_nodes=nodes[receiver],
                sender_nodes=nodes[sender],
                row_mass=row_mass,
                column_mass=column_mass,
                total_mass=total_mass,
            )
    return output


def boundary_moments(
    states: np.ndarray,
    blocks: dict[tuple[int, int], DirectedBoundaryBlock],
) -> dict[tuple[int, int], np.ndarray]:
    """Aggregate one class/channel vector for every directed client pair."""
    output: dict[tuple[int, int], np.ndarray] = {}
    for key, block in blocks.items():
        sender_state = states[block.sender_nodes]
        output[key] = (block.column_mass @ sender_state) / block.total_mass
    return output


def moment_forcing(
    num_nodes: int,
    channels: int,
    blocks: dict[tuple[int, int], DirectedBoundaryBlock],
    moments: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    forcing = np.zeros((num_nodes, channels), dtype=np.float64)
    for key, block in blocks.items():
        forcing[block.receiver_nodes] += block.row_mass[:, None] * moments[key][None, :]
    return forcing


def conservation_error(
    states: np.ndarray,
    blocks: dict[tuple[int, int], DirectedBoundaryBlock],
    moments: dict[tuple[int, int], np.ndarray],
) -> float:
    """Verify exact preservation of the aggregate cross-boundary message."""
    worst = 0.0
    for key, block in blocks.items():
        exact_total = block.column_mass @ states[block.sender_nodes]
        approximate_total = block.row_mass.sum() * moments[key]
        denominator = max(float(np.linalg.norm(exact_total)), 1e-12)
        worst = max(
            worst,
            float(np.linalg.norm(exact_total - approximate_total) / denominator),
        )
    return worst


def _local_systems(
    propagation: sp.spmatrix,
    membership: np.ndarray,
    tau: float,
) -> dict[int, tuple[np.ndarray, tuple[np.ndarray, bool]]]:
    propagation = propagation.tocsr()
    output: dict[int, tuple[np.ndarray, tuple[np.ndarray, bool]]] = {}
    for client in np.unique(membership).astype(int):
        local_nodes = np.flatnonzero(membership == client)
        local = propagation[local_nodes][:, local_nodes].toarray()
        matrix = (1.0 + float(tau)) * np.eye(len(local_nodes)) - local
        factor = scipy.linalg.cho_factor(matrix, lower=True, check_finite=False)
        output[client] = (local_nodes, factor)
    return output


def solve_with_moment_forcing(
    propagation: sp.spmatrix,
    membership: np.ndarray,
    signals: np.ndarray,
    tau: float,
    forcing: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(signals, dtype=np.float64)
    for _, (local_nodes, factor) in _local_systems(
        propagation, membership, tau
    ).items():
        rhs = float(tau) * signals[local_nodes] + forcing[local_nodes]
        output[local_nodes] = scipy.linalg.cho_solve(
            factor, rhs, check_finite=False
        )
    return output


def solve_without_boundary(
    propagation: sp.spmatrix,
    membership: np.ndarray,
    signals: np.ndarray,
    tau: float,
) -> np.ndarray:
    return solve_with_moment_forcing(
        propagation,
        membership,
        signals,
        tau,
        np.zeros_like(signals, dtype=np.float64),
    )


def self_consistent_moment_solve(
    propagation: sp.spmatrix,
    membership: np.ndarray,
    signals: np.ndarray,
    tau: float,
    *,
    max_iterations: int = 100,
    damping: float = 0.5,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, int, float, float]:
    blocks = directed_boundary_blocks(propagation, membership)
    states = solve_without_boundary(propagation, membership, signals, tau)
    residual = float("inf")
    conservation = 0.0
    for iteration in range(1, int(max_iterations) + 1):
        moments = boundary_moments(states, blocks)
        conservation = max(
            conservation, conservation_error(states, blocks, moments)
        )
        forcing = moment_forcing(
            states.shape[0], states.shape[1], blocks, moments
        )
        proposal = solve_with_moment_forcing(
            propagation, membership, signals, tau, forcing
        )
        updated = float(damping) * proposal + (1.0 - float(damping)) * states
        denominator = max(float(np.linalg.norm(states)), 1e-12)
        residual = float(np.linalg.norm(updated - states) / denominator)
        states = updated
        if residual <= float(tolerance):
            break
    return states, iteration, residual, conservation
