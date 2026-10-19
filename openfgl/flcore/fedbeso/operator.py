from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


@dataclass
class BoundarySchurBlock:
    local_nodes: np.ndarray
    outside_nodes: np.ndarray
    boundary_local: np.ndarray
    shifted_local: np.ndarray
    correction_boundary: np.ndarray
    outside_solver: spla.SuperLU
    coupling_local_outside: sp.csr_matrix

    def shifted_with_correction(self, correction: np.ndarray) -> np.ndarray:
        output = self.shifted_local.copy()
        index = np.ix_(self.boundary_local, self.boundary_local)
        output[index] -= correction
        return output

    def local_forcing(self, signals: np.ndarray) -> np.ndarray:
        return np.asarray(signals[self.local_nodes], dtype=np.float64)

    def full_forcing(self, signals: np.ndarray) -> np.ndarray:
        outside_response = self.outside_solver.solve(
            np.asarray(signals[self.outside_nodes], dtype=np.float64)
        )
        return self.local_forcing(signals) - self.coupling_local_outside @ outside_response


def normalized_laplacian(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    edge_index, _ = remove_self_loops(edge_index.detach().cpu())
    edge_index = to_undirected(edge_index, num_nodes=int(num_nodes))
    edge_index, _ = coalesce(edge_index, None, int(num_nodes), int(num_nodes))
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    adjacency = sp.coo_matrix(
        (np.ones(row.size, dtype=np.float64), (row, col)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inverse_sqrt[positive] = degree[positive] ** -0.5
    normalized = sp.diags(inverse_sqrt) @ adjacency @ sp.diags(inverse_sqrt)
    return (sp.eye(num_nodes, format="csr") - normalized).tocsr()


def balanced_random_membership(num_nodes: int, num_clients: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    permutation = generator.permutation(int(num_nodes))
    membership = np.empty(num_nodes, dtype=np.int64)
    for client, nodes in enumerate(np.array_split(permutation, int(num_clients))):
        membership[nodes] = client
    return membership


def metis_membership(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_clients: int,
    seed: int,
) -> np.ndarray:
    import pymetis

    undirected = to_undirected(edge_index.detach().cpu(), num_nodes=int(num_nodes))
    rows: list[list[int]] = [[] for _ in range(int(num_nodes))]
    for source, target in undirected.t().tolist():
        if source != target:
            rows[int(source)].append(int(target))
    rows = [sorted(set(neighbors)) for neighbors in rows]
    options = pymetis.Options(seed=int(seed))
    _, membership = pymetis.part_graph(
        int(num_clients), adjacency=rows, options=options
    )
    return np.asarray(membership, dtype=np.int64)


def mixed_membership(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_clients: int,
    seed: int,
    mixing: float,
) -> np.ndarray:
    """Topology-only interpolation between METIS and randomized ownership.

    Random cross-client membership swaps preserve every client's node count.
    ``mixing=0`` is exactly METIS; increasing values apply more swaps and
    progressively disrupt topology-local ownership without using labels.
    """
    fraction = float(mixing)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("mixing must lie in [0, 1]")
    membership = metis_membership(
        edge_index, num_nodes, num_clients, seed
    ).copy()
    if fraction == 0.0:
        return membership
    generator = np.random.default_rng(int(seed) * 1000003 + 4049)
    swap_count = int(round(fraction * int(num_nodes)))
    for _ in range(swap_count):
        left = int(generator.integers(0, int(num_nodes)))
        candidates = np.flatnonzero(membership != membership[left])
        if candidates.size == 0:
            break
        right = int(candidates[int(generator.integers(0, candidates.size))])
        membership[left], membership[right] = (
            membership[right], membership[left]
        )
    return membership


def boundary_schur_block(
    laplacian: sp.csr_matrix,
    membership: np.ndarray,
    client_id: int,
    tau: float,
) -> BoundarySchurBlock:
    local_nodes = np.flatnonzero(membership == int(client_id))
    outside_nodes = np.flatnonzero(membership != int(client_id))
    local = laplacian[local_nodes][:, local_nodes].tocsr()
    local_outside = laplacian[local_nodes][:, outside_nodes].tocsr()
    outside = laplacian[outside_nodes][:, outside_nodes].tocsc()
    boundary_local = np.flatnonzero(np.diff(local_outside.indptr) > 0)
    shifted_local = (local + float(tau) * sp.eye(local.shape[0], format="csr")).toarray()
    shifted_outside = outside + float(tau) * sp.eye(outside.shape[0], format="csc")
    solver = spla.splu(shifted_outside)
    if boundary_local.size:
        coupling = local_outside[boundary_local]
        solution = solver.solve(coupling.transpose().toarray())
        correction = np.asarray(coupling @ solution, dtype=np.float64)
        correction = 0.5 * (correction + correction.T)
    else:
        correction = np.zeros((0, 0), dtype=np.float64)
    return BoundarySchurBlock(
        local_nodes=local_nodes,
        outside_nodes=outside_nodes,
        boundary_local=boundary_local,
        shifted_local=shifted_local,
        correction_boundary=correction,
        outside_solver=solver,
        coupling_local_outside=local_outside,
    )


def truncated_psd(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    if matrix.size == 0:
        return matrix.copy(), 1.0
    eigenvalues, eigenvectors = scipy.linalg.eigh(matrix, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    resolved = min(max(int(rank), 0), eigenvalues.size)
    retained = eigenvalues[:resolved]
    approximation = (
        eigenvectors[:, :resolved] * retained[None, :]
    ) @ eigenvectors[:, :resolved].T
    total = float(eigenvalues.sum())
    energy = 1.0 if total <= 1e-12 else float(retained.sum() / total)
    return approximation, energy


def solve_dense_spd(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    factor = scipy.linalg.cho_factor(matrix, lower=True, check_finite=False)
    return scipy.linalg.cho_solve(factor, rhs, check_finite=False)
