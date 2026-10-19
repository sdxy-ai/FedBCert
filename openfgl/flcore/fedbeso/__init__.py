"""Boundary-equivalent spectral operator diagnostics for subgraph FL."""

from .operator import (
    BoundarySchurBlock,
    balanced_random_membership,
    boundary_schur_block,
    metis_membership,
    mixed_membership,
    normalized_laplacian,
    truncated_psd,
)
from .boundary_moments import (
    boundary_moments,
    conservation_error,
    directed_boundary_blocks,
    moment_forcing,
    propagation_from_laplacian,
    self_consistent_moment_solve,
    solve_with_moment_forcing,
    solve_without_boundary,
)

__all__ = [
    "BoundarySchurBlock",
    "balanced_random_membership",
    "boundary_schur_block",
    "metis_membership",
    "mixed_membership",
    "normalized_laplacian",
    "truncated_psd",
    "boundary_moments",
    "conservation_error",
    "directed_boundary_blocks",
    "moment_forcing",
    "propagation_from_laplacian",
    "self_consistent_moment_solve",
    "solve_with_moment_forcing",
    "solve_without_boundary",
]
