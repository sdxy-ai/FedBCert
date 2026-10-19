from __future__ import annotations

import numpy as np
import torch

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    canonical_undirected,
    certificate_statistics,
    induced_edges,
    missing_degrees,
    row_normalized_shift,
    uncertainty_basis,
)


def fixture():
    full = canonical_undirected(
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), 4
    )
    membership = np.asarray([0, 0, 1, 1])
    local = induced_edges(full, membership)
    return full, local


def test_boundary_budget_equals_shift_row_l1_difference() -> None:
    full, local = fixture()
    local_shift, local_degree = row_normalized_shift(
        local, 4, device=torch.device("cpu")
    )
    full_shift, _ = row_normalized_shift(full, 4, device=torch.device("cpu"))
    missing = missing_degrees(full, local, 4)
    budget = boundary_uncertainty(local_degree, missing)
    actual = (full_shift.to_dense() - local_shift.to_dense()).abs().sum(dim=1)
    assert torch.allclose(actual, budget, atol=1e-6, rtol=1e-6)


def test_logit_bound_contains_true_unknown_edge_completion() -> None:
    torch.manual_seed(7)
    full, local = fixture()
    local_shift, local_degree = row_normalized_shift(
        local, 4, device=torch.device("cpu")
    )
    full_shift, _ = row_normalized_shift(full, 4, device=torch.device("cpu"))
    missing = missing_degrees(full, local, 4)
    basis = uncertainty_basis(
        local_shift, boundary_uncertainty(local_degree, missing), 3
    )
    model = PolynomialSpectralClassifier(5, 6, 3, 3, 0.0).eval()
    features = torch.randn(4, 5)
    with torch.no_grad():
        hidden = model.bounded_encoding(features)
        local_logits = model.classifier(model.filter(hidden, local_shift))
        full_logits = model.classifier(model.filter(hidden, full_shift))
        certificate = certificate_statistics(model, local_logits, basis)
    actual = (full_logits - local_logits).abs()
    assert torch.all(actual <= certificate.class_logit_bounds + 1e-6)
