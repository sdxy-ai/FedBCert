"""Independent, same-predictor controls for nonnegative completion geometry.

This module leaves the historical certificate implementation and its numerical
results unchanged.  Returned radii include the spectral fusion coefficient.
The acceptance guard is an engineering threshold, not a rounding-error proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .certificate import (
    PolynomialSpectralClassifier,
    cone_pairwise_certificate_statistics,
    positive_orthant_unit_support,
    retained_mass_uncertainty_basis,
)


@dataclass(frozen=True)
class GeometryCertificateStatistics:
    """Node-aligned results; radius is a representation, not logit, bound.

    For cone/combined the radius remains the diameter control's valid norm
    radius.  It does not encode their directional margin bounds.
    """

    worst_slack: torch.Tensor
    certified: torch.Tensor
    radius: torch.Tensor

    @property
    def representation_radius(self) -> torch.Tensor:
        return self.radius


def positive_orthant_direction_span(direction: torch.Tensor) -> torch.Tensor:
    """Exact projection width: ||q_+|| + ||q_-|| <= sqrt(2) ||q||."""
    if not direction.is_floating_point() or not bool(torch.isfinite(direction).all()):
        raise ValueError("directions must be finite floating-point tensors")
    lower, upper = positive_orthant_unit_support(direction)
    return upper - lower


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _pairwise_slack(
    model: PolynomialSpectralClassifier,
    logits: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    predicted = logits.argmax(dim=1)
    margins = logits.gather(1, predicted[:, None]) - logits
    differences = (
        model.classifier.weight[predicted, None, :]
        - model.classifier.weight[None, :, :]
    )
    slack = margins - radius[:, None] * differences.norm(p=2, dim=2)
    slack.scatter_(1, predicted[:, None], float("inf"))
    return slack.min(dim=1).values


@torch.no_grad()
def evaluate_geometry_baselines(
    model: PolynomialSpectralClassifier,
    features: torch.Tensor,
    shift: torch.Tensor,
    row_budget: torch.Tensor,
    *,
    guard: float = 1e-6,
    logits: torch.Tensor | None = None,
    basis: Sequence[torch.Tensor] | None = None,
) -> dict[str, GeometryCertificateStatistics]:
    """Evaluate ball, diameter, and directional rules on identical logits.

    Keys are ball_pairwise, nonnegative_diameter, cone, and combined.  The
    first two use pairwise weight differences regardless of the historical
    model's margin-mode setting.  Diameter uses r_ball / sqrt(2): both local
    and remote propagated rows belong to the nonnegative unit ball, whose
    diameter is at most sqrt(2).  Thus each retained-mass uncertainty basis
    term, including its cap, scales by 1/sqrt(2).

    The cone rule calls the existing routine without its symmetric union.
    Combined is the union of the *whole* cone and diameter acceptance events,
    implemented by taking their maximum worst competing-class slack.

    Only unchanged linear_relu and dual_feature_spectral architectures are
    supported.  Remote encodings must obey the same architecture/domain; the
    current local-row checks alone do not establish a remote-data contract.
    Optional cached logits/bases must exactly match a fresh evaluation.  A
    mismatch raises instead of silently certifying another snapshot.

    Every returned slack/radius is finite.  Invalid input or non-finite
    arithmetic raises ValueError; the caller may map it to Unavailable.  A
    zero spectral fusion coefficient is an explicit graph-invariant case,
    certified even at a nominal tie, as in the historical implementation.
    """
    if not isinstance(model, PolynomialSpectralClassifier):
        raise ValueError("model must be a PolynomialSpectralClassifier")
    if any(module.training for module in model.modules()):
        raise ValueError("geometry certification requires model.eval()")
    if model.encoder_type not in {"linear_relu", "dual_feature_spectral"}:
        raise ValueError("unsupported encoder type for nonnegative geometry")
    if not math.isfinite(float(guard)) or float(guard) < 0.0:
        raise ValueError("guard must be finite and nonnegative")
    alpha = float(model.spectral_fusion_scale)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("spectral fusion coefficient must be finite and nonnegative")
    if model.encoder_type == "linear_relu" and alpha != 1.0:
        raise ValueError("linear_relu forward does not apply a fusion scale; require scale=1")
    if features.ndim != 2 or features.size(0) == 0:
        raise ValueError("features must be a nonempty node-by-feature matrix")
    if not features.is_floating_point():
        raise ValueError("features must use a floating-point dtype")
    if features.dtype != model.theta.dtype or features.device != model.theta.device:
        raise ValueError("features and model must share dtype and device")
    _finite("features", features)
    for name, value in model.named_parameters():
        _finite(name, value)
    _finite("polynomial coefficients", model.theta)
    node_count = features.size(0)
    if shift.layout not in {torch.sparse_coo, torch.sparse_csr}:
        raise ValueError("shift must be a sparse COO or CSR matrix")
    if shift.shape != (node_count, node_count):
        raise ValueError("shift must be node aligned")
    if shift.dtype != features.dtype or shift.device != features.device:
        raise ValueError("shift and features must share dtype and device")
    coo = shift.to_sparse_coo().coalesce()
    _finite("shift", coo.values())
    if bool((coo.values() < 0.0).any()):
        raise ValueError("shift must be nonnegative")
    tolerance = 64.0 * torch.finfo(features.dtype).eps
    row_sum = torch.sparse.sum(coo, dim=1).to_dense()
    if not torch.allclose(row_sum, torch.ones_like(row_sum), rtol=0.0, atol=tolerance):
        raise ValueError("shift must be row stochastic")
    if row_budget.shape != (node_count,):
        raise ValueError("row_budget must have one entry per node")
    if row_budget.dtype != features.dtype or row_budget.device != features.device:
        raise ValueError("row_budget and features must share dtype and device")
    _finite("row_budget", row_budget)
    if bool(((row_budget < 0.0) | (row_budget > 2.0)).any()):
        raise ValueError("row_budget must lie in [0, 2]")

    hidden = model.bounded_encoding(features)
    _finite("hidden encodings", hidden)
    if bool((hidden < 0.0).any()):
        raise ValueError("nonnegative geometry requires nonnegative encodings")
    if bool((hidden.norm(p=2, dim=1) > 1.0 + tolerance).any()):
        raise ValueError("hidden row norms must be at most one")
    fresh_logits = model(features, shift)
    _finite("logits", fresh_logits)
    if fresh_logits.ndim != 2 or fresh_logits.size(1) < 2:
        raise ValueError("at least two logit classes are required")
    if logits is not None:
        _finite("cached logits", logits)
        if (logits.dtype != fresh_logits.dtype or logits.device != fresh_logits.device
                or logits.shape != fresh_logits.shape or not torch.equal(logits, fresh_logits)):
            raise ValueError("cached logits do not match this model and local view")
    else:
        logits = fresh_logits
    fresh_basis = retained_mass_uncertainty_basis(shift, row_budget, model.order_count)
    if basis is not None:
        if len(basis) != len(fresh_basis) or any(
            cached.dtype != fresh.dtype or cached.device != fresh.device
            or cached.shape != fresh.shape or not torch.equal(cached, fresh)
            for cached, fresh in zip(basis, fresh_basis)
        ):
            raise ValueError("cached basis does not match retained-mass recurrence")
    basis = fresh_basis
    ball_radius = model.boundary_radius(basis)
    diameter_radius = ball_radius / math.sqrt(2.0)
    ball_slack = _pairwise_slack(model, logits, ball_radius)
    diameter_slack = _pairwise_slack(model, logits, diameter_radius)
    cone = cone_pairwise_certificate_statistics(
        model, features, shift, row_budget, logits, basis,
        count_semantics="upper_bound", combine_symmetric=False,
    )
    cone_slack = cone.worst_slack
    # The legacy alpha=0 path uses its configured margin mode; both penalties
    # vanish, so it still returns the same nominal competing-class margin.
    combined_slack = torch.maximum(cone_slack, diameter_slack)
    results: dict[str, GeometryCertificateStatistics] = {}
    for kind, slack, radius in (
        ("ball_pairwise", ball_slack, ball_radius),
        ("nonnegative_diameter", diameter_slack, diameter_radius),
        ("cone", cone_slack, diameter_radius),
        ("combined", combined_slack, diameter_radius),
    ):
        _finite(f"{kind} slack", slack)
        _finite(f"{kind} radius", radius)
        certified = torch.ones_like(slack, dtype=torch.bool) if alpha == 0.0 else slack > guard
        results[kind] = GeometryCertificateStatistics(slack, certified, radius)
    return results
