from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    remove_self_loops,
    to_undirected,
)


def canonical_undirected(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edges, _ = remove_self_loops(edge_index.detach().cpu())
    edges = to_undirected(edges, num_nodes=int(num_nodes))
    edges, _ = coalesce(edges, None, int(num_nodes), int(num_nodes))
    return edges.long()


def induced_edges(
    edge_index: torch.Tensor, membership: np.ndarray
) -> torch.Tensor:
    member = torch.as_tensor(membership, dtype=torch.long)
    keep = member[edge_index[0]] == member[edge_index[1]]
    return edge_index[:, keep]


def missing_degrees(
    full_edge_index: torch.Tensor,
    local_edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    full = torch.bincount(full_edge_index[1], minlength=int(num_nodes))
    local = torch.bincount(local_edge_index[1], minlength=int(num_nodes))
    missing = full - local
    if bool((missing < 0).any()):
        raise RuntimeError("local graph cannot contain more incident edges than full graph")
    return missing.to(torch.float32)


def row_normalized_shift(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    edges = canonical_undirected(edge_index, num_nodes)
    edges, _ = add_self_loops(edges, num_nodes=int(num_nodes))
    edges, _ = coalesce(edges, None, int(num_nodes), int(num_nodes))
    target = edges[1]
    degree = torch.bincount(target, minlength=int(num_nodes)).to(dtype)
    weights = degree[target].reciprocal()
    indices = torch.stack([target, edges[0]], dim=0).to(device)
    shift = torch.sparse_coo_tensor(
        indices,
        weights.to(device),
        size=(int(num_nodes), int(num_nodes)),
        dtype=dtype,
        device=device,
    ).coalesce()
    return shift, degree.to(device)


def boundary_uncertainty(
    local_degree_with_self: torch.Tensor,
    missing_degree: torch.Tensor,
) -> torch.Tensor:
    """Row total-variation budget for an unknown boundary completion.

    Adding ``b_v`` unknown neighbors changes the row-normalized distribution by
    exactly ``2 b_v / (d_v + b_v)`` when ``d_v`` includes the retained self-loop.
    The result needs no remote endpoint identity or feature.
    """
    local = local_degree_with_self.to(missing_degree.device, missing_degree.dtype)
    denominator = (local + missing_degree).clamp_min(1.0)
    return (2.0 * missing_degree / denominator).clamp(min=0.0, max=2.0)


def uncertainty_basis(
    local_shift: torch.Tensor,
    row_budget: torch.Tensor,
    order_count: int,
) -> List[torch.Tensor]:
    """Nodewise bounds for ``||(P*)^k H - P^k H||_2`` with ``||H_v||<=1``."""
    if order_count < 1:
        raise ValueError("order_count must be positive")
    values: List[torch.Tensor] = [torch.zeros_like(row_budget)]
    for _ in range(1, int(order_count)):
        propagated = torch.sparse.mm(
            local_shift, values[-1].unsqueeze(1)
        ).squeeze(1)
        values.append((propagated + row_budget).clamp(max=2.0))
    return values


def retained_mass_uncertainty_basis(
    local_shift: torch.Tensor,
    row_budget: torch.Tensor,
    order_count: int,
) -> List[torch.Tensor]:
    """Tight bounds using mass retained on observed endpoints.

    With row L1 budget ``delta``, the completed row keeps
    ``a = 1 - delta / 2`` of its mass on retained endpoints.  Hence the
    previous-order error contracts by ``a``.  The recurrence is also sound
    for an upper row budget because ``a(delta) * t + delta`` is monotone in
    ``delta`` for every intermediate error ``t`` in ``[0, 2]``.
    """
    if order_count < 1:
        raise ValueError("order_count must be positive")
    if bool(((row_budget < 0.0) | (row_budget > 2.0)).any()):
        raise ValueError("row budgets must lie in [0, 2]")
    retained_mass = 1.0 - 0.5 * row_budget
    values: List[torch.Tensor] = [torch.zeros_like(row_budget)]
    for _ in range(1, int(order_count)):
        propagated = torch.sparse.mm(
            local_shift, values[-1].unsqueeze(1)
        ).squeeze(1)
        values.append((retained_mass * propagated + row_budget).clamp(max=2.0))
    return values


def completion_uncertainty_basis(
    local_shift: torch.Tensor,
    exact_row_budget: torch.Tensor,
    missing_degree: torch.Tensor,
    order_count: int,
    *,
    information: Literal[
        "nodewise_exact", "boundary_presence", "global_max", "universal"
    ],
    global_max_row_budget: float | None = None,
    recurrence: Literal["legacy", "retained_mass"] = "legacy",
) -> List[torch.Tensor]:
    """Construct sound completion-set bases under progressively coarser ledgers.

    ``nodewise_exact`` uses the proposed per-node row budget.  The remaining
    modes are deliberately conservative same-threat-model baselines:

    * ``boundary_presence`` knows only whether a node has an unknown neighbor,
      so every exposed row receives the worst possible L1 budget two;
    * ``global_max`` assigns the largest exact row budget in the federation to
      every row and therefore discards nodewise localization;
    * ``universal`` assumes the maximum difference two at every positive
      propagation order and needs no boundary ledger at all.

    All four bound the same endpoint-blind completion set.  They change only
    certificate information and never model logits or training state.
    """
    if exact_row_budget.shape != missing_degree.shape:
        raise ValueError("row budget and missing degree must have equal shape")
    if information == "nodewise_exact":
        row_budget = exact_row_budget
    elif information == "boundary_presence":
        row_budget = 2.0 * (missing_degree > 0).to(exact_row_budget.dtype)
    elif information == "global_max":
        if global_max_row_budget is None:
            raise ValueError("global_max requires global_max_row_budget")
        value = float(global_max_row_budget)
        if not 0.0 <= value <= 2.0:
            raise ValueError("global maximum row budget must lie in [0, 2]")
        row_budget = torch.full_like(exact_row_budget, value)
    elif information == "universal":
        if order_count < 1:
            raise ValueError("order_count must be positive")
        return [torch.zeros_like(exact_row_budget)] + [
            torch.full_like(exact_row_budget, 2.0)
            for _ in range(1, int(order_count))
        ]
    else:
        raise ValueError(f"unsupported completion information: {information}")
    if recurrence == "legacy":
        return uncertainty_basis(local_shift, row_budget, int(order_count))
    if recurrence == "retained_mass":
        return retained_mass_uncertainty_basis(
            local_shift, row_budget, int(order_count)
        )
    raise ValueError(f"unsupported uncertainty recurrence: {recurrence}")


class PolynomialSpectralClassifier(nn.Module):
    """A bounded encoder followed by a power-basis spectral graph filter."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        order_count: int,
        dropout: float,
        encoder_type: str = "linear_relu",
        encoder_depth: int = 1,
        encoder_residual_scale: float = 0.5,
        utility_diffusion_hops: int = 1,
        utility_diffusion_scale: float = 0.25,
        spectral_fusion_scale: float | None = None,
        certificate_margin_mode: Literal[
            "classwise_sum", "pairwise_difference"
        ] = "classwise_sum",
    ) -> None:
        super().__init__()
        if order_count < 2:
            raise ValueError("order_count must be at least two")
        if encoder_type not in {
            "linear_relu",
            "bounded_residual_mlp",
            "feature_only_mlp",
            "dual_feature_spectral",
            "dual_feature_identity_null_spectral",
            "shielded_sage_spectral",
            "shielded_diffusion_spectral",
        }:
            raise ValueError(f"unsupported encoder_type: {encoder_type}")
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be positive")
        if encoder_type == "linear_relu" and encoder_depth != 1:
            raise ValueError("linear_relu requires encoder_depth=1")
        if encoder_type == "bounded_residual_mlp" and encoder_depth < 2:
            raise ValueError("bounded_residual_mlp requires encoder_depth>=2")
        if encoder_type == "feature_only_mlp" and encoder_depth < 2:
            raise ValueError("feature_only_mlp requires encoder_depth>=2")
        if encoder_type == "dual_feature_spectral" and encoder_depth < 2:
            raise ValueError("dual_feature_spectral requires encoder_depth>=2")
        if (
            encoder_type == "dual_feature_identity_null_spectral"
            and encoder_depth < 2
        ):
            raise ValueError(
                "dual_feature_identity_null_spectral requires encoder_depth>=2"
            )
        if encoder_type == "shielded_sage_spectral" and encoder_depth < 2:
            raise ValueError("shielded_sage_spectral requires encoder_depth>=2")
        if encoder_type == "shielded_diffusion_spectral" and encoder_depth < 2:
            raise ValueError("shielded_diffusion_spectral requires encoder_depth>=2")
        if encoder_residual_scale <= 0:
            raise ValueError("encoder_residual_scale must be positive")
        if utility_diffusion_hops < 1:
            raise ValueError("utility_diffusion_hops must be positive")
        if utility_diffusion_scale <= 0:
            raise ValueError("utility_diffusion_scale must be positive")
        if spectral_fusion_scale is not None and spectral_fusion_scale < 0:
            raise ValueError("spectral_fusion_scale must be nonnegative")
        if certificate_margin_mode not in {
            "classwise_sum", "pairwise_difference"
        }:
            raise ValueError("unsupported certificate_margin_mode")
        self.order_count = int(order_count)
        self.dropout = float(dropout)
        self.encoder_type = str(encoder_type)
        self.encoder_depth = int(encoder_depth)
        self.encoder_residual_scale = float(encoder_residual_scale)
        self.utility_diffusion_hops = int(utility_diffusion_hops)
        self.utility_diffusion_scale = float(utility_diffusion_scale)
        self.spectral_fusion_scale = float(
            spectral_fusion_scale
            if spectral_fusion_scale is not None
            else (
                0.0
                if self.encoder_type == "feature_only_mlp"
                else 0.25
                if self.encoder_type in {
                    "dual_feature_spectral",
                    "dual_feature_identity_null_spectral",
                }
                else 1.0
            )
        )
        self.certificate_margin_mode = str(certificate_margin_mode)
        self.encoder = nn.Linear(int(input_dim), int(hidden_dim))
        self.encoder_blocks = nn.ModuleList(
            nn.Linear(int(hidden_dim), int(hidden_dim))
            for _ in range(
                self.encoder_depth - 1
                if self.encoder_type in {
                    "bounded_residual_mlp",
                    "feature_only_mlp",
                }
                else 0
            )
        )
        self.encoder_norms = nn.ModuleList(
            nn.LayerNorm(int(hidden_dim))
            for _ in range(
                self.encoder_depth - 1
                if self.encoder_type == "feature_only_mlp"
                else 0
            )
        )
        theta = torch.zeros(self.order_count, dtype=torch.float32)
        if self.encoder_type == "dual_feature_identity_null_spectral":
            # The identity-null branch is a genuine structural residual:
            # sum_{k>=1} theta_k (P^k-I)H.  It must contain neither a k=0
            # term nor a classifier bias, otherwise P=I would still add
            # graph-independent capacity and reintroduce the E19 confound.
            theta[1] = 0.5
        else:
            theta[0] = 0.5
            theta[1] = 0.5
        if self.encoder_type == "feature_only_mlp":
            self.register_buffer("theta", theta)
        else:
            self.theta = nn.Parameter(theta)
        self.classifier = nn.Linear(
            int(hidden_dim),
            int(num_classes),
            bias=self.encoder_type != "dual_feature_identity_null_spectral",
        )
        # E10 adds a graph-invariant feature expert without changing the
        # certified spectral path above.  It is intentionally absent for all
        # historical encoder types, preserving every E0--E9 state dictionary.
        if self.encoder_type in {
            "dual_feature_spectral",
            "dual_feature_identity_null_spectral",
            "shielded_diffusion_spectral",
        }:
            self.feature_encoder: nn.Linear | None = nn.Linear(
                int(input_dim), int(hidden_dim)
            )
            self.feature_blocks = nn.ModuleList(
                nn.Linear(int(hidden_dim), int(hidden_dim))
                for _ in range(self.encoder_depth - 1)
            )
            self.feature_norms = nn.ModuleList(
                nn.LayerNorm(int(hidden_dim))
                for _ in range(self.encoder_depth - 1)
            )
            self.feature_classifier: nn.Linear | None = nn.Linear(
                int(hidden_dim), int(num_classes)
            )
        else:
            self.feature_encoder = None
            self.feature_blocks = nn.ModuleList()
            self.feature_norms = nn.ModuleList()
            self.feature_classifier = None

        # E11 separates the final prediction function from the certified
        # spectral expert.  The utility expert is a local mean-GraphSAGE
        # network.  At evaluation time its class is used everywhere, while a
        # node receives a certificate only when that class agrees with an
        # independently certified spectral class.  This agreement shield
        # preserves the utility prediction exactly and never certifies the
        # graph-sensitive utility branch itself.
        if self.encoder_type == "shielded_sage_spectral":
            self.utility_self_in: nn.Linear | None = nn.Linear(
                int(input_dim), int(hidden_dim)
            )
            self.utility_neigh_in: nn.Linear | None = nn.Linear(
                int(input_dim), int(hidden_dim), bias=False
            )
            self.utility_self_blocks = nn.ModuleList(
                nn.Linear(int(hidden_dim), int(hidden_dim))
                for _ in range(self.encoder_depth - 2)
            )
            self.utility_neigh_blocks = nn.ModuleList(
                nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
                for _ in range(self.encoder_depth - 2)
            )
            self.utility_norms = nn.ModuleList(
                nn.LayerNorm(int(hidden_dim))
                for _ in range(self.encoder_depth - 2)
            )
            self.utility_self_out: nn.Linear | None = nn.Linear(
                int(hidden_dim), int(num_classes)
            )
            self.utility_neigh_out: nn.Linear | None = nn.Linear(
                int(hidden_dim), int(num_classes), bias=False
            )
        else:
            self.utility_self_in = None
            self.utility_neigh_in = None
            self.utility_self_blocks = nn.ModuleList()
            self.utility_neigh_blocks = nn.ModuleList()
            self.utility_norms = nn.ModuleList()
            self.utility_self_out = None
            self.utility_neigh_out = None

        # E12 retains the E10 feature expert and adds label-free local
        # diffusion features as a residual in representation space.  Raw
        # features are propagated before projection, avoiding the repeated
        # nonlinear mean aggregation that failed in E11.  This utility path
        # is deliberately outside the certificate; soundness comes from the
        # already-audited agreement shield.
        if self.encoder_type == "shielded_diffusion_spectral":
            self.diffusion_projections = nn.ModuleList(
                nn.Linear(int(input_dim), int(hidden_dim), bias=False)
                for _ in range(self.utility_diffusion_hops)
            )
            self.diffusion_norm = nn.LayerNorm(int(hidden_dim))
        else:
            self.diffusion_projections = nn.ModuleList()
            self.diffusion_norm = None

    def bounded_encoding(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.encoder(features))
        if self.encoder_type == "linear_relu":
            # Preserve the frozen E0--E8 implementation exactly.  Dropout is
            # disabled when certificates are evaluated, so evaluated rows
            # still satisfy the unit-norm premise.
            hidden = F.normalize(hidden, p=2, dim=1, eps=1e-12)
            return F.dropout(hidden, p=self.dropout, training=self.training)

        for layer in self.encoder_blocks:
            update = F.gelu(layer(hidden))
            update = F.dropout(update, p=self.dropout, training=self.training)
            hidden = hidden + self.encoder_residual_scale * update
        # E9 residual encoders normalize after every stochastic operation.
        # Hence ||h_v||_2 <= 1 in both training and evaluation modes, and the
        # existing endpoint-blind certificate needs no theoretical change.
        return F.normalize(hidden, p=2, dim=1, eps=1e-12)

    def filter(self, hidden: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "dual_feature_identity_null_spectral":
            base = hidden
            current = hidden
            output = torch.zeros_like(hidden)
            for order in range(1, self.order_count):
                current = torch.sparse.mm(shift, current)
                output = output + self.theta[order] * (current - base)
            return output
        current = hidden
        output = self.theta[0] * current
        for order in range(1, self.order_count):
            current = torch.sparse.mm(shift, current)
            output = output + self.theta[order] * current
        return output

    def feature_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """Encode local node attributes without consulting any graph edge.

        This branch may be expressive and need not be norm bounded: local
        features are fixed throughout the boundary-completion set, so its
        logits have exactly zero boundary sensitivity.
        """
        if self.feature_encoder is None:
            raise RuntimeError("feature expert is disabled for this encoder type")
        hidden = F.gelu(self.feature_encoder(features))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        for layer, norm in zip(self.feature_blocks, self.feature_norms):
            update = F.gelu(layer(hidden))
            update = F.dropout(update, p=self.dropout, training=self.training)
            hidden = norm(hidden + self.encoder_residual_scale * update)
        return hidden

    def feature_only_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """The exact two-layer MLP used by the E14 pure-feature baseline."""
        if self.encoder_type != "feature_only_mlp":
            raise RuntimeError("feature-only MLP is disabled")
        hidden = F.gelu(self.encoder(features))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        for layer, norm in zip(self.encoder_blocks, self.encoder_norms):
            update = F.gelu(layer(hidden))
            update = F.dropout(update, p=self.dropout, training=self.training)
            hidden = norm(hidden + self.encoder_residual_scale * update)
        return hidden

    def spectral_logits(
        self, features: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.bounded_encoding(features)
        return self.classifier(self.filter(hidden, shift))

    @property
    def has_agreement_shield(self) -> bool:
        return self.encoder_type in {
            "shielded_sage_spectral",
            "shielded_diffusion_spectral",
        }

    def utility_logits(
        self, features: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """Return the E11 local GraphSAGE utility-expert logits."""
        if self.encoder_type == "shielded_diffusion_spectral":
            return self.diffusion_feature_logits(features, shift) + (
                0.25 * self.spectral_logits(features, shift)
            )
        if (
            self.utility_self_in is None
            or self.utility_neigh_in is None
            or self.utility_self_out is None
            or self.utility_neigh_out is None
        ):
            raise RuntimeError("utility expert is disabled for this encoder type")
        neighbour = torch.sparse.mm(shift, features)
        hidden = F.gelu(
            self.utility_self_in(features) + self.utility_neigh_in(neighbour)
        )
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        for self_layer, neighbour_layer, norm in zip(
            self.utility_self_blocks,
            self.utility_neigh_blocks,
            self.utility_norms,
        ):
            neighbour = torch.sparse.mm(shift, hidden)
            update = F.gelu(
                self_layer(hidden) + neighbour_layer(neighbour)
            )
            update = F.dropout(update, p=self.dropout, training=self.training)
            hidden = norm(hidden + self.encoder_residual_scale * update)
        neighbour = torch.sparse.mm(shift, hidden)
        return self.utility_self_out(hidden) + self.utility_neigh_out(neighbour)

    def diffusion_feature_logits(
        self, features: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """E12 feature-expert logits after label-free diffusion residuals."""
        if self.encoder_type == "shielded_diffusion_spectral":
            if self.feature_classifier is None or self.diffusion_norm is None:
                raise RuntimeError("diffusion utility expert is incomplete")
            base = self.feature_encoding(features)
            current = features
            residual = torch.zeros_like(base)
            for projection in self.diffusion_projections:
                current = torch.sparse.mm(shift, current)
                residual = residual + F.gelu(projection(current))
            residual = residual / float(len(self.diffusion_projections))
            utility_hidden = self.diffusion_norm(
                base + self.utility_diffusion_scale * residual
            )
            utility_hidden = F.dropout(
                utility_hidden, p=self.dropout, training=self.training
            )
            return self.feature_classifier(utility_hidden)
        raise RuntimeError("diffusion utility expert is disabled")

    def training_logits(
        self, features: torch.Tensor, shift: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused, certificate, feature, and spectral logits.

        For the dual model, callers train the two experts with separate
        classification losses. The invariant feature logits still contribute
        to the certified fused margin, but are detached for that auxiliary
        objective. This prevents both direct and indirect certificate-driven
        changes to the utility expert.
        """
        if self.encoder_type == "feature_only_mlp":
            feature = self.classifier(self.feature_only_encoding(features))
            # The detached copy prevents a certificate objective from
            # regularizing a predictor whose graph sensitivity is exactly zero.
            return feature, feature.detach(), feature, feature.detach()
        spectral = self.spectral_logits(features, shift)
        if self.encoder_type == "shielded_diffusion_spectral":
            feature = self.diffusion_feature_logits(features, shift)
            utility = feature + 0.25 * spectral
            # As in E10, classification losses are applied separately to the
            # feature/diffusion and spectral experts.  The fused utility class
            # is used only for prediction and agreement routing.
            return utility, spectral, feature, spectral
        if self.has_agreement_shield:
            utility = self.utility_logits(features, shift)
            # The certificate loss only sees the bounded spectral expert.
            # The first return value is the ordinary utility prediction used
            # for checkpoint selection; the agreement shield is applied by
            # the evaluation helpers once the nodewise certificate is known.
            return utility, spectral, utility, spectral
        if self.feature_classifier is None:
            return spectral, spectral, spectral, spectral
        feature = self.feature_classifier(self.feature_encoding(features))
        fused = feature + self.spectral_fusion_scale * spectral
        certificate = feature.detach() + self.spectral_fusion_scale * spectral
        return fused, certificate, feature, spectral

    def forward(self, features: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        logits, _, _, _ = self.training_logits(features, shift)
        return logits

    def boundary_radius(self, basis: List[torch.Tensor]) -> torch.Tensor:
        if len(basis) != self.order_count:
            raise ValueError("uncertainty basis does not match filter order")
        radius = sum(
            self.theta[order].abs() * basis[order]
            for order in range(1, self.order_count)
        )
        return self.spectral_fusion_scale * radius

    def sensitivity(self) -> torch.Tensor:
        weights = torch.arange(
            self.order_count,
            device=self.theta.device,
            dtype=self.theta.dtype,
        )
        return torch.sum(weights * self.theta.abs())


@dataclass
class CertificateStatistics:
    certified: torch.Tensor
    worst_slack: torch.Tensor
    representation_radius: torch.Tensor
    class_logit_bounds: torch.Tensor


@dataclass
class PredictionCertificateStatistics:
    """Final logits and certificate metadata after the E11 agreement shield."""

    logits: torch.Tensor
    safe_logits: torch.Tensor
    utility_logits: torch.Tensor
    certificate: CertificateStatistics
    shield: torch.Tensor


def certificate_statistics(
    model: PolynomialSpectralClassifier,
    logits: torch.Tensor,
    basis: List[torch.Tensor],
) -> CertificateStatistics:
    radius = model.boundary_radius(basis)
    class_norm = model.classifier.weight.norm(p=2, dim=1)
    class_bounds = radius[:, None] * class_norm[None, :]
    predicted = logits.argmax(dim=1)
    predicted_logits = logits.gather(1, predicted[:, None])
    if model.certificate_margin_mode == "pairwise_difference":
        predicted_weights = model.classifier.weight[predicted]
        pairwise_norm = (
            predicted_weights[:, None, :] - model.classifier.weight[None, :, :]
        ).norm(p=2, dim=2)
        slack = predicted_logits - logits - radius[:, None] * pairwise_norm
    else:
        predicted_bounds = class_bounds.gather(1, predicted[:, None])
        slack = predicted_logits - logits - predicted_bounds - class_bounds
    own_class = F.one_hot(predicted, num_classes=logits.size(1)).bool()
    slack = slack.masked_fill(own_class, float("inf"))
    worst_slack = slack.min(dim=1).values
    # With a zero spectral coefficient, every output logit is exactly graph
    # invariant.  Deterministic argmax therefore stays unchanged even when
    # two nominal logits tie; a strict positive-margin test would be needlessly
    # incomplete for this explicit pure-feature degeneration.
    certified = (
        torch.ones_like(worst_slack, dtype=torch.bool)
        if model.spectral_fusion_scale == 0.0
        else worst_slack > 0.0
    )
    return CertificateStatistics(
        certified=certified,
        worst_slack=worst_slack,
        representation_radius=radius,
        class_logit_bounds=class_bounds,
    )


def positive_orthant_unit_support(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact support interval over ``{h >= 0 : ||h||_2 <= 1}``.

    The bounded spectral encoder used by the frozen linear and dual-expert
    models is a ReLU followed by row-wise L2 normalization.  Its range is
    therefore a strict subset of the full unit ball used by the historical
    symmetric certificate.  For a direction ``q``, the exact extrema are
    ``-||q_-||_2`` and ``||q_+||_2``.
    """
    if direction.ndim < 1:
        raise ValueError("direction must have at least one dimension")
    upper = direction.clamp_min(0.0).norm(p=2, dim=-1)
    lower = -(-direction).clamp_min(0.0).norm(p=2, dim=-1)
    return lower, upper


def cone_pairwise_certificate_statistics(
    model: PolynomialSpectralClassifier,
    features: torch.Tensor,
    shift: torch.Tensor,
    row_budget: torch.Tensor,
    logits: torch.Tensor,
    basis: List[torch.Tensor],
    *,
    count_semantics: Literal["exact", "upper_bound"] = "upper_bound",
    combine_symmetric: bool = True,
) -> CertificateStatistics:
    """One-sided class-pair certificate using the ReLU encoder range.

    For every ordered class pair, this routine propagates lower and upper
    scalar support values instead of a direction-free Euclidean radius.  A
    completed local row retains ``1-delta/2`` of the observed transition mass;
    its remaining mass may select any remote hidden row in the positive
    orthant unit ball.  ``upper_bound`` additionally takes the endpoint at zero
    missing stubs, so it is sound when the ledger stores only ``b_actual <= b``.

    ``combine_symmetric`` takes the union of two independently sound
    certificates (the historical retained-mass pairwise certificate and the
    cone certificate).  This makes the new certificate monotonically no worse
    than the frozen control while retaining the cone gain where it is tighter.
    """
    if count_semantics not in {"exact", "upper_bound"}:
        raise ValueError(f"unsupported count semantics: {count_semantics}")
    if logits.ndim != 2 or logits.size(0) != features.size(0):
        raise ValueError("logits and features must be node aligned")
    if row_budget.ndim != 1 or row_budget.numel() != features.size(0):
        raise ValueError("row budget must contain one value per node")
    if bool(((row_budget < 0.0) | (row_budget > 2.0)).any()):
        raise ValueError("row budgets must lie in [0, 2]")

    symmetric = certificate_statistics(model, logits, basis)
    if model.spectral_fusion_scale == 0.0:
        return symmetric

    hidden = model.bounded_encoding(features)
    # GELU residual encoders can leave the positive cone and must continue to
    # use the symmetric unit-ball certificate.  A small tolerance only absorbs
    # floating-point representations of exact ReLU zeros.
    if bool((hidden < -32.0 * torch.finfo(hidden.dtype).eps).any()):
        raise ValueError("cone certificate requires nonnegative bounded encodings")
    if bool((hidden.norm(p=2, dim=1) > 1.0 + 64.0 * torch.finfo(hidden.dtype).eps).any()):
        raise ValueError("cone certificate requires hidden row norm at most one")

    spectral_logits = model.classifier(model.filter(hidden, shift))
    predicted = logits.argmax(dim=1)
    class_count = int(logits.size(1))
    cone_slack = torch.full_like(logits, float("-inf"))
    retained = (1.0 - 0.5 * row_budget).clamp(min=0.0, max=1.0)[:, None]
    boundary = 1.0 - retained

    # A direction must remain fixed while it is propagated through P.  We
    # therefore evaluate one predicted class at a time and all competitors in
    # a single sparse-matrix multiply.  Memory is O(|V| C), not O(|V| C^2).
    for predicted_class in predicted.unique(sorted=True).tolist():
        pair_direction = (
            model.classifier.weight[int(predicted_class)][None, :]
            - model.classifier.weight
        )
        remote_lower, remote_upper = positive_orthant_unit_support(pair_direction)
        current = hidden @ pair_direction.transpose(0, 1)
        lower = current
        upper = current
        identity_null = (
            model.encoder_type == "dual_feature_identity_null_spectral"
        )
        base_pair = current
        if identity_null:
            polynomial_lower = torch.zeros_like(current)
        else:
            coefficient = model.theta[0]
            polynomial_lower = coefficient * (
                lower if float(coefficient.detach().item()) >= 0.0 else upper
            )
        for order in range(1, model.order_count):
            local_lower = torch.sparse.mm(shift, lower)
            local_upper = torch.sparse.mm(shift, upper)
            completed_lower = retained * local_lower + boundary * remote_lower[None, :]
            completed_upper = retained * local_upper + boundary * remote_upper[None, :]
            if count_semantics == "upper_bound":
                # The actual boundary count may be any integer from zero to
                # the ledger upper bound.  The affine mass interpolation has
                # extrema at its two endpoints.
                lower = torch.minimum(local_lower, completed_lower)
                upper = torch.maximum(local_upper, completed_upper)
            else:
                lower = completed_lower
                upper = completed_upper
            coefficient = model.theta[order]
            selected_endpoint = (
                lower if float(coefficient.detach().item()) >= 0.0 else upper
            )
            if identity_null:
                selected_endpoint = selected_endpoint - base_pair
            polynomial_lower = polynomial_lower + coefficient * selected_endpoint

        bias = model.classifier.bias
        if bias is not None:
            polynomial_lower = polynomial_lower + (
                bias[int(predicted_class)] - bias
            )[None, :]
        nominal_spectral_pair = (
            spectral_logits[:, int(predicted_class), None] - spectral_logits
        )
        nominal_fused_pair = logits[:, int(predicted_class), None] - logits
        alpha = float(model.spectral_fusion_scale)
        invariant_pair = nominal_fused_pair - alpha * nominal_spectral_pair
        completed_fused_lower = invariant_pair + alpha * polynomial_lower
        active = predicted == int(predicted_class)
        cone_slack[active] = completed_fused_lower[active]

    own_class = F.one_hot(predicted, num_classes=class_count).bool()
    cone_slack = cone_slack.masked_fill(own_class, float("inf"))
    cone_worst = cone_slack.min(dim=1).values
    worst_slack = (
        torch.maximum(cone_worst, symmetric.worst_slack)
        if combine_symmetric
        else cone_worst
    )
    return CertificateStatistics(
        certified=worst_slack > 0.0,
        worst_slack=worst_slack,
        # Keep the historical symmetric classwise bounds for the independent
        # full-completion logit audit.  The new decision itself is pairwise and
        # one-sided, so no false classwise interpretation is introduced.
        representation_radius=symmetric.representation_radius,
        class_logit_bounds=symmetric.class_logit_bounds,
    )


def prediction_certificate_statistics(
    model: PolynomialSpectralClassifier,
    features: torch.Tensor,
    shift: torch.Tensor,
    basis: List[torch.Tensor],
) -> PredictionCertificateStatistics:
    """Evaluate ordinary predictions and sound certificates for one graph.

    Historical models return their usual logits and certificates unchanged.
    For E11, a node is routed to the safe spectral expert iff the safe expert
    is certified and its nominal class agrees with the utility expert.  The
    two local classes therefore remain identical before and after routing.
    Nodes outside this fixed shield are deliberately marked uncertified.
    """
    safe_logits = model.spectral_logits(features, shift)
    safe_stats = certificate_statistics(model, safe_logits, basis)
    if not model.has_agreement_shield:
        logits = model(features, shift)
        return PredictionCertificateStatistics(
            logits=logits,
            safe_logits=logits,
            utility_logits=logits,
            certificate=certificate_statistics(model, logits, basis),
            shield=safe_stats.certified,
        )

    utility_logits = model.utility_logits(features, shift)
    agreement = safe_logits.argmax(dim=1) == utility_logits.argmax(dim=1)
    shield = safe_stats.certified & agreement
    logits = torch.where(shield[:, None], safe_logits, utility_logits)
    certificate = CertificateStatistics(
        certified=shield,
        worst_slack=safe_stats.worst_slack,
        representation_radius=safe_stats.representation_radius,
        class_logit_bounds=torch.where(
            shield[:, None],
            safe_stats.class_logit_bounds,
            torch.full_like(safe_stats.class_logit_bounds, float("inf")),
        ),
    )
    return PredictionCertificateStatistics(
        logits=logits,
        safe_logits=safe_logits,
        utility_logits=utility_logits,
        certificate=certificate,
        shield=shield,
    )


def external_teacher_agreement_statistics(
    model: PolynomialSpectralClassifier,
    features: torch.Tensor,
    shift: torch.Tensor,
    basis: List[torch.Tensor],
    teacher_logits: torch.Tensor,
    *,
    certificate_kind: Literal["symmetric", "cone_hybrid"] = "cone_hybrid",
) -> PredictionCertificateStatistics:
    """Certify an immutable external utility expert through class agreement.

    The external logits are never differentiated and are used only for the
    client's own nodes.  Nominal routing selects the safe expert exactly when
    its certified class equals the teacher class.  Consequently, the routed
    prediction class equals the strong teacher class at every node, while
    only agreement-shielded nodes receive a completion-set certificate.
    """
    if teacher_logits.ndim != 2 or teacher_logits.size(0) != features.size(0):
        raise ValueError("external teacher logits must be node aligned")
    if not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError("external teacher logits contain non-finite values")
    if certificate_kind not in {"symmetric", "cone_hybrid"}:
        raise ValueError(f"unsupported external certificate: {certificate_kind}")

    teacher_logits = teacher_logits.detach()
    if model.encoder_type == "feature_only_mlp":
        safe_logits = model(features, shift)
    else:
        safe_logits = model.spectral_logits(features, shift)
    if safe_logits.shape != teacher_logits.shape:
        raise ValueError("safe and external teacher class dimensions differ")

    if certificate_kind == "cone_hybrid" and model.encoder_type != "feature_only_mlp":
        safe_stats = cone_pairwise_certificate_statistics(
            model,
            features,
            shift,
            basis[1],
            safe_logits,
            basis,
            count_semantics="upper_bound",
            combine_symmetric=True,
        )
    else:
        safe_stats = certificate_statistics(model, safe_logits, basis)

    agreement = safe_logits.argmax(dim=1) == teacher_logits.argmax(dim=1)
    shield = safe_stats.certified & agreement
    routed_logits = torch.where(shield[:, None], safe_logits, teacher_logits)
    certificate = CertificateStatistics(
        certified=shield,
        worst_slack=safe_stats.worst_slack,
        representation_radius=safe_stats.representation_radius,
        class_logit_bounds=torch.where(
            shield[:, None],
            safe_stats.class_logit_bounds,
            torch.full_like(safe_stats.class_logit_bounds, float("inf")),
        ),
    )
    if not torch.equal(
        routed_logits.argmax(dim=1), teacher_logits.argmax(dim=1)
    ):
        raise RuntimeError("agreement routing changed the external teacher class")
    return PredictionCertificateStatistics(
        logits=routed_logits,
        safe_logits=safe_logits,
        utility_logits=teacher_logits,
        certificate=certificate,
        shield=shield,
    )


def certified_hinge_loss(
    model: PolynomialSpectralClassifier,
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    basis: List[torch.Tensor],
    margin: float,
) -> torch.Tensor:
    radius = model.boundary_radius(basis)
    true_logits = logits.gather(1, labels[:, None])
    if model.certificate_margin_mode == "pairwise_difference":
        true_weights = model.classifier.weight[labels]
        pairwise_norm = (
            true_weights[:, None, :] - model.classifier.weight[None, :, :]
        ).norm(p=2, dim=2)
        slack = true_logits - logits - radius[:, None] * pairwise_norm
    else:
        class_norm = model.classifier.weight.norm(p=2, dim=1)
        true_bounds = radius * class_norm[labels]
        slack = true_logits - logits - true_bounds[:, None] - (
            radius[:, None] * class_norm[None, :]
        )
    true_class = F.one_hot(labels, num_classes=logits.size(1)).bool()
    slack = slack.masked_fill(true_class, float("inf"))
    worst_slack = slack.min(dim=1).values
    return F.relu(float(margin) - worst_slack[mask]).mean()


def teacher_target_certified_hinge_loss(
    model: PolynomialSpectralClassifier,
    safe_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    basis: List[torch.Tensor],
    margin: float,
) -> torch.Tensor:
    """Make the safe expert certify the frozen utility expert's class.

    This is completion-certified agreement distillation (CAD), not ordinary
    logit matching: the target class must beat every competitor after the
    analytic completion radius is subtracted.  Teacher logits are detached by
    construction, so no gradient or state can flow into the utility expert.
    """
    if safe_logits.shape != teacher_logits.shape:
        raise ValueError("safe and teacher logits must have identical shape")
    if mask.dtype != torch.bool or mask.numel() != safe_logits.size(0):
        raise ValueError("CAD mask must be one boolean value per node")
    teacher_class = teacher_logits.detach().argmax(dim=1)
    return certified_hinge_loss(
        model,
        safe_logits,
        teacher_class,
        mask,
        basis,
        margin,
    )
