from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .certificate import (
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    certificate_statistics,
    certified_hinge_loss,
    cone_pairwise_certificate_statistics,
    external_teacher_agreement_statistics,
    prediction_certificate_statistics,
    retained_mass_uncertainty_basis,
    row_normalized_shift,
    uncertainty_basis,
    teacher_target_certified_hinge_loss,
)


TensorState = Dict[str, torch.Tensor]


def _identity_shift_like(reference: torch.Tensor) -> torch.Tensor:
    """Return an exact sparse identity propagation operator on the same device."""
    node_count = int(reference.size(0))
    diagonal = torch.arange(node_count, device=reference.device)
    indices = torch.stack((diagonal, diagonal), dim=0)
    values = torch.ones(node_count, dtype=reference.dtype, device=reference.device)
    return torch.sparse_coo_tensor(
        indices, values, (node_count, node_count), device=reference.device
    ).coalesce()


def _zero_uncertainty_basis_like(
    basis: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Return the exact zero-radius basis for graph-invariant propagation."""
    return [torch.zeros_like(value) for value in basis]


@dataclass(frozen=True)
class BoundaryLedgerView:
    """Endpoint-blind client input produced by a boundary-stub ledger.

    ``local_edge_index`` is remapped to the client's local node numbering.
    The view contains a count of unknown incident edges for each local node,
    but never a remote endpoint identifier or a cross-client edge pair.
    ``global_nodes`` is retained only by the benchmark auditor so predictions
    can be restored to the original node order; it is not server payload.
    """

    client_id: int
    global_nodes: torch.Tensor
    local_edge_index: torch.Tensor
    missing_degree: torch.Tensor

    @property
    def retained_undirected_edges(self) -> int:
        return int(self.local_edge_index.size(1) // 2)

    @property
    def missing_degree_sum(self) -> int:
        return int(self.missing_degree.sum().item())


def boundary_severity_from_aggregates(
    retained_undirected_edges: int,
    missing_degree_sum: int,
) -> float:
    """Compute the global cut ratio without cross-client endpoint identities."""
    retained = int(retained_undirected_edges)
    missing_sum = int(missing_degree_sum)
    if retained < 0 or missing_sum < 0 or missing_sum % 2:
        raise ValueError("edge counts must be nonnegative and missing sum even")
    cut = missing_sum // 2
    return float(cut / max(retained + cut, 1))


def pairwise_masked_sum(
    client_values: Sequence[Sequence[int]],
    *,
    seed: int,
    modulus: int = 2**61 - 1,
) -> tuple[List[int], dict]:
    """Simulate an honest-but-curious-server pairwise masked secure sum.

    Every client adds/subtracts the same pseudorandom mask for every peer.
    Masks cancel only after server aggregation.  This function validates the
    arithmetic used by the experimental protocol; it is not a production key
    exchange or a differential-privacy mechanism.
    """
    values = [[int(value) for value in row] for row in client_values]
    if len(values) < 2 or not values or not values[0]:
        raise ValueError("at least two nonempty client vectors are required")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("client vectors must have equal width")
    if any(value < 0 or value >= modulus for row in values for value in row):
        raise ValueError("client values must lie in the aggregation field")
    raw_totals = [sum(row[column] for row in values) for column in range(width)]
    if any(total >= modulus for total in raw_totals):
        raise ValueError("aggregate would wrap around the aggregation field")
    shares = [row.copy() for row in values]
    generator = np.random.default_rng(int(seed))
    for left in range(len(shares)):
        for right in range(left + 1, len(shares)):
            for column in range(width):
                mask = int(generator.integers(0, modulus, dtype=np.int64))
                shares[left][column] = (shares[left][column] + mask) % modulus
                shares[right][column] = (shares[right][column] - mask) % modulus
    aggregate = [
        sum(row[column] for row in shares) % modulus for column in range(width)
    ]
    if aggregate != raw_totals:
        raise RuntimeError("pairwise masks did not cancel exactly")
    return aggregate, {
        "aggregation": "pairwise_masked_sum_simulation",
        "client_count": len(values),
        "payload_scalars_per_client": width,
        "total_payload_scalars": len(values) * width,
        "modulus": int(modulus),
        "masks_cancel_exactly": True,
        "individual_unmasked_values_visible_to_server": False,
        "differential_privacy_claimed": False,
    }


def project_auxiliary_gradient(
    primary: Sequence[torch.Tensor],
    auxiliary: Sequence[torch.Tensor],
    epsilon: float = 1e-12,
) -> tuple[List[torch.Tensor], dict]:
    """Project only a conflicting auxiliary gradient off the primary gradient.

    The asymmetric construction keeps the classification gradient unchanged.
    If ``<g_primary, g_auxiliary> < 0``, the returned auxiliary direction is
    projected toward the primary-orthogonal hyperplane.  Because the stable
    denominator contains ``epsilon``, its residual inner product may remain
    slightly negative.  The formal descent guarantee therefore comes from the
    subsequent relative-norm cap, not from exact orthogonality.
    """
    if len(primary) != len(auxiliary) or not primary:
        raise ValueError("gradient collections must be nonempty and aligned")
    dot = sum(
        (left.detach().double() * right.detach().double()).sum()
        for left, right in zip(primary, auxiliary)
    )
    primary_norm_sq = sum(
        left.detach().double().square().sum() for left in primary
    )
    auxiliary_norm_sq = sum(
        right.detach().double().square().sum() for right in auxiliary
    )
    conflict = bool(dot.item() < 0.0 and primary_norm_sq.item() > epsilon)
    if conflict:
        coefficient = dot / (primary_norm_sq + float(epsilon))
        projected = [
            right - coefficient.to(right.dtype) * left
            for left, right in zip(primary, auxiliary)
        ]
    else:
        projected = [right.clone() for right in auxiliary]
    post_dot = sum(
        (left.detach().double() * right.detach().double()).sum()
        for left, right in zip(primary, projected)
    )
    denominator = torch.sqrt(primary_norm_sq * auxiliary_norm_sq).clamp_min(epsilon)
    return projected, {
        "conflict": conflict,
        "pre_projection_dot": float(dot.item()),
        "post_projection_dot": float(post_dot.item()),
        "pre_projection_cosine": float((dot / denominator).item()),
    }


@dataclass
class BoundaryFederatedClient:
    client_id: int
    global_nodes: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    train_mask: torch.Tensor
    validation_mask: torch.Tensor
    test_mask: torch.Tensor
    shift: torch.Tensor
    uncertainty_basis: List[torch.Tensor]
    missing_degree: torch.Tensor

    @property
    def train_count(self) -> int:
        return int(self.train_mask.sum().item())


def cpu_state(model: torch.nn.Module) -> TensorState:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def weighted_average_states(
    states: Sequence[TensorState], sample_counts: Sequence[int]
) -> TensorState:
    if not states or len(states) != len(sample_counts):
        raise ValueError("states and sample_counts must be nonempty and aligned")
    total = float(sum(int(value) for value in sample_counts))
    if total <= 0:
        raise ValueError("the total client training count must be positive")
    output: TensorState = {}
    for key, first in states[0].items():
        if not all(key in state and state[key].shape == first.shape for state in states):
            raise ValueError(f"incompatible client state for {key}")
        if not torch.is_floating_point(first):
            output[key] = first.detach().clone()
            continue
        accumulator = torch.zeros_like(first, dtype=torch.float64)
        for state, count in zip(states, sample_counts):
            accumulator.add_(state[key].double(), alpha=float(count) / total)
        output[key] = accumulator.to(first.dtype)
    return output


def build_federated_clients(
    full_edge_index: torch.Tensor,
    membership: np.ndarray,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    missing_degree: torch.Tensor,
    order_count: int,
    device: torch.device,
    uncertainty_recurrence: str = "legacy",
) -> List[BoundaryFederatedClient]:
    """Compatibility wrapper around endpoint-blind ledger construction."""
    ledgers = build_boundary_ledger_views(
        full_edge_index, membership, missing_degree
    )
    return build_federated_clients_from_ledgers(
        ledgers,
        features,
        labels,
        train_mask,
        validation_mask,
        test_mask,
        order_count,
        device,
        uncertainty_recurrence,
    )


def build_boundary_ledger_views(
    full_edge_index: torch.Tensor,
    membership: np.ndarray,
    missing_degree: torch.Tensor,
) -> List[BoundaryLedgerView]:
    """Convert a benchmark partition into client-local boundary-stub ledgers."""
    membership = np.asarray(membership, dtype=np.int64)
    member_tensor = torch.as_tensor(membership, dtype=torch.long)
    ledgers: List[BoundaryLedgerView] = []
    for client_id in sorted(np.unique(membership).tolist()):
        nodes = torch.as_tensor(
            np.flatnonzero(membership == int(client_id)), dtype=torch.long
        )
        global_to_local = torch.full(
            (int(membership.size),), -1, dtype=torch.long
        )
        global_to_local[nodes] = torch.arange(nodes.numel(), dtype=torch.long)
        edge_keep = (
            (member_tensor[full_edge_index[0]] == int(client_id))
            & (member_tensor[full_edge_index[1]] == int(client_id))
        )
        local_edges_global = full_edge_index[:, edge_keep]
        local_edges = global_to_local[local_edges_global]
        if bool((local_edges < 0).any()):
            raise RuntimeError("ledger contains a remote endpoint")
        ledgers.append(BoundaryLedgerView(
            client_id=int(client_id),
            global_nodes=nodes,
            local_edge_index=local_edges,
            missing_degree=missing_degree[nodes].detach().cpu(),
        ))
    return ledgers


def build_federated_clients_from_ledgers(
    ledgers: Sequence[BoundaryLedgerView],
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    order_count: int,
    device: torch.device,
    uncertainty_recurrence: str = "legacy",
) -> List[BoundaryFederatedClient]:
    """Materialize clients without accepting a full or cross-client edge list."""
    clients: List[BoundaryFederatedClient] = []
    for ledger in ledgers:
        nodes = ledger.global_nodes
        shift, retained_degree = row_normalized_shift(
            ledger.local_edge_index, int(nodes.numel()), device=device
        )
        local_missing = ledger.missing_degree.to(device)
        uncertainty = boundary_uncertainty(retained_degree, local_missing)
        if uncertainty_recurrence == "legacy":
            basis = uncertainty_basis(shift, uncertainty, int(order_count))
        elif uncertainty_recurrence == "retained_mass":
            basis = retained_mass_uncertainty_basis(
                shift, uncertainty, int(order_count)
            )
        else:
            raise ValueError(
                f"unsupported uncertainty recurrence: {uncertainty_recurrence}"
            )
        client = BoundaryFederatedClient(
            client_id=int(ledger.client_id),
            global_nodes=nodes.to(device),
            features=features[nodes].to(device),
            labels=labels[nodes].to(device),
            train_mask=train_mask[nodes].to(device),
            validation_mask=validation_mask[nodes].to(device),
            test_mask=test_mask[nodes].to(device),
            shift=shift,
            uncertainty_basis=basis,
            missing_degree=local_missing,
        )
        if client.train_count <= 0:
            raise RuntimeError(
                f"client {ledger.client_id} has no labelled training nodes"
            )
        clients.append(client)
    return clients


def train_client(
    global_state: TensorState,
    client: BoundaryFederatedClient,
    *,
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
    robust_lambda: float,
    certificate_margin: float,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    spectral_fusion_scale: float | None = None,
    certificate_margin_mode: str = "classwise_sum",
) -> TensorState:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PolynomialSpectralClassifier(
        input_dim,
        hidden_dim,
        num_classes,
        order_count,
        dropout,
        encoder_type,
        encoder_depth,
        encoder_residual_scale,
        utility_diffusion_hops,
        utility_diffusion_scale,
        spectral_fusion_scale,
        certificate_margin_mode,
    ).to(device)
    model.load_state_dict(global_state, strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    for _ in range(int(local_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, certificate_logits, feature_logits, spectral_logits = model.training_logits(
            client.features, client.shift
        )
        if model.feature_classifier is None and not model.has_agreement_shield:
            loss = F.cross_entropy(
                logits[client.train_mask], client.labels[client.train_mask]
            )
        else:
            loss = F.cross_entropy(
                feature_logits[client.train_mask], client.labels[client.train_mask]
            ) + F.cross_entropy(
                spectral_logits[client.train_mask], client.labels[client.train_mask]
            )
        if robust_lambda > 0:
            loss = loss + float(robust_lambda) * certified_hinge_loss(
                model,
                certificate_logits,
                client.labels,
                client.train_mask,
                client.uncertainty_basis,
                float(certificate_margin),
            )
        loss.backward()
        optimizer.step()
    return cpu_state(model)


def train_client_external_teacher(
    global_state: TensorState,
    client: BoundaryFederatedClient,
    teacher_logits: torch.Tensor,
    *,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    order_count: int,
    dropout: float,
    encoder_type: str,
    encoder_depth: int,
    encoder_residual_scale: float,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    loss_mode: str,
    distillation_weight: float,
    distillation_temperature: float,
    cad_weight: float,
    certificate_margin: float,
    certificate_margin_mode: str = "pairwise_difference",
    propagation_mode: str = "observed",
) -> TensorState:
    """Train a safe expert against labels and a frozen strong utility teacher."""
    allowed_modes = {"ce", "kd", "cad", "kd_cad"}
    if loss_mode not in allowed_modes:
        raise ValueError(f"unsupported external-teacher loss mode: {loss_mode}")
    if teacher_logits.shape != (client.features.size(0), int(num_classes)):
        raise ValueError("client teacher logits are not node/class aligned")
    if distillation_temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    if distillation_weight < 0.0 or cad_weight < 0.0:
        raise ValueError("external-teacher loss weights must be nonnegative")
    if propagation_mode not in {"observed", "identity"}:
        raise ValueError(f"unsupported propagation mode: {propagation_mode}")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PolynomialSpectralClassifier(
        input_dim,
        hidden_dim,
        num_classes,
        order_count,
        dropout,
        encoder_type=encoder_type,
        encoder_depth=encoder_depth,
        encoder_residual_scale=encoder_residual_scale,
        certificate_margin_mode=certificate_margin_mode,
    ).to(device)
    model.load_state_dict(global_state, strict=True)
    frozen_teacher = teacher_logits.detach().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    safe_shift = (
        client.shift
        if propagation_mode == "observed"
        else _identity_shift_like(client.shift)
    )
    safe_basis = (
        client.uncertainty_basis
        if propagation_mode == "observed"
        else _zero_uncertainty_basis_like(client.uncertainty_basis)
    )
    for _ in range(int(local_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        safe_logits = (
            model(client.features, safe_shift)
            if encoder_type == "feature_only_mlp"
            else model.spectral_logits(client.features, safe_shift)
        )
        active = client.train_mask
        loss = F.cross_entropy(safe_logits[active], client.labels[active])
        if loss_mode in {"kd", "kd_cad"}:
            temperature = float(distillation_temperature)
            distillation = F.kl_div(
                F.log_softmax(safe_logits[active] / temperature, dim=1),
                F.softmax(frozen_teacher[active] / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = loss + float(distillation_weight) * distillation
        if loss_mode in {"cad", "kd_cad"}:
            cad = teacher_target_certified_hinge_loss(
                model,
                safe_logits,
                frozen_teacher,
                active,
                safe_basis,
                float(certificate_margin),
            )
            loss = loss + float(cad_weight) * cad
        loss.backward()
        optimizer.step()
    return cpu_state(model)


def train_client_identity_null_staged(
    global_state: TensorState,
    client: BoundaryFederatedClient,
    *,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    order_count: int,
    dropout: float,
    encoder_type: str,
    encoder_depth: int,
    encoder_residual_scale: float,
    utility_diffusion_hops: int,
    utility_diffusion_scale: float,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    spectral_fusion_scale: float,
    certificate_margin_mode: str,
    phase: str,
) -> TensorState:
    """Train a feature stage or a fused-loss identity-null correction stage."""
    if encoder_type != "dual_feature_identity_null_spectral":
        raise ValueError("staged residual training requires identity-null encoder")
    if phase not in {"feature_pretrain", "residual_correction"}:
        raise ValueError(f"unsupported staged residual phase: {phase}")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PolynomialSpectralClassifier(
        input_dim,
        hidden_dim,
        num_classes,
        order_count,
        dropout,
        encoder_type,
        encoder_depth,
        encoder_residual_scale,
        utility_diffusion_hops,
        utility_diffusion_scale,
        spectral_fusion_scale,
        certificate_margin_mode,
    ).to(device)
    model.load_state_dict(global_state, strict=True)
    feature_phase = phase == "feature_pretrain"
    for name, parameter in model.named_parameters():
        is_feature_parameter = name.startswith("feature_")
        parameter.requires_grad_(
            is_feature_parameter if feature_phase else not is_feature_parameter
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("staged residual phase selected no trainable parameters")
    optimizer = torch.optim.Adam(
        parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    for _ in range(int(local_epochs)):
        # The correction stage is deterministic: the frozen feature target
        # and structural residual contain no dropout noise, but gradients are
        # still tracked normally in evaluation mode.
        model.train(feature_phase)
        optimizer.zero_grad(set_to_none=True)
        fused, _, feature_logits, _ = model.training_logits(
            client.features, client.shift
        )
        target_logits = feature_logits if feature_phase else fused
        loss = F.cross_entropy(
            target_logits[client.train_mask], client.labels[client.train_mask]
        )
        loss.backward()
        optimizer.step()
    return cpu_state(model)


def train_client_projected(
    global_state: TensorState,
    client: BoundaryFederatedClient,
    *,
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
    robust_lambda: float,
    projection_ratio_cap: float,
    certificate_margin: float,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    spectral_fusion_scale: float | None = None,
    certificate_margin_mode: str = "classwise_sum",
) -> tuple[TensorState, dict]:
    if robust_lambda <= 0:
        raise ValueError("projected training requires a positive robust weight")
    if projection_ratio_cap <= 0:
        raise ValueError("projection_ratio_cap must be positive")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PolynomialSpectralClassifier(
        input_dim,
        hidden_dim,
        num_classes,
        order_count,
        dropout,
        encoder_type,
        encoder_depth,
        encoder_residual_scale,
        utility_diffusion_hops,
        utility_diffusion_scale,
        spectral_fusion_scale,
        certificate_margin_mode,
    ).to(device)
    model.load_state_dict(global_state, strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    conflict_count = 0
    cosine_sum = 0.0
    minimum_post_dot = float("inf")
    capped_count = 0
    applied_scale_sum = 0.0
    step_count = 0
    for _ in range(int(local_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, certificate_logits, feature_logits, spectral_logits = model.training_logits(
            client.features, client.shift
        )
        if model.feature_classifier is None and not model.has_agreement_shield:
            primary_loss = F.cross_entropy(
                logits[client.train_mask], client.labels[client.train_mask]
            )
        else:
            primary_loss = F.cross_entropy(
                feature_logits[client.train_mask], client.labels[client.train_mask]
            ) + F.cross_entropy(
                spectral_logits[client.train_mask], client.labels[client.train_mask]
            )
        auxiliary_loss = certified_hinge_loss(
            model,
            certificate_logits,
            client.labels,
            client.train_mask,
            client.uncertainty_basis,
            float(certificate_margin),
        )
        primary_raw = torch.autograd.grad(
            primary_loss, parameters, retain_graph=True, allow_unused=True
        )
        auxiliary_raw = torch.autograd.grad(
            auxiliary_loss, parameters, allow_unused=True
        )
        primary = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, primary_raw)
        ]
        auxiliary = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, auxiliary_raw)
        ]
        projected, diagnostics = project_auxiliary_gradient(primary, auxiliary)
        primary_norm = torch.sqrt(
            sum(gradient.detach().double().square().sum() for gradient in primary)
        )
        projected_norm = torch.sqrt(
            sum(gradient.detach().double().square().sum() for gradient in projected)
        )
        allowed_scale = (
            float(projection_ratio_cap) * primary_norm
            / (float(robust_lambda) * projected_norm + 1e-12)
        )
        applied_scale = min(1.0, float(allowed_scale.item()))
        capped_count += int(applied_scale < 1.0 - 1e-12)
        applied_scale_sum += applied_scale
        for parameter, primary_gradient, auxiliary_gradient in zip(
            parameters, primary, projected
        ):
            parameter.grad = (
                primary_gradient
                + float(robust_lambda) * applied_scale * auxiliary_gradient
            ).detach()
        optimizer.step()
        step_count += 1
        conflict_count += int(diagnostics["conflict"])
        cosine_sum += diagnostics["pre_projection_cosine"]
        minimum_post_dot = min(
            minimum_post_dot, diagnostics["post_projection_dot"]
        )
    return cpu_state(model), {
        "projection_step_count": step_count,
        "projection_conflict_count": conflict_count,
        "mean_pre_projection_cosine": cosine_sum / max(step_count, 1),
        "projection_capped_count": capped_count,
        "mean_auxiliary_scale": applied_scale_sum / max(step_count, 1),
        "minimum_post_projection_dot": (
            minimum_post_dot if step_count else 0.0
        ),
    }


def _classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float]:
    prediction = logits[mask].argmax(dim=1).detach().cpu().numpy()
    truth = labels[mask].detach().cpu().numpy()
    return (
        float(f1_score(truth, prediction, average="macro", zero_division=0)),
        float(np.mean(prediction == truth)),
    )


def _certificate_subgroup_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    certified: torch.Tensor,
    correct: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    count = int(mask.sum().item())
    if count == 0:
        return {
            "node_count": 0,
            "local_macro_f1": float("nan"),
            "local_accuracy": float("nan"),
            "certified_ratio": float("nan"),
            "certified_accuracy": float("nan"),
        }
    local_f1, local_accuracy = _classification_metrics(logits, labels, mask)
    return {
        "node_count": count,
        "local_macro_f1": local_f1,
        "local_accuracy": local_accuracy,
        "certified_ratio": float((certified & mask).sum().item() / count),
        "certified_accuracy": float(
            (certified & correct & mask).sum().item() / count
        ),
    }


def _certificate_bound_ablation(
    model: PolynomialSpectralClassifier,
    logits: torch.Tensor,
    client: BoundaryFederatedClient,
    *,
    include_cone: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate old and tightened certificates on one fixed predictor.

    The first-order uncertainty basis always equals the row budget, so both
    recurrences can be reconstructed without accessing remote endpoints.
    This diagnostic never changes logits, checkpoints, or training state.
    """
    if model.has_agreement_shield:
        return {}
    row_budget = client.uncertainty_basis[1]
    tight_basis = retained_mass_uncertainty_basis(
        client.shift, row_budget, model.order_count
    )
    legacy_basis = uncertainty_basis(client.shift, row_budget, model.order_count)
    original_mode = model.certificate_margin_mode
    output: dict[str, torch.Tensor] = {}
    try:
        for recurrence, basis in (
            ("tight", tight_basis),
            ("legacy", legacy_basis),
        ):
            for margin_mode, short_name in (
                ("pairwise_difference", "pairwise"),
                ("classwise_sum", "classwise"),
            ):
                model.certificate_margin_mode = margin_mode
                output[f"{recurrence}_{short_name}"] = certificate_statistics(
                    model, logits, basis
                ).certified
    finally:
        model.certificate_margin_mode = original_mode
    if include_cone:
        if model.encoder_type not in {
            "linear_relu",
            "feature_only_mlp",
            "dual_feature_spectral",
            "dual_feature_identity_null_spectral",
        }:
            raise ValueError(
                "positive-cone audit requires a ReLU-normalized spectral encoder"
            )
        cone = cone_pairwise_certificate_statistics(
            model,
            client.features,
            client.shift,
            row_budget,
            logits,
            tight_basis,
            count_semantics="upper_bound",
            combine_symmetric=False,
        )
        output["cone_raw"] = cone.certified
        output["cone_hybrid"] = cone.certified | output["tight_pairwise"]
    return output


def _logit_audit_tolerance(
    local_logits: torch.Tensor,
    completion_logits: torch.Tensor,
) -> torch.Tensor:
    """Return a dtype-aware tolerance for the empirical logit-bound audit.

    The local and completion passes use different batch shapes.  Even a
    graph-invariant feature expert can therefore differ by a few float32 ULPs
    because GEMM and normalization reductions are associated differently.
    This tolerance is used only when auditing the analytic bound; it never
    changes logits, predictions, certificate radii, or certified nodes.
    """
    if local_logits.dtype != completion_logits.dtype:
        raise ValueError("logit audit tensors must use the same dtype")
    if not local_logits.dtype.is_floating_point:
        return torch.zeros_like(local_logits)
    scale = torch.maximum(local_logits.abs(), completion_logits.abs())
    epsilon = torch.finfo(local_logits.dtype).eps
    dtype_tolerance = 64.0 * epsilon * (1.0 + scale)
    historical_tolerance = 1e-6 + 1e-5 * scale
    return torch.maximum(dtype_tolerance, historical_tolerance)


def _completion_logits_with_aligned_nodewise_encoding(
    model: PolynomialSpectralClassifier,
    clients: Sequence[BoundaryFederatedClient],
    full_shift: torch.Tensor,
    local_logits: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a completion while reusing local-batch nodewise encodings.

    Node encoders and feature experts are graph invariant by construction.
    Re-evaluating them in one global batch can nevertheless change the last
    float32 bits because GEMM kernels use a different batch shape.  The
    analytic model defines one nodewise encoding followed by graph
    propagation, so the completion audit assembles those encodings from the
    same client batches and changes only the propagation operator.

    This removes a numerical nuisance without changing the mathematical
    predictor, certificate, or local logits.  Agreement-shielded models have
    graph-dependent utility branches and are handled by their dedicated
    evaluator instead.
    """
    if model.has_agreement_shield:
        raise ValueError("agreement-shielded completion requires fixed routing")
    if model.encoder_type == "feature_only_mlp":
        return local_logits.clone()

    node_count = int(local_logits.size(0))
    hidden_dim = int(model.classifier.in_features)
    bounded_hidden = torch.empty(
        (node_count, hidden_dim),
        device=local_logits.device,
        dtype=local_logits.dtype,
    )
    feature_logits = (
        torch.empty_like(local_logits)
        if model.feature_classifier is not None
        else None
    )
    for client in clients:
        nodes = client.global_nodes
        bounded_hidden[nodes] = model.bounded_encoding(client.features)
        if feature_logits is not None:
            feature_logits[nodes] = model.feature_classifier(
                model.feature_encoding(client.features)
            )
    spectral_logits = model.classifier(model.filter(bounded_hidden, full_shift))
    if feature_logits is None:
        return spectral_logits
    return feature_logits + model.spectral_fusion_scale * spectral_logits


def evaluate_external_teacher_state(
    state: TensorState,
    clients: Sequence[BoundaryFederatedClient],
    teacher_logits: torch.Tensor,
    *,
    global_features: torch.Tensor,
    global_labels: torch.Tensor,
    global_validation_mask: torch.Tensor,
    global_test_mask: torch.Tensor,
    full_shift: torch.Tensor,
    hidden_dim: int,
    num_classes: int,
    order_count: int,
    dropout: float,
    encoder_type: str,
    encoder_depth: int,
    encoder_residual_scale: float,
    device: torch.device,
    include_test: bool,
    certificate_margin_mode: str = "pairwise_difference",
    certificate_kind: str = "cone_hybrid",
    propagation_mode: str = "observed",
) -> dict:
    """Evaluate a strong utility teacher with a fixed certified safe shield.

    Ordinary predictions are exactly the external teacher's predictions.  A
    node is certified only if the endpoint-blind safe expert is certified and
    agrees with that teacher.  Completion diagnostics therefore perturb only
    the safe branch on shielded nodes; no guarantee is attributed to the
    utility teacher outside the shield.
    """
    model = PolynomialSpectralClassifier(
        int(global_features.size(1)),
        hidden_dim,
        num_classes,
        order_count,
        dropout,
        encoder_type=encoder_type,
        encoder_depth=encoder_depth,
        encoder_residual_scale=encoder_residual_scale,
        certificate_margin_mode=certificate_margin_mode,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    teacher_logits = teacher_logits.detach().to(device).float()
    node_count = int(global_features.size(0))
    if teacher_logits.shape != (node_count, int(num_classes)):
        raise ValueError("global external teacher logits are not node/class aligned")
    if propagation_mode not in {"observed", "identity"}:
        raise ValueError(f"unsupported propagation mode: {propagation_mode}")

    local_safe = torch.empty_like(teacher_logits)
    routed_logits = torch.empty_like(teacher_logits)
    class_bounds = torch.empty_like(teacher_logits)
    certified = torch.zeros(node_count, dtype=torch.bool, device=device)
    agreement = torch.zeros_like(certified)
    boundary_node = torch.zeros_like(certified)
    with torch.no_grad():
        for client in clients:
            nodes = client.global_nodes
            safe_shift = (
                client.shift
                if propagation_mode == "observed"
                else _identity_shift_like(client.shift)
            )
            safe_basis = (
                client.uncertainty_basis
                if propagation_mode == "observed"
                else _zero_uncertainty_basis_like(client.uncertainty_basis)
            )
            statistics = external_teacher_agreement_statistics(
                model,
                client.features,
                safe_shift,
                safe_basis,
                teacher_logits[nodes],
                certificate_kind=certificate_kind,
            )
            local_safe[nodes] = statistics.safe_logits
            routed_logits[nodes] = statistics.logits
            class_bounds[nodes] = statistics.certificate.class_logit_bounds
            certified[nodes] = statistics.shield
            agreement[nodes] = (
                statistics.safe_logits.argmax(dim=1)
                == statistics.utility_logits.argmax(dim=1)
            )
            boundary_node[nodes] = client.missing_degree > 0

        if encoder_type == "feature_only_mlp" or propagation_mode == "identity":
            completion_safe = local_safe.clone()
        else:
            hidden = torch.empty(
                (node_count, int(model.classifier.in_features)),
                dtype=local_safe.dtype,
                device=device,
            )
            for client in clients:
                hidden[client.global_nodes] = model.bounded_encoding(client.features)
            completion_safe = model.classifier(model.filter(hidden, full_shift))

    teacher_prediction = teacher_logits.argmax(dim=1)
    routed_prediction = routed_logits.argmax(dim=1)
    safe_prediction = local_safe.argmax(dim=1)
    completion_safe_prediction = completion_safe.argmax(dim=1)
    if not torch.equal(routed_prediction, teacher_prediction):
        raise RuntimeError("fixed shield failed to preserve teacher predictions")
    correct = teacher_prediction == global_labels
    safe_deviation = (completion_safe - local_safe).abs()
    tolerance = _logit_audit_tolerance(local_safe, completion_safe)

    output: dict = {
        "utility_prediction_preserved_for_all_nodes": True,
        "certificate_kind": certificate_kind,
        "propagation_mode": propagation_mode,
        "completion_audit_scope": "safe branch on fixed shield nodes only",
    }
    splits = [("validation", global_validation_mask)]
    if include_test:
        splits.append(("test", global_test_mask))
    for split_name, mask in splits:
        count = max(int(mask.sum().item()), 1)
        utility_f1, utility_accuracy = _classification_metrics(
            teacher_logits, global_labels, mask
        )
        routed_f1, routed_accuracy = _classification_metrics(
            routed_logits, global_labels, mask
        )
        safe_f1, safe_accuracy = _classification_metrics(
            local_safe, global_labels, mask
        )
        if utility_f1 != routed_f1 or utility_accuracy != routed_accuracy:
            raise RuntimeError("routed utility metrics differ from the teacher")

        active_classes = (certified & mask)[:, None].expand_as(safe_deviation)
        logit_violations = int(
            (
                active_classes
                & (safe_deviation > class_bounds + tolerance)
            ).sum().item()
        )
        certified_flips = int(
            (
                certified
                & mask
                & (safe_prediction != completion_safe_prediction)
            ).sum().item()
        )
        client_utility_f1: list[float] = []
        client_certified_ratio: list[float] = []
        client_certified_accuracy: list[float] = []
        for client in clients:
            client_mask = torch.zeros_like(mask)
            client_mask[client.global_nodes] = True
            active = mask & client_mask
            if not bool(active.any()):
                continue
            active_count = int(active.sum().item())
            active_f1, _ = _classification_metrics(
                teacher_logits, global_labels, active
            )
            client_utility_f1.append(active_f1)
            client_certified_ratio.append(
                float((certified & active).sum().item() / active_count)
            )
            client_certified_accuracy.append(
                float((certified & correct & active).sum().item() / active_count)
            )
        prefix = f"{split_name}_"
        output.update({
            prefix + "utility_macro_f1": utility_f1,
            prefix + "utility_accuracy": utility_accuracy,
            prefix + "routed_macro_f1": routed_f1,
            prefix + "routed_accuracy": routed_accuracy,
            prefix + "safe_macro_f1": safe_f1,
            prefix + "safe_accuracy": safe_accuracy,
            prefix + "safe_teacher_agreement_ratio": float(
                (agreement & mask).sum().item() / count
            ),
            prefix + "certified_ratio": float(
                (certified & mask).sum().item() / count
            ),
            prefix + "certified_accuracy": float(
                (certified & correct & mask).sum().item() / count
            ),
            prefix + "certified_completion_violations": certified_flips,
            prefix + "logit_bound_violations": logit_violations,
            prefix + "mean_shielded_safe_logit_change": float(
                safe_deviation[active_classes].mean().item()
                if bool(active_classes.any()) else 0.0
            ),
            prefix + "worst_client_utility_macro_f1": float(
                min(client_utility_f1)
            ),
            prefix + "worst_client_certified_ratio": float(
                min(client_certified_ratio)
            ),
            prefix + "worst_client_certified_accuracy": float(
                min(client_certified_accuracy)
            ),
            prefix + "boundary_certified_ratio": float(
                (certified & mask & boundary_node).sum().item()
                / max(int((mask & boundary_node).sum().item()), 1)
            ),
            prefix + "interior_certified_ratio": float(
                (certified & mask & ~boundary_node).sum().item()
                / max(int((mask & ~boundary_node).sum().item()), 1)
            ),
        })
    return output


def evaluate_federated_state(
    state: TensorState,
    clients: Sequence[BoundaryFederatedClient],
    *,
    global_features: torch.Tensor,
    global_labels: torch.Tensor,
    global_validation_mask: torch.Tensor,
    global_test_mask: torch.Tensor,
    full_shift: torch.Tensor,
    hidden_dim: int,
    num_classes: int,
    order_count: int,
    dropout: float,
    encoder_type: str = "linear_relu",
    encoder_depth: int = 1,
    encoder_residual_scale: float = 0.5,
    utility_diffusion_hops: int = 1,
    utility_diffusion_scale: float = 0.25,
    device: torch.device,
    include_test: bool,
    spectral_fusion_scale: float | None = None,
    certificate_margin_mode: str = "classwise_sum",
    include_cone_certificate_audit: bool = False,
) -> dict:
    model = PolynomialSpectralClassifier(
        int(global_features.size(1)),
        hidden_dim,
        num_classes,
        order_count,
        dropout,
        encoder_type,
        encoder_depth,
        encoder_residual_scale,
        utility_diffusion_hops,
        utility_diffusion_scale,
        spectral_fusion_scale,
        certificate_margin_mode,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    node_count = int(global_features.size(0))
    local_logits = torch.empty((node_count, num_classes), device=device)
    class_bounds = torch.empty_like(local_logits)
    certified = torch.zeros(node_count, dtype=torch.bool, device=device)
    boundary_node = torch.zeros(node_count, dtype=torch.bool, device=device)
    bound_ablation = {
        name: torch.zeros(node_count, dtype=torch.bool, device=device)
        for name in (() if model.has_agreement_shield else (
            "tight_pairwise",
            "tight_classwise",
            "legacy_pairwise",
            "legacy_classwise",
            *(("cone_raw", "cone_hybrid") if include_cone_certificate_audit else ()),
        ))
    }
    with torch.no_grad():
        for client in clients:
            prediction_stats = prediction_certificate_statistics(
                model, client.features, client.shift, client.uncertainty_basis
            )
            logits = prediction_stats.logits
            stats = prediction_stats.certificate
            local_logits[client.global_nodes] = logits
            class_bounds[client.global_nodes] = stats.class_logit_bounds
            certified[client.global_nodes] = stats.certified
            boundary_node[client.global_nodes] = client.missing_degree > 0
            for name, values in _certificate_bound_ablation(
                model,
                logits,
                client,
                include_cone=include_cone_certificate_audit,
            ).items():
                bound_ablation[name][client.global_nodes] = values
        if model.has_agreement_shield:
            full_safe_logits = model.spectral_logits(global_features, full_shift)
            full_utility_logits = model.utility_logits(global_features, full_shift)
            # ``certified`` is the fixed local-view routing mask.  It must not
            # be recomputed after revealing a completion.
            full_logits = torch.where(
                certified[:, None], full_safe_logits, full_utility_logits
            )
        else:
            full_logits = _completion_logits_with_aligned_nodewise_encoding(
                model, clients, full_shift, local_logits
            )

    output: dict = {}
    splits = [("validation", global_validation_mask)]
    if include_test:
        splits.append(("test", global_test_mask))
    class_deviation = (full_logits - local_logits).abs()
    numerical_tolerance = _logit_audit_tolerance(local_logits, full_logits)
    prediction = local_logits.argmax(dim=1)
    full_prediction = full_logits.argmax(dim=1)
    correct = prediction == global_labels
    for split_name, mask in splits:
        local_f1, local_accuracy = _classification_metrics(
            local_logits, global_labels, mask
        )
        full_f1, full_accuracy = _classification_metrics(
            full_logits, global_labels, mask
        )
        count = max(int(mask.sum().item()), 1)
        split_classes = mask[:, None].expand_as(class_deviation)
        bound_violations = int(
            (
                split_classes
                & (class_deviation > class_bounds + numerical_tolerance)
            ).sum().item()
        )
        certified_flips = int(
            (certified & (prediction != full_prediction) & mask).sum().item()
        )
        client_f1 = []
        client_accuracy = []
        client_certified_ratio = []
        client_certified_accuracy = []
        for client in clients:
            client_global_mask = torch.zeros_like(mask)
            client_global_mask[client.global_nodes] = True
            active = mask & client_global_mask
            if bool(active.any()):
                active_count = int(active.sum().item())
                active_f1, active_accuracy = _classification_metrics(
                    local_logits, global_labels, active
                )
                client_f1.append(active_f1)
                client_accuracy.append(active_accuracy)
                client_certified_ratio.append(
                    float((certified & active).sum().item() / active_count)
                )
                client_certified_accuracy.append(
                    float((certified & correct & active).sum().item() / active_count)
                )
        certified_accuracy = float(
            (certified & correct & mask).sum().item() / count
        )
        prefix = f"{split_name}_"
        subgroup_metrics = {}
        for subgroup, subgroup_mask in (
            ("boundary", mask & boundary_node),
            ("interior", mask & ~boundary_node),
        ):
            for name, value in _certificate_subgroup_metrics(
                local_logits,
                global_labels,
                certified,
                correct,
                subgroup_mask,
            ).items():
                subgroup_metrics[prefix + subgroup + "_" + name] = value
        ablation_metrics = {}
        for name, values in bound_ablation.items():
            ablation_metrics[prefix + "bound_" + name + "_certified_ratio"] = float(
                (values & mask).sum().item() / count
            )
            ablation_metrics[
                prefix + "bound_" + name + "_certified_accuracy"
            ] = float((values & correct & mask).sum().item() / count)
            ablation_metrics[
                prefix + "bound_" + name + "_certified_completion_violations"
            ] = int(
                (values & (prediction != full_prediction) & mask).sum().item()
            )
        output.update(
            {
                prefix + "local_macro_f1": local_f1,
                prefix + "local_accuracy": local_accuracy,
                prefix + "true_completion_macro_f1": full_f1,
                prefix + "true_completion_accuracy": full_accuracy,
                prefix + "observed_worst_macro_f1": min(local_f1, full_f1),
                prefix + "certified_ratio": float(
                    (certified & mask).sum().item() / count
                ),
                prefix + "certified_accuracy": certified_accuracy,
                prefix + "certified_accuracy_over_clean_accuracy": float(
                    certified_accuracy / local_accuracy
                    if local_accuracy > 0.0 else 0.0
                ),
                prefix + "completion_flip_fraction": float(
                    ((prediction != full_prediction) & mask).sum().item() / count
                ),
                prefix + "mean_absolute_completion_logit_change": float(
                    class_deviation[split_classes].mean().item()
                ),
                prefix + "maximum_absolute_completion_logit_change": float(
                    class_deviation[split_classes].max().item()
                ),
                prefix + "certified_completion_violations": certified_flips,
                prefix + "logit_bound_violations": bound_violations,
                prefix + "worst_client_local_macro_f1": float(min(client_f1)),
                prefix + "worst_client_local_accuracy": float(
                    min(client_accuracy)
                ),
                prefix + "worst_client_certified_ratio": float(
                    min(client_certified_ratio)
                ),
                prefix + "worst_client_certified_accuracy": float(
                    min(client_certified_accuracy)
                ),
                **subgroup_metrics,
                **ablation_metrics,
            }
        )
    return output


def evaluate_personalized_states(
    states: Sequence[TensorState],
    clients: Sequence[BoundaryFederatedClient],
    *,
    global_features: torch.Tensor,
    global_labels: torch.Tensor,
    global_validation_mask: torch.Tensor,
    global_test_mask: torch.Tensor,
    full_shift: torch.Tensor,
    hidden_dim: int,
    num_classes: int,
    order_count: int,
    dropout: float,
    encoder_type: str = "linear_relu",
    encoder_depth: int = 1,
    encoder_residual_scale: float = 0.5,
    utility_diffusion_hops: int = 1,
    utility_diffusion_scale: float = 0.25,
    device: torch.device,
    include_test: bool,
    spectral_fusion_scale: float | None = None,
    certificate_margin_mode: str = "classwise_sum",
    include_cone_certificate_audit: bool = False,
) -> dict:
    """Evaluate one independently selected model per client.

    Completion diagnostics use each client's own model on the full shift and
    retain only predictions for nodes owned by that client.  This avoids
    inventing a global model for the Local baseline.
    """
    if len(states) != len(clients) or not states:
        raise ValueError("personalized states and clients must be aligned")
    node_count = int(global_features.size(0))
    local_logits = torch.empty((node_count, num_classes), device=device)
    full_logits = torch.empty_like(local_logits)
    class_bounds = torch.empty_like(local_logits)
    certified = torch.zeros(node_count, dtype=torch.bool, device=device)
    boundary_node = torch.zeros(node_count, dtype=torch.bool, device=device)
    bound_ablation = {
        name: torch.zeros(node_count, dtype=torch.bool, device=device)
        for name in (
            ()
            if encoder_type in {
                "shielded_sage_spectral",
                "shielded_diffusion_spectral",
            }
            else (
            "tight_pairwise",
            "tight_classwise",
            "legacy_pairwise",
            "legacy_classwise",
            *(("cone_raw", "cone_hybrid") if include_cone_certificate_audit else ()),
            )
        )
    }
    with torch.no_grad():
        for state, client in zip(states, clients):
            model = PolynomialSpectralClassifier(
                int(global_features.size(1)),
                hidden_dim,
                num_classes,
                order_count,
                dropout,
                encoder_type,
                encoder_depth,
                encoder_residual_scale,
                utility_diffusion_hops,
                utility_diffusion_scale,
                spectral_fusion_scale,
                certificate_margin_mode,
            ).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            prediction_stats = prediction_certificate_statistics(
                model, client.features, client.shift, client.uncertainty_basis
            )
            client_logits = prediction_stats.logits
            stats = prediction_stats.certificate
            nodes = client.global_nodes
            local_logits[nodes] = client_logits
            class_bounds[nodes] = stats.class_logit_bounds
            certified[nodes] = stats.certified
            boundary_node[nodes] = client.missing_degree > 0
            for name, values in _certificate_bound_ablation(
                model,
                client_logits,
                client,
                include_cone=include_cone_certificate_audit,
            ).items():
                bound_ablation[name][nodes] = values
            if model.has_agreement_shield:
                full_safe_logits = model.spectral_logits(global_features, full_shift)
                full_utility_logits = model.utility_logits(global_features, full_shift)
                full_logits[nodes] = torch.where(
                    stats.certified[:, None],
                    full_safe_logits[nodes],
                    full_utility_logits[nodes],
                )
            else:
                full_logits[nodes] = model(global_features, full_shift)[nodes]

    output: dict = {}
    splits = [("validation", global_validation_mask)]
    if include_test:
        splits.append(("test", global_test_mask))
    class_deviation = (full_logits - local_logits).abs()
    numerical_tolerance = _logit_audit_tolerance(local_logits, full_logits)
    prediction = local_logits.argmax(dim=1)
    full_prediction = full_logits.argmax(dim=1)
    correct = prediction == global_labels
    for split_name, mask in splits:
        local_f1, local_accuracy = _classification_metrics(
            local_logits, global_labels, mask
        )
        full_f1, full_accuracy = _classification_metrics(
            full_logits, global_labels, mask
        )
        count = max(int(mask.sum().item()), 1)
        split_classes = mask[:, None].expand_as(class_deviation)
        bound_violations = int(
            (
                split_classes
                & (class_deviation > class_bounds + numerical_tolerance)
            ).sum().item()
        )
        certified_flips = int(
            (certified & (prediction != full_prediction) & mask).sum().item()
        )
        client_f1 = []
        client_accuracy = []
        client_certified_ratio = []
        client_certified_accuracy = []
        for client in clients:
            client_global_mask = torch.zeros_like(mask)
            client_global_mask[client.global_nodes] = True
            active = mask & client_global_mask
            if bool(active.any()):
                active_count = int(active.sum().item())
                active_f1, active_accuracy = _classification_metrics(
                    local_logits, global_labels, active
                )
                client_f1.append(active_f1)
                client_accuracy.append(active_accuracy)
                client_certified_ratio.append(
                    float((certified & active).sum().item() / active_count)
                )
                client_certified_accuracy.append(
                    float((certified & correct & active).sum().item() / active_count)
                )
        certified_accuracy = float(
            (certified & correct & mask).sum().item() / count
        )
        prefix = f"{split_name}_"
        subgroup_metrics = {}
        for subgroup, subgroup_mask in (
            ("boundary", mask & boundary_node),
            ("interior", mask & ~boundary_node),
        ):
            for name, value in _certificate_subgroup_metrics(
                local_logits,
                global_labels,
                certified,
                correct,
                subgroup_mask,
            ).items():
                subgroup_metrics[prefix + subgroup + "_" + name] = value
        ablation_metrics = {}
        for name, values in bound_ablation.items():
            ablation_metrics[prefix + "bound_" + name + "_certified_ratio"] = float(
                (values & mask).sum().item() / count
            )
            ablation_metrics[
                prefix + "bound_" + name + "_certified_accuracy"
            ] = float((values & correct & mask).sum().item() / count)
            ablation_metrics[
                prefix + "bound_" + name + "_certified_completion_violations"
            ] = int(
                (values & (prediction != full_prediction) & mask).sum().item()
            )
        output.update({
            prefix + "local_macro_f1": local_f1,
            prefix + "local_accuracy": local_accuracy,
            prefix + "true_completion_macro_f1": full_f1,
            prefix + "true_completion_accuracy": full_accuracy,
            prefix + "observed_worst_macro_f1": min(local_f1, full_f1),
            prefix + "certified_ratio": float(
                (certified & mask).sum().item() / count
            ),
            prefix + "certified_accuracy": certified_accuracy,
            prefix + "certified_accuracy_over_clean_accuracy": float(
                certified_accuracy / local_accuracy
                if local_accuracy > 0.0 else 0.0
            ),
            prefix + "completion_flip_fraction": float(
                ((prediction != full_prediction) & mask).sum().item() / count
            ),
            prefix + "mean_absolute_completion_logit_change": float(
                class_deviation[split_classes].mean().item()
            ),
            prefix + "maximum_absolute_completion_logit_change": float(
                class_deviation[split_classes].max().item()
            ),
            prefix + "certified_completion_violations": certified_flips,
            prefix + "logit_bound_violations": bound_violations,
            prefix + "worst_client_local_macro_f1": float(min(client_f1)),
            prefix + "worst_client_local_accuracy": float(min(client_accuracy)),
            prefix + "worst_client_certified_ratio": float(
                min(client_certified_ratio)
            ),
            prefix + "worst_client_certified_accuracy": float(
                min(client_certified_accuracy)
            ),
            **subgroup_metrics,
            **ablation_metrics,
        })
    return output
