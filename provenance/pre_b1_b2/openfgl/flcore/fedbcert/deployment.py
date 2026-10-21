"""Small, version-bound deployment demonstration, not a new verifier theorem.

This adapter deliberately supports only the stock ReLU polynomial predictors
on CPU. Ledger provenance is an external assumption, never inferred locally.
The existing certificate backend uses floating-point arithmetic; a positive
margin guard is NOT an outward-rounded machine-checkable numerical proof.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .certificate import (
    PolynomialSpectralClassifier,
    boundary_uncertainty,
    canonical_undirected,
    certificate_statistics,
    cone_pairwise_certificate_statistics,
    retained_mass_uncertainty_basis,
    row_normalized_shift,
)


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.dtype, array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class BoundaryLedgerReceipt:
    """Receipt supplied by a separately audited ledger authority.

    ``upper_bound_attested`` is a caller assertion, NOT a local validation of
    actual missing degrees. Counts always mean upper bounds, including when
    an authority happens to supply exact counts. Node order is snapshot order.
    """

    bounds: torch.Tensor
    ledger_version: str
    source: str
    view_sha256: str
    preprocessing_version: str
    upper_bound_attested: bool


class FrozenLocalDeployment:
    """Own a copied nominal view and return predictions plus separate states.

    Unsupported model/inference inputs raise ValueError before certification;
    they are not silently repaired. Once inference is available, missing or
    invalid ledger evidence returns Unavailable with the nominal output intact.
    No labels, remote features, remote node IDs, or completion graph are inputs.
    """

    def __init__(
        self,
        model: PolynomialSpectralClassifier,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        view_version: str,
        preprocessing_version: str,
        node_ids: Sequence[str] | None = None,
    ) -> None:
        self._validate_model(model)
        if not _nonempty(view_version) or not _nonempty(preprocessing_version):
            raise ValueError("view and preprocessing versions must be nonempty")
        if (not isinstance(features, torch.Tensor) or features.layout != torch.strided
                or features.ndim != 2 or features.size(0) < 1
                or features.device.type != "cpu"
                or features.dtype != model.theta.dtype
                or features.size(1) != model.encoder.in_features
                or not bool(torch.isfinite(features).all())):
            raise ValueError("features must be finite, CPU, dense and model-aligned")
        size = features.size(0)
        if (not isinstance(edge_index, torch.Tensor)
                or edge_index.layout != torch.strided
                or edge_index.ndim != 2 or edge_index.size(0) != 2
                or edge_index.dtype != torch.long or edge_index.device.type != "cpu"
                or (edge_index.numel() and bool(
                    ((edge_index < 0) | (edge_index >= size)).any()))):
            raise ValueError("local edges must be CPU int64 indices in the local node range")
        ids = tuple(str(index) for index in range(size)) if node_ids is None else tuple(node_ids)
        if len(ids) != size or any(not _nonempty(item) for item in ids) or len(set(ids)) != size:
            raise ValueError("node IDs must be unique nonempty strings in feature-row order")
        self._model = deepcopy(model)
        self._features = features.detach().clone()
        # Explicit interface semantics: simple undirected unit-weight graph,
        # remove supplied self-loops/duplicates, then add exactly one self-loop.
        self._edges = canonical_undirected(edge_index, size)
        self._shift, self._degree = row_normalized_shift(
            self._edges, size, device=torch.device("cpu"), dtype=features.dtype
        )
        self._node_ids = ids
        self.view_version = view_version
        self.preprocessing_version = preprocessing_version
        self._core_sha256 = hashlib.sha256(
            Path(__file__).with_name("certificate.py").read_bytes()
        ).hexdigest()
        self._model_sha256 = self._model_hash()
        self._view_sha256 = self._view_hash()
        self._shift_sha256 = _tensor_hash(self._shift.indices()) + _tensor_hash(self._shift.values())
        self._degree_sha256 = _tensor_hash(self._degree)
        with torch.no_grad():
            self._logits = self._model(self._features, self._shift).detach().clone()
        if not bool(torch.isfinite(self._logits).all()):
            raise ValueError("nominal logits must be finite")
        self._logits_sha256 = _tensor_hash(self._logits)

    @staticmethod
    def _validate_model(model: PolynomialSpectralClassifier) -> None:
        if type(model) is not PolynomialSpectralClassifier:
            raise ValueError("only the stock PolynomialSpectralClassifier is supported")
        if model.encoder_type not in {"linear_relu", "dual_feature_spectral"}:
            raise ValueError("unsupported predictor: this adapter does not certify teachers or shields")
        if any(module.training for module in model.modules()):
            raise ValueError("freeze an eval-mode predictor; the adapter never changes its mode")
        if model.certificate_margin_mode != "pairwise_difference":
            raise ValueError("this adapter requires the pairwise_difference certificate configuration")
        if (model.encoder_type == "linear_relu" and model.spectral_fusion_scale != 1.0):
            raise ValueError("linear_relu forward requires spectral_fusion_scale=1 in this adapter")
        if (not math.isfinite(model.spectral_fusion_scale) or model.spectral_fusion_scale < 0
                or model.order_count < 2 or model.theta.shape != (model.order_count,)
                or model.classifier.out_features < 2
                or (model.encoder_type == "linear_relu" and model.encoder_depth != 1)
                or (model.encoder_type == "dual_feature_spectral" and model.encoder_depth < 2)):
            raise ValueError("invalid polynomial or classification configuration")
        if model.theta.device.type != "cpu" or model.theta.dtype not in {torch.float32, torch.float64}:
            raise ValueError("this demonstration supports CPU float32/float64 only")
        for module in model.modules():
            if type(module) not in {PolynomialSpectralClassifier, nn.Linear, nn.LayerNorm, nn.ModuleList}:
                raise ValueError("custom modules are outside this adapter's contract")
            if module._forward_hooks or module._forward_pre_hooks:
                raise ValueError("forward hooks are outside this adapter's contract")
            if any(callable(value) for value in vars(module).values()):
                raise ValueError("instance method overrides are outside this adapter's contract")
        if len(model.encoder_blocks) or len(model.encoder_norms):
            raise ValueError("the supported spectral encoding is ReLU followed by row normalization")
        if (model.encoder_type == "linear_relu" and model.feature_classifier is not None
                or model.encoder_type == "dual_feature_spectral" and (
                    model.feature_classifier is None or model.feature_encoder is None
                    or len(model.feature_blocks) != model.encoder_depth - 1
                    or len(model.feature_norms) != model.encoder_depth - 1)):
            raise ValueError("feature-branch architecture does not match the declared predictor")
        for value in model.state_dict().values():
            if (value.device.type != "cpu" or value.dtype != model.theta.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("model state must be finite and have one CPU floating-point dtype")

    def _model_hash(self) -> str:
        fields = (
            "order_count", "dropout", "encoder_type", "encoder_depth",
            "encoder_residual_scale", "utility_diffusion_hops", "utility_diffusion_scale",
            "spectral_fusion_scale", "certificate_margin_mode",
        )
        return _json_hash({
            "core_sha256": self._core_sha256,
            "configuration": {key: getattr(self._model, key) for key in fields},
            "modules": {key: {"type": type(value).__qualname__, "configuration": value.extra_repr(),
                              "training": value.training}
                        for key, value in self._model.named_modules()},
            "state": {key: _tensor_hash(value) for key, value in self._model.state_dict().items()},
        })

    def _view_hash(self) -> str:
        return _json_hash({
            "view_version": self.view_version,
            "preprocessing_version": self.preprocessing_version,
            "node_ids": self._node_ids,
            "features": _tensor_hash(self._features),
            "canonical_edges": _tensor_hash(self._edges),
            "graph_semantics": "simple_undirected_unit_weight_with_one_self_loop",
        })

    @property
    def view_sha256(self) -> str:
        return self._view_sha256

    @property
    def nominal_logits(self) -> torch.Tensor:
        return self._logits.clone()

    def _check_snapshot(self) -> None:
        self._validate_model(self._model)
        shift_hash = _tensor_hash(self._shift.indices()) + _tensor_hash(self._shift.values())
        if (self._model_hash() != self._model_sha256 or self._view_hash() != self._view_sha256
                or _tensor_hash(self._logits) != self._logits_sha256
                or shift_hash != self._shift_sha256 or _tensor_hash(self._degree) != self._degree_sha256):
            raise RuntimeError("frozen snapshot changed; create a new snapshot and ledger receipt")

    def _ledger_problem(self, ledger: BoundaryLedgerReceipt | None) -> str | None:
        if ledger is None:
            return "missing_ledger_no_automatic_fallback"
        if type(ledger) is not BoundaryLedgerReceipt:
            return "unsupported_ledger_receipt"
        if not _nonempty(ledger.ledger_version) or not _nonempty(ledger.source):
            return "missing_ledger_version_or_source"
        if ledger.upper_bound_attested is not True:
            return "upper_bound_not_externally_attested"
        if ledger.view_sha256 != self._view_sha256:
            return "ledger_view_mismatch"
        if ledger.preprocessing_version != self.preprocessing_version:
            return "ledger_preprocessing_mismatch"
        bounds = ledger.bounds
        if (not isinstance(bounds, torch.Tensor) or bounds.layout != torch.strided
                or bounds.device.type != "cpu" or bounds.dtype not in {
                    torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
                } or bounds.ndim != 1 or bounds.numel() != len(self._node_ids)
                or bool((bounds < 0).any())):
            return "ledger_bounds_must_be_node_aligned_nonnegative_integers"
        return None

    def certify(
        self,
        ledger: BoundaryLedgerReceipt | None = None,
        *,
        mode: str = "bounded_ledger",
        margin_guard: float = 1e-6,
    ) -> dict:
        """Return JSON-ready records; Unavailable never becomes an abstention.

        Universal mode must be selected explicitly and takes no ledger. It uses
        delta=2, m=0 (all finite remote counts covered conservatively). Both
        modes retain fixed local inputs and the bounded ReLU representation.
        """
        if mode not in {"bounded_ledger", "universal"}:
            raise ValueError("select bounded_ledger or universal explicitly")
        if not isinstance(margin_guard, (int, float)) or not math.isfinite(margin_guard) or margin_guard <= 0:
            raise ValueError("margin_guard must be finite and strictly positive")
        if mode == "universal" and ledger is not None:
            raise ValueError("universal mode takes no ledger; do not silently discard ledger evidence")
        self._check_snapshot()
        problem = self._ledger_problem(ledger) if mode == "bounded_ledger" else None
        ledger_metadata = None
        if problem is None and ledger is not None:
            ledger_metadata = {
                "ledger_version": ledger.ledger_version,
                "source": ledger.source,
                "view_sha256": ledger.view_sha256,
                "preprocessing_version": ledger.preprocessing_version,
                "upper_bound_attested": True,
                "bounds_sha256": _tensor_hash(ledger.bounds),
                "count_semantics": "upper_bound",
            }
            ledger_metadata["receipt_sha256"] = _json_hash(ledger_metadata)
        symmetric = cone = None
        if problem is None:
            with torch.no_grad():
                budget = (
                    torch.full_like(self._degree, 2.0) if mode == "universal"
                    else boundary_uncertainty(self._degree, ledger.bounds.to(self._degree.dtype))
                )
                basis = retained_mass_uncertainty_basis(self._shift, budget, self._model.order_count)
                symmetric = certificate_statistics(self._model, self._logits, basis)
                cone = cone_pairwise_certificate_statistics(
                    self._model, self._features, self._shift, budget, self._logits, basis,
                    count_semantics="upper_bound", combine_symmetric=False,
                )
            if not bool(torch.isfinite(symmetric.worst_slack).all() & torch.isfinite(cone.worst_slack).all()):
                problem = "nonfinite_certificate_arithmetic"
        self._check_snapshot()
        records = []
        for index, node_id in enumerate(self._node_ids):
            record = {
                "node_id": node_id,
                "nominal_class": int(self._logits[index].argmax().item()),
                "state": "Unavailable",
                "reason": problem,
                "method": None,
                "minimum_gap_lower_bound": None,
                "symmetric_gap_lower_bound": None,
                "cone_gap_lower_bound": None,
            }
            if problem is None:
                sym_gap = float(symmetric.worst_slack[index].item())
                cone_gap = float(cone.worst_slack[index].item())
                gap = max(sym_gap, cone_gap)
                accepted = gap > margin_guard
                record.update({
                    "state": "Certified" if accepted else "Deferred",
                    "reason": "positive_guarded_completion_margin" if accepted else "sufficient_bound_inconclusive",
                    "method": "relu_cone_pairwise" if cone_gap > sym_gap else "retained_mass_pairwise",
                    "minimum_gap_lower_bound": gap,
                    "symmetric_gap_lower_bound": sym_gap,
                    "cone_gap_lower_bound": cone_gap,
                })
            records.append(record)
        return {
            "schema": "fedbcert_deployment_demo_v1",
            "mode": mode,
            "certificate_object": "same_frozen_polynomial_predictor_class_invariance",
            "snapshot": {
                "model_sha256": self._model_sha256,
                "core_sha256": self._core_sha256,
                "view_sha256": self._view_sha256,
                "view_version": self.view_version,
                "preprocessing_version": self.preprocessing_version,
                "nominal_logits_sha256": self._logits_sha256,
                "order_count": self._model.order_count,
                "maximum_power": self._model.order_count - 1,
                "hidden_domain": "nonnegative_l2_unit_ball",
                "dtype": str(self._features.dtype),
                "device": "cpu",
                "torch_version": torch.__version__,
                "graph_semantics": "simple_undirected_unit_weight_with_one_self_loop",
            },
            "ledger": ledger_metadata,
            "margin_guard": float(margin_guard),
            "numerical_assurance": "floating_point_backend_with_guard_not_outward_rounded_proof",
            "conditions": [
                "model_local_graph_features_preprocessing_and_node_order_remain_fixed",
                "all_remote_encoder_outputs_belong_to_the_declared_hidden_domain",
                "bounded_ledger_requires_true_counts_not_exceed_attested_bounds",
            ],
            "not_guaranteed": [
                "prediction_correctness", "logit_confidence_or_ranking_invariance",
                "ledger_truth_or_completeness", "privacy_or_poisoning_resistance",
                "safe_business_actions", "arbitrary_GNN_or_teacher_invariance",
            ],
            "nominal_logits": self._logits.tolist(),
            "records": records,
        }
