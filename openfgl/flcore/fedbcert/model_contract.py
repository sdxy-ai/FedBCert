"""Shared stock-model checks for the CPU deployment and replay adapters.

This is a supported-interface guard in a trusted Python/PyTorch runtime, not
protection against arbitrary hostile code in the same process or a numerical
soundness proof. No forward operation, certificate bound or threshold changes.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import json
import math
from pathlib import Path
from types import FunctionType

import torch
from torch import nn
from torch.nn.modules import module as module_runtime

from .certificate import PolynomialSpectralClassifier


_TYPES = (PolynomialSpectralClassifier, nn.Linear, nn.LayerNorm, nn.ModuleList)
_FIELDS = (
    "order_count", "dropout", "encoder_type", "encoder_depth",
    "encoder_residual_scale", "utility_diffusion_hops", "utility_diffusion_scale",
    "spectral_fusion_scale", "certificate_margin_mode",
)
# Save identities at trusted import, before accepting a caller-owned model.
# Class monkey-patching after that point must not bypass exact-type checks.
_CLASS_NAMESPACES = tuple((cls, dict(vars(cls))) for cls in (*_TYPES, nn.Module))
_CLASS_BASES = tuple((cls, cls.__bases__, cls.__mro__) for cls in (*_TYPES, nn.Module))
_FUNCTIONS = []
for _, namespace in _CLASS_NAMESPACES:
    for value in namespace.values():
        functions = (value,)
        if isinstance(value, (staticmethod, classmethod)):
            functions = (value.__func__,)
        elif isinstance(value, property):
            functions = (value.fget, value.fset, value.fdel)
        for function in functions:
            if isinstance(function, FunctionType):
                _FUNCTIONS.append((function, function.__code__, function.__defaults__,
                                   None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)))


def _plain_mapping(value):
    return type(value) in {dict, OrderedDict}


def _modules(model):
    """Walk raw registered children only after checking each stock type."""
    def walk(name, item, ancestors):
        if type(item) not in _TYPES or id(item) in ancestors:
            raise ValueError("custom or cyclic modules are outside the stock contract")
        if any(not _plain_mapping(vars(item).get(key)) for key in ("_modules", "_parameters", "_buffers")):
            raise ValueError("custom module/state registries are unsupported")
        if any(callable(value) for value in vars(item).values()):
            raise ValueError("instance method overrides or compiled calls are unsupported")
        if item.training is not False:
            raise ValueError("stock certification requires model.eval() for every module")
        for key, value in vars(item).items():
            if "hooks" in key and (not _plain_mapping(value) or value):
                raise ValueError("module hooks are outside the stock contract")
        yield name, item
        for key, child in item._modules.items():
            if child is not None:
                yield from walk(f"{name}.{key}" if name else key, child, ancestors | {id(item)})
    yield from walk("", model, set())


def _linear(item, input_size=None, output_size=None):
    if type(item) is not nn.Linear:
        raise ValueError("expected a stock Linear module")
    if (type(item.in_features) is not int or type(item.out_features) is not int
            or item.in_features < 1 or item.out_features < 1
            or tuple(item.weight.shape) != (item.out_features, item.in_features)
            or (item.bias is not None and tuple(item.bias.shape) != (item.out_features,))
            or (input_size is not None and item.in_features != input_size)
            or (output_size is not None and item.out_features != output_size)):
        raise ValueError("Linear configuration and parameter shapes disagree")


def validate_stock_model(model, *, require_pairwise=False):
    """Fail closed for the two existing CPU stock architectures only."""
    for cls, trusted in _CLASS_NAMESPACES:
        current = vars(cls)
        if set(current) != set(trusted) or any(current[key] is not value for key, value in trusted.items()):
            raise ValueError("stock class methods changed after trusted import")
    for cls, bases, mro in _CLASS_BASES:
        if cls.__bases__ != bases or cls.__mro__ != mro:
            raise ValueError("stock class inheritance changed after trusted import")
    for function, code, defaults, kwdefaults in _FUNCTIONS:
        if (function.__code__ is not code or function.__defaults__ is not defaults
                or function.__kwdefaults__ != kwdefaults):
            raise ValueError("stock function implementation changed after trusted import")
    for key, value in vars(module_runtime).items():
        if key.startswith("_global_") and "hooks" in key and (not _plain_mapping(value) or value):
            raise ValueError("global module hooks are outside the stock contract")
    if type(model) is not PolynomialSpectralClassifier:
        raise ValueError("only the stock PolynomialSpectralClassifier is supported")
    modules = tuple(_modules(model))
    for key in ("encoder_blocks", "encoder_norms", "feature_blocks", "feature_norms"):
        if type(getattr(model, key)) is not nn.ModuleList:
            raise ValueError("model branch containers must be stock ModuleList instances")
    if model.encoder_type not in {"linear_relu", "dual_feature_spectral"}:
        raise ValueError("unsupported predictor: teachers and shields are outside this adapter")
    if model.certificate_margin_mode not in {"classwise_sum", "pairwise_difference"}:
        raise ValueError("unsupported certificate margin mode")
    if require_pairwise and model.certificate_margin_mode != "pairwise_difference":
        raise ValueError("this adapter requires pairwise_difference")
    if (type(model.order_count) is not int or model.order_count < 2
            or model.theta.shape != (model.order_count,)
            or type(model.encoder_depth) is not int
            or (model.encoder_type == "linear_relu" and model.encoder_depth != 1)
            or (model.encoder_type == "dual_feature_spectral" and model.encoder_depth < 2)):
        raise ValueError("invalid polynomial or encoder configuration")
    for key in ("dropout", "encoder_residual_scale", "utility_diffusion_scale", "spectral_fusion_scale"):
        value = getattr(model, key)
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("model scalar configuration must be finite")
    if (not 0.0 <= model.dropout <= 1.0 or model.encoder_residual_scale <= 0
            or model.utility_diffusion_scale <= 0
            or type(model.utility_diffusion_hops) is not int or model.utility_diffusion_hops < 1):
        raise ValueError("invalid model scalar configuration")
    model._validate_certificate_fusion_contract()
    if model.theta.device.type != "cpu" or model.theta.dtype not in {torch.float32, torch.float64}:
        raise ValueError("this adapter supports CPU float32/float64 only")
    for _, item in modules:
        if type(item) is nn.Linear:
            _linear(item)
        elif type(item) is nn.LayerNorm:
            shape = item.normalized_shape
            if (type(shape) is not tuple or len(shape) != 1
                    or type(shape[0]) is not int or shape[0] < 1
                    or type(item.eps) not in {float, int} or not math.isfinite(item.eps) or item.eps < 0
                    or type(item.elementwise_affine) is not bool
                    or (item.elementwise_affine and item.weight is None)
                    or (not item.elementwise_affine and (item.weight is not None or item.bias is not None))
                    or (item.weight is not None and tuple(item.weight.shape) != shape)
                    or (item.bias is not None and tuple(item.bias.shape) != shape)):
                raise ValueError("LayerNorm configuration and parameter shapes disagree")
        for value in (*item._parameters.values(), *item._buffers.values()):
            if value is not None and (
                type(value) not in {torch.Tensor, nn.Parameter} or value.layout != torch.strided
                or value.device.type != "cpu" or value.dtype != model.theta.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("model state must be finite stock CPU tensors of one dtype")
    _linear(model.encoder)
    hidden = model.encoder.out_features
    _linear(model.classifier, hidden)
    if model.classifier.out_features < 2 or len(model.encoder_blocks) or len(model.encoder_norms):
        raise ValueError("unsupported classifier or spectral encoder structure")
    if model.encoder_type == "linear_relu":
        if (model.feature_classifier is not None or model.feature_encoder is not None
                or len(model.feature_blocks) or len(model.feature_norms)):
            raise ValueError("feature branch does not match linear_relu")
    else:
        _linear(model.feature_encoder, model.encoder.in_features, hidden)
        _linear(model.feature_classifier, hidden, model.classifier.out_features)
        if len(model.feature_blocks) != model.encoder_depth - 1 or len(model.feature_norms) != model.encoder_depth - 1:
            raise ValueError("feature-branch depth does not match its configuration")
        for layer, norm in zip(model.feature_blocks, model.feature_norms):
            _linear(layer, hidden, hidden)
            if type(norm) is not nn.LayerNorm or norm.normalized_shape != (hidden,):
                raise ValueError("feature LayerNorm shape mismatch")


def model_fingerprint(model, *, require_pairwise=False):
    """Bind validated state, topology and explicit non-parameter semantics.

    Do not call caller-overridable extra_repr() to obtain configuration.
    Fingerprints are local snapshot identifiers, not authority attestations.
    """
    validate_stock_model(model, require_pairwise=require_pairwise)
    modules, tensors = [], []
    for name, item in _modules(model):
        config = {}
        if type(item) is nn.Linear:
            config = dict(in_features=item.in_features, out_features=item.out_features,
                          bias=item.bias is not None)
        elif type(item) is nn.LayerNorm:
            config = dict(normalized_shape=item.normalized_shape, eps=item.eps,
                          elementwise_affine=item.elementwise_affine,
                          weight=item.weight is not None, bias=item.bias is not None)
        modules.append((name, type(item).__module__, type(item).__qualname__, item.training,
                        list(item._modules), config))
        for kind, mapping in (("parameter", item._parameters), ("buffer", item._buffers)):
            for key, value in mapping.items():
                if value is None:
                    tensors.append((name, kind, key, None))
                else:
                    array = value.detach().contiguous().numpy()
                    tensors.append((name, kind, key, str(array.dtype), array.shape,
                                    hashlib.sha256(array.tobytes()).hexdigest()))
    implementations = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("certificate.py", "model_contract.py")
    }
    payload = dict(configuration={key: getattr(model, key) for key in _FIELDS},
                   modules=modules, tensors=tensors, implementation=implementations)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf8")).hexdigest()
