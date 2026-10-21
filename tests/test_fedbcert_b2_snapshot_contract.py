"""B2 regressions for the aggregate replay's frozen stock-model contract.

Only tiny synthetic fixtures are used: no labels, historical checkpoint,
training, or paper-result access is needed.
"""
from __future__ import annotations

import pytest
from collections import UserDict
import torch
from torch import nn

from openfgl.flcore.fedbcert.certificate import (
    PolynomialSpectralClassifier, row_normalized_shift,
)
from openfgl.flcore.fedbcert.on_demand import (
    AggregateReplayService, state_fingerprint,
)
from openfgl.flcore.fedbcert.deployment import FrozenLocalDeployment


class _PredictorSubclass(PolynomialSpectralClassifier):
    pass


class _LinearSubclass(nn.Linear):
    pass


def _fixture(model_type=PolynomialSpectralClassifier):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(28220)
        model = model_type(
            input_dim=3, hidden_dim=5, num_classes=3, order_count=3,
            dropout=0.2, encoder_type="dual_feature_spectral",
            encoder_depth=2, spectral_fusion_scale=0.25,
            certificate_margin_mode="pairwise_difference",
        ).double().eval()
        with torch.no_grad():
            model.theta.copy_(torch.tensor([0.4, -0.2, 0.8], dtype=torch.float64))
        features = torch.randn(8, 3, dtype=torch.float64)
    edges = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 1], [1, 2, 3, 4, 5, 6, 7, 6]],
        dtype=torch.long,
    )
    owners = torch.tensor([0, 0, 1, 1, 2, 2, 0, 2], dtype=torch.long)
    return model, features, edges, owners


def _central(model, features, edges):
    shift, _ = row_normalized_shift(
        edges, features.size(0), device=torch.device("cpu"), dtype=features.dtype,
    )
    with torch.no_grad():
        return model(features, shift)


OPERATIONS = ("local_view", "nominal", "metadata", "exact")


def _call(service, operation):
    if operation == "local_view":
        return service.local_view(0)
    if operation == "nominal":
        return service.nominal(0)
    if operation == "metadata":
        return service.metadata(0, "upper_count")
    if operation == "exact":
        return service.exact([0, 3])
    raise AssertionError(operation)


def _mutate(model, name):
    if name == "layer_norm_eps":
        model.feature_norms[0].eps *= 17.0
    elif name == "layer_norm_affine":
        model.feature_norms[0].elementwise_affine = False
    elif name == "layer_norm_bias":
        model.feature_norms[0].bias = None
    elif name == "linear_in_features":
        model.encoder.in_features += 1
    elif name == "linear_out_features":
        model.classifier.out_features += 1
    elif name == "fusion_scale":
        model.spectral_fusion_scale += 0.125
    elif name == "dropout":
        model.dropout = 0.0
    elif name == "child_training":
        model.feature_norms[0].train()
    elif name == "parameters":
        with torch.no_grad():
            model.theta[0].add_(0.01)
    else:
        raise AssertionError(name)


def test_constructor_rejects_predictor_subclass():
    model, features, edges, owners = _fixture(_PredictorSubclass)
    with pytest.raises(ValueError):
        AggregateReplayService(model, features, edges, owners)


def test_constructor_rejects_nonstock_child_module_type():
    model, features, edges, owners = _fixture()
    original = model.encoder
    replacement = _LinearSubclass(
        original.in_features, original.out_features, bias=original.bias is not None,
    ).double().eval()
    replacement.load_state_dict(original.state_dict())
    model.encoder = replacement
    with pytest.raises(ValueError):
        AggregateReplayService(model, features, edges, owners)


@pytest.mark.parametrize("target", [
    "forward", "bounded_encoding", "feature_encoding", "filter", "encoder.forward",
])
def test_constructor_rejects_instance_method_overrides(target):
    model, features, edges, owners = _fixture()
    module, attribute = (model.encoder, "forward") if target == "encoder.forward" else (model, target)
    setattr(module, attribute, lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        AggregateReplayService(model, features, edges, owners)


@pytest.mark.parametrize("hook", [
    "register_forward_hook", "register_forward_pre_hook",
    "register_full_backward_hook", "register_full_backward_pre_hook",
    "register_state_dict_pre_hook", "register_state_dict_post_hook",
    "register_load_state_dict_pre_hook", "register_load_state_dict_post_hook",
])
def test_constructor_rejects_module_hooks(hook):
    model, features, edges, owners = _fixture()
    handle = getattr(model.encoder, hook)(lambda *args, **kwargs: None)
    try:
        with pytest.raises(ValueError):
            AggregateReplayService(model, features, edges, owners)
    finally:
        handle.remove()


@pytest.mark.parametrize("hook", [
    "register_module_forward_hook", "register_module_forward_pre_hook",
    "register_module_full_backward_hook", "register_module_full_backward_pre_hook",
])
def test_constructor_rejects_global_hooks_without_leaving_them_registered(hook):
    model, features, edges, owners = _fixture()
    handle = getattr(nn.modules.module, hook)(lambda *args, **kwargs: None)
    try:
        with pytest.raises(ValueError):
            AggregateReplayService(model, features, edges, owners)
    finally:
        handle.remove()


@pytest.mark.parametrize("mutation", [
    "layer_norm_eps", "layer_norm_affine", "layer_norm_bias",
    "linear_in_features", "linear_out_features", "fusion_scale",
    "dropout", "child_training", "parameters",
])
def test_configuration_change_changes_fingerprint_or_is_rejected(mutation):
    model, *_ = _fixture()
    before = state_fingerprint(model)
    _mutate(model, mutation)
    try:
        after = state_fingerprint(model)
    except ValueError:
        # A structurally inconsistent or non-eval model must not receive the
        # unchanged supported-model fingerprint; outright rejection is stricter.
        return
    assert after != before


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("mutation", [
    "layer_norm_eps", "layer_norm_affine", "layer_norm_bias",
    "linear_in_features", "linear_out_features", "child_training",
])
def test_public_operations_reject_configuration_mutation_before_cache_use(
    mutation, cached, operation,
):
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    if cached:
        assert service.nominal(0).encoded_rows > 0
        assert service.nominal(0).encoded_rows == 0
    _mutate(model, mutation)
    with pytest.raises(ValueError):
        _call(service, operation)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
@pytest.mark.parametrize("mutation", ["instance_override", "module_hook", "global_hook"])
def test_new_overrides_and_hooks_rejected_after_service_creation(
    mutation, cached, operation,
):
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    if cached:
        service.nominal(0)
    handle = None
    try:
        if mutation == "instance_override":
            model.bounded_encoding = lambda *args, **kwargs: None
        elif mutation == "module_hook":
            handle = model.encoder.register_forward_hook(lambda *args, **kwargs: None)
        else:
            handle = nn.modules.module.register_module_forward_hook(lambda *args, **kwargs: None)
        with pytest.raises(ValueError):
            _call(service, operation)
    finally:
        if handle is not None:
            handle.remove()


def test_fresh_service_accepts_new_valid_eps_without_old_nominal_cache():
    model, features, edges, owners = _fixture()
    stale = AggregateReplayService(model, features, edges, owners)
    stale.nominal(0)
    old_hash, old_snapshot = stale.snapshot_model_hash, stale.snapshot_id
    model.feature_norms[0].eps = 0.2
    with pytest.raises(ValueError):
        stale.nominal(0)
    fresh = AggregateReplayService(model, features, edges, owners)
    assert fresh.snapshot_model_hash != old_hash
    assert fresh.snapshot_id != old_snapshot
    view = fresh.local_view(0)
    nominal = fresh.nominal(0)
    assert nominal.encoded_rows == view.nodes.numel()
    assert fresh.nominal(0).encoded_rows == 0
    with torch.no_grad():
        assert torch.equal(nominal.logits, model(view.features, view.shift))
    actual = fresh.exact(range(features.size(0)))
    assert torch.allclose(actual.logits, _central(model, features, edges), atol=2e-13, rtol=2e-13)
    assert actual.accounting["reused_encoded_rows"] > 0
    assert actual.accounting["encoded_rows"] > 0


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "cached"])
def test_mutating_public_local_view_does_not_pollute_service_data_or_caches(cached):
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    expected = service.local_view(0)
    if cached:
        service.nominal(0)
    returned = service.local_view(0)
    returned.nodes.zero_()
    returned.features.zero_()
    returned.shift.values().zero_()
    returned.retained_degree.zero_()
    clean = service.local_view(0)
    assert torch.equal(clean.nodes, expected.nodes)
    assert torch.equal(clean.features, expected.features)
    assert torch.equal(clean.shift.indices(), expected.shift.indices())
    assert torch.equal(clean.shift.values(), expected.shift.values())
    assert torch.equal(clean.retained_degree, expected.retained_degree)
    nominal = service.nominal(0)
    with torch.no_grad():
        assert torch.equal(nominal.logits, model(clean.features, clean.shift))
    actual = service.exact([0, 3])
    reference = _central(model, features, edges)
    assert torch.allclose(actual.logits, reference[actual.nodes], atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("target", ["predictor_forward", "linear_forward", "layernorm_forward", "extra_repr", "new_descriptor", "compiled_call"])
@pytest.mark.parametrize("cached", [False, True])
def test_changed_stock_class_or_compiled_behavior_is_rejected(monkeypatch, target, cached):
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    if cached:
        service.nominal(0)
    if target == "predictor_forward":
        monkeypatch.setattr(PolynomialSpectralClassifier, "forward", lambda *args: None)
    elif target == "linear_forward":
        monkeypatch.setattr(nn.Linear, "forward", lambda *args: None)
    elif target == "layernorm_forward":
        monkeypatch.setattr(nn.LayerNorm, "forward", lambda *args: None)
    elif target == "extra_repr":
        monkeypatch.setattr(nn.LayerNorm, "extra_repr", lambda *args: "unchanged")
    elif target == "new_descriptor":
        monkeypatch.setattr(nn.LayerNorm, "unexpected_descriptor", property(lambda self: 0), raising=False)
    else:
        model._compiled_call_impl = lambda *args: None
    with pytest.raises(ValueError):
        service.exact([0])
    with pytest.raises(ValueError):
        FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")


@pytest.mark.parametrize("change", ["nan_eps", "negative_eps", "normalized_shape", "bias_shape", "feature_width", "module_mode", "parameter_dtype"])
def test_shared_contract_rejects_inconsistent_or_nonfinite_configs(change):
    model, features, edges, owners = _fixture()
    if change == "nan_eps":
        model.feature_norms[0].eps = float("nan")
    elif change == "negative_eps":
        model.feature_norms[0].eps = -1.
    elif change == "normalized_shape":
        model.feature_norms[0].normalized_shape = (6,)
    elif change == "bias_shape":
        model.feature_norms[0].bias = nn.Parameter(torch.zeros(6, dtype=torch.float64))
    elif change == "feature_width":
        model.feature_encoder.out_features += 1
    elif change == "module_mode":
        model.encoder.train()
    else:
        model.classifier.weight = nn.Parameter(model.classifier.weight.float())
    with pytest.raises(ValueError):
        AggregateReplayService(model, features, edges, owners)
    with pytest.raises(ValueError):
        FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")


def test_layernorm_bias_absence_is_bound_and_fresh_stock_instance_still_supported():
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    service.nominal(0)
    model.feature_norms[0].bias = None
    with pytest.raises(ValueError):
        service.exact([0])
    fresh = AggregateReplayService(model, features, edges, owners)
    actual = fresh.exact([0, 3])
    assert torch.allclose(actual.logits, _central(model, features, edges)[actual.nodes], atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("registry", ["_forward_hooks", "_forward_pre_hooks", "_state_dict_hooks", "_load_state_dict_pre_hooks", "_modules", "_parameters", "_buffers"])
@pytest.mark.parametrize("cached", [False, True])
def test_custom_registry_mappings_are_rejected_before_hooks_or_cache_use(registry, cached):
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    if cached:
        service.nominal(0)
    target = model.encoder
    # UserDict isn't a dict; an isinstance(dict)-only hooks check misses it.
    replacement = UserDict(getattr(target, registry))
    if "hooks" in registry:
        replacement[999] = lambda *args: pytest.fail("unsupported hook executed")
    setattr(target, registry, replacement)
    with pytest.raises(ValueError):
        service.nominal(0)
    with pytest.raises(ValueError):
        FrozenLocalDeployment(model, features, edges, view_version="v", preprocessing_version="p")


def test_mutating_function_code_in_place_is_detected(monkeypatch):
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    service.nominal(0)
    original = nn.Linear.forward
    monkeypatch.setattr(original, "__code__", (lambda *args: None).__code__)
    with pytest.raises(ValueError, match="implementation changed"):
        service.exact([0])


def test_custom_global_hook_registry_is_rejected(monkeypatch):
    model, features, edges, owners = _fixture()
    service = AggregateReplayService(model, features, edges, owners)
    service.nominal(0)
    monkeypatch.setattr(nn.modules.module, "_global_forward_hooks", UserDict({999: lambda *args: None}))
    with pytest.raises(ValueError):
        service.nominal(0)
