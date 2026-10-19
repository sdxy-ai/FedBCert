"""Isolated E28b generator and validation-only controlled experiment helpers."""
from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from openfgl.flcore.fedbcert.certificate import (
    PolynomialSpectralClassifier, boundary_uncertainty, certificate_statistics,
    cone_pairwise_certificate_statistics, induced_edges, missing_degrees,
    row_normalized_shift, uncertainty_basis,
)
from openfgl.flcore.fedbcert.federated import (
    _identity_shift_like, build_boundary_ledger_views,
    build_federated_clients_from_ledgers, cpu_state, train_client,
    weighted_average_states,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/fedbcert_e28b_controlled_protocol_v2.json"
MODEL_NAMES = ("observed_p", "identity", "strong_mlp")
CERT_NAMES = ("additive_classwise", "retained_pairwise", "cone_only", "cone_hybrid")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def tensor_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def derive_seed(seed, stream, *extra):
    return int(np.random.SeedSequence([int(seed), int(stream), *map(int, extra)]).generate_state(1)[0])


def run_matrix(protocol):
    return [dict(seed=seed, h=h, rho=rho, model=model,
                 graph_id=f"h{h:.2f}_rho{rho:.2f}_seed{seed}")
            for seed in protocol["seeds"]["development"]
            for h in protocol["development"]["homophily"]
            for rho in protocol["development"]["cross_client_mass"]
            for model in MODEL_NAMES]


def pair_probability(h, rho, same_client, same_class, block_size, degree=12.0):
    mass = h if same_class else 1.0 - h
    if same_client:
        denominator = block_size - 1 if same_class else block_size
        return (1.0 - rho) * degree * mass / denominator
    return rho * degree * mass / (2 * block_size)


def generate_graph(protocol, seed, h, rho, *, block_size=None):
    """One Bernoulli draw per unordered candidate pair; independent RNG streams.

    block_size is test/engineering-only. The development CLI never overrides it.
    """
    g = protocol["generator"]
    m = int(block_size or g["nodes_per_client_class"])
    if m < 20 or m % 10:
        raise ValueError("block size must be >=20 and divisible by ten")
    if g["clients"] != 3 or g["classes"] != 2 or not (0 <= rho <= 1 and 0 <= h <= 1):
        raise ValueError("unsupported generator contract")
    membership = np.repeat(np.arange(3), 2 * m)
    labels = np.tile(np.repeat(np.arange(2), m), 3)
    seeds = {name: derive_seed(seed, sid) for name, sid in protocol["seeds"]["streams"].items()}
    rng = np.random.default_rng(seeds["edges"])
    pairs = []
    for left in range(6):
        for right in range(left, 6):
            probability = pair_probability(h, rho, left // 2 == right // 2,
                                           left % 2 == right % 2, m,
                                           g["expected_degree_excluding_self"])
            if not 0 <= probability <= 1:
                raise ValueError("invalid Bernoulli probability")
            if left == right:
                a, b = np.triu_indices(m, 1)
            else:
                a = np.repeat(np.arange(m), m)
                b = np.tile(np.arange(m), m)
            chosen = rng.random(len(a)) < probability
            pairs.append(np.stack((a[chosen] + left * m, b[chosen] + right * m)))
    undirected = np.concatenate(pairs, axis=1)
    edges = torch.as_tensor(np.concatenate((undirected, undirected[::-1]), axis=1), dtype=torch.long)
    features = np.random.default_rng(seeds["features"]).normal(size=(6 * m, g["feature_dim"]))
    features[:, 0] += (2 * labels - 1) * g["feature_signal"]
    masks = {name: np.zeros(6 * m, dtype=bool) for name in g["splits"]}
    split_rng = np.random.default_rng(seeds["splits"])
    fractions = list(g["splits"].items())
    for block in range(6):
        nodes = split_rng.permutation(np.arange(block * m, (block + 1) * m))
        start = 0
        for name, fraction in fractions:
            count = int(round(fraction * m))
            masks[name][nodes[start:start + count]] = True
            start += count
        if start != m:
            raise ValueError("splits must cover each stratum")
    local_edges = induced_edges(edges, membership)
    boundary = missing_degrees(edges, local_edges, 6 * m)
    cut = membership[undirected[0]] != membership[undirected[1]]
    same = labels[undirected[0]] == labels[undirected[1]]
    metadata = dict(seed=int(seed), h=float(h), rho=float(rho), nodes=6 * m,
                    edges=int(undirected.shape[1]), mean_degree=2 * undirected.shape[1] / (6 * m),
                    actual_homophily=float(same.mean()) if len(same) else None,
                    actual_cut_ratio=float(cut.mean()) if len(cut) else 0.0,
                    boundary_nodes=int((boundary > 0).sum()), derived_seeds=seeds,
                    split_counts={key: int(mask.sum()) for key, mask in masks.items()})
    return dict(features=torch.as_tensor(features, dtype=torch.float32),
                labels=torch.as_tensor(labels), membership=torch.as_tensor(membership),
                full_edge_index=edges, missing_degree=boundary,
                masks={key: torch.as_tensor(value) for key, value in masks.items()}, metadata=metadata)


def development_labels(data):
    """No audit/test targets are passed to training or diagnostics."""
    allowed = data["masks"]["train"] | data["masks"]["validation"]
    labels = data["labels"].clone()
    labels[~allowed] = -1
    return labels


def make_model(protocol, name, *, dtype=torch.float32, device="cpu"):
    spec, t = protocol["models"][name], protocol["training"]
    model = PolynomialSpectralClassifier(
        protocol["generator"]["feature_dim"], spec["hidden_dim"], 2, t["order_count"],
        t["dropout"], spec["encoder"], spec["encoder_depth"], t["residual_scale"],
        spectral_fusion_scale=0.0 if name == "strong_mlp" else t["alpha"],
        certificate_margin_mode="pairwise_difference",
    ).to(device=device, dtype=dtype)
    if sum(p.numel() for p in model.parameters()) != spec["parameters"]:
        raise ValueError("parameter-count contract mismatch")
    return model


def make_clients(protocol, data, device, name="observed_p", *, dtype=torch.float32):
    ledgers = build_boundary_ledger_views(data["full_edge_index"], data["membership"].numpy(), data["missing_degree"])
    clients = build_federated_clients_from_ledgers(
        ledgers, data["features"], development_labels(data), data["masks"]["train"],
        data["masks"]["validation"], torch.zeros_like(data["masks"]["test_sealed"]),
        protocol["training"]["order_count"], torch.device(device), "retained_mass")
    # Rebuild stochastic weights in float64 rather than casting float32 values.
    if dtype == torch.float64:
        from openfgl.flcore.fedbcert.certificate import retained_mass_uncertainty_basis
        for client, ledger in zip(clients, ledgers):
            client.features = client.features.double()
            client.shift, degree = row_normalized_shift(ledger.local_edge_index, len(ledger.global_nodes), device=torch.device(device), dtype=dtype)
            client.missing_degree = client.missing_degree.double()
            delta = boundary_uncertainty(degree, client.missing_degree)
            client.uncertainty_basis = retained_mass_uncertainty_basis(client.shift, delta, protocol["training"]["order_count"])
    if name != "observed_p":
        clients = [replace(client, shift=_identity_shift_like(client.shift),
                           uncertainty_basis=[torch.zeros_like(v) for v in client.uncertainty_basis])
                   for client in clients]
    return clients


def exposure_groups(client, order_count):
    boundary = client.missing_degree > 0
    exposed = boundary.clone()
    for _ in range(max(order_count - 2, 0)):
        exposed |= torch.sparse.mm(client.shift, exposed.to(client.features.dtype)[:, None]).squeeze(1) > 0
    return dict(all=torch.ones_like(exposed), boundary=boundary,
                near_boundary=exposed & ~boundary, exposed=exposed, unexposed=~exposed)


def macro_f1(labels, predictions):
    if len(labels) == 0:
        return None
    return float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0))


@torch.no_grad()
def validation_score(model, clients):
    model.eval()
    labels, predictions = [], []
    for client in clients:
        predictions.append(model(client.features, client.shift)[client.validation_mask].argmax(1).cpu())
        labels.append(client.labels[client.validation_mask].cpu())
    return macro_f1(torch.cat(labels).numpy(), torch.cat(predictions).numpy())


def resource_usage(device):
    return dict(process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
                cuda_peak_gib=(torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0))


def check_resources(protocol, device, started):
    caps = protocol["stopping"]["resource_caps"]
    usage = resource_usage(device)
    if time.monotonic() - started > caps["seconds_per_training_run"]:
        raise RuntimeError("per-run wall-time resource cap")
    if any(usage[key] > caps[key] for key in usage):
        raise RuntimeError("memory resource cap")
    return usage


def train_model(protocol, data, name, device, output, *, rounds=None):
    t, spec = protocol["training"], protocol["models"][name]
    model_seed = data["metadata"]["derived_seeds"]["model"]
    torch.manual_seed(model_seed)
    model = make_model(protocol, name, device=device)
    initial = cpu_state(model)
    initial_digest = tensor_digest(initial)
    clients = make_clients(protocol, data, device, name)
    state, best, best_f1, best_round = initial, None, -1.0, 0
    trajectory = []
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for round_id in range(1, (rounds or t["rounds"]) + 1):
        states = []
        for client in clients:
            states.append(train_client(
                state, client, input_dim=protocol["generator"]["feature_dim"],
                hidden_dim=spec["hidden_dim"], num_classes=2, order_count=t["order_count"],
                dropout=t["dropout"], encoder_type=spec["encoder"], encoder_depth=spec["encoder_depth"],
                encoder_residual_scale=t["residual_scale"], robust_lambda=0.0, certificate_margin=0.0,
                local_epochs=t["local_epochs"], learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
                seed=derive_seed(model_seed, round_id, client.client_id), device=device,
                spectral_fusion_scale=0.0 if name == "strong_mlp" else t["alpha"],
                certificate_margin_mode="pairwise_difference"))
        state = weighted_average_states(states, [client.train_count for client in clients])
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError("nonfinite state")
        model.load_state_dict(state)
        score = validation_score(model, clients)
        trajectory.append(dict(round=round_id, validation_macro_f1=score))
        if score > best_f1:
            best, best_f1, best_round = {k: v.clone() for k, v in state.items()}, score, round_id
        check_resources(protocol, device, start)
    if output is not None:
        output.mkdir(exist_ok=False)
        torch.save(initial, output / "initial_state.pt")
        torch.save(best, output / "selected_state.pt")
        write_json(output / "validation_trajectory.json", trajectory)
    return best, dict(model=name, initial_tensor_sha256=initial_digest,
                      selected_tensor_sha256=tensor_digest(best), selected_round=best_round,
                      selected_validation_macro_f1=best_f1, trainable_parameters=spec["parameters"],
                      training_seconds=time.monotonic() - start, device=str(device), **resource_usage(device))


@torch.no_grad()
def evaluate_validation(protocol, data, name, state):
    """Freeze the state first. Only validation-indexed arrays leave this helper."""
    start = time.monotonic()
    tol = protocol["diagnostic"]["tolerance"]
    model = make_model(protocol, name, dtype=torch.float64).eval()
    model.load_state_dict(state)
    reference = make_clients(protocol, data, "cpu", dtype=torch.float64)
    active = reference if name == "observed_p" else make_clients(protocol, data, "cpu", name, dtype=torch.float64)
    # Identity is invariant in every completed world, not merely nominal I.
    full_shift, _ = row_normalized_shift(data["full_edge_index"], len(data["features"]), device=torch.device("cpu"), dtype=torch.float64)
    full_logits = model(data["features"].double(), full_shift if name == "observed_p" else _identity_shift_like(full_shift))
    collected = {key: [] for key in ("nodes", "labels", "nominal_logits", "completion_logits", "softmax_change", "flip")}
    group_arrays = {key: [] for key in ("all", "boundary", "near_boundary", "exposed", "unexposed")}
    certificates = {key: [] for key in CERT_NAMES} if name == "observed_p" else {"structural_invariance": []}
    slacks = {key: [] for key in certificates}
    radii, violations = [], []
    max_hidden_excess = max_logit_excess = max_cone_excess = -float("inf")
    full_hidden = model.bounded_encoding(data["features"].double()) if name == "observed_p" else None
    full_powers = [full_hidden] if full_hidden is not None else []
    for _ in range(1, model.order_count):
        if full_hidden is not None:
            full_powers.append(torch.sparse.mm(full_shift, full_powers[-1]))
    for client, ref in zip(active, reference):
        mask, nodes = client.validation_mask, client.global_nodes
        logits = model(client.features, client.shift)
        completed = full_logits[nodes]
        prediction = logits.argmax(1)
        flip = prediction != completed.argmax(1)
        change = (logits.softmax(1) - completed.softmax(1)).abs().max(1).values
        if name == "observed_p":
            # d_v is recovered from the exact retained self-loop weight.
            diagonal = client.shift.indices()[0] == client.shift.indices()[1]
            degree = client.shift.values()[diagonal].reciprocal()
            delta = boundary_uncertainty(degree, client.missing_degree)
            model.certificate_margin_mode = "classwise_sum"
            additive = certificate_statistics(model, logits, uncertainty_basis(client.shift, delta, model.order_count))
            model.certificate_margin_mode = "pairwise_difference"
            symmetric = certificate_statistics(model, logits, client.uncertainty_basis)
            cone = cone_pairwise_certificate_statistics(model, client.features, client.shift, delta, logits, client.uncertainty_basis, count_semantics="upper_bound", combine_symmetric=False)
            stats = {"additive_classwise": additive, "retained_pairwise": symmetric, "cone_only": cone}
            stats["cone_hybrid"] = replace(cone, worst_slack=torch.maximum(symmetric.worst_slack, cone.worst_slack))
            hidden = model.bounded_encoding(client.features)
            for k in range(model.order_count):
                if k:
                    hidden = torch.sparse.mm(client.shift, hidden)
                excess = ((full_powers[k][nodes] - hidden).norm(dim=1) - client.uncertainty_basis[k])[mask].max().item()
                max_hidden_excess = max(max_hidden_excess, excess)
            max_logit_excess = max(max_logit_excess, ((completed - logits).abs() - symmetric.class_logit_bounds)[mask].max().item())
            other = 1 - prediction  # E28b has two classes.
            completed_margin = completed.gather(1, prediction[:, None]).squeeze(1) - completed.gather(1, other[:, None]).squeeze(1)
            max_cone_excess = max(max_cone_excess, (cone.worst_slack - completed_margin)[mask].max().item())
            for key, stat in stats.items():
                cert = stat.worst_slack > tol
                certificates[key].append(cert[mask])
                slacks[key].append(stat.worst_slack[mask])
                bad = mask & cert & flip
                violations.extend(nodes[bad].tolist())
            radii.append(symmetric.representation_radius[mask])
        else:
            if bool(flip[mask].any()) or not torch.allclose(logits[mask], completed[mask], rtol=0, atol=tol):
                raise RuntimeError("graph invariant control changed under completion")
            certificates["structural_invariance"].append(torch.ones(int(mask.sum()), dtype=torch.bool))
            nominal_margin = logits.max(1).values - logits.min(1).values
            slacks["structural_invariance"].append(nominal_margin[mask])
            radii.append(torch.zeros(int(mask.sum()), dtype=torch.float64))
        for key, array in dict(nodes=nodes, labels=client.labels, nominal_logits=logits,
                               completion_logits=completed, softmax_change=change, flip=flip).items():
            collected[key].append(array[mask].cpu())
        for key, array in exposure_groups(ref, model.order_count).items():
            group_arrays[key].append(array[mask])
    evidence = {key: torch.cat(value) for key, value in collected.items()}
    evidence["groups"] = {key: torch.cat(value) for key, value in group_arrays.items()}
    evidence["certificates"] = {key: torch.cat(value) for key, value in certificates.items()}
    evidence["slacks"] = {key: torch.cat(value) for key, value in slacks.items()}
    evidence["radius"] = torch.cat(radii)
    if not all(torch.isfinite(evidence[k]).all() for k in ("nominal_logits", "completion_logits", "softmax_change", "radius")):
        raise RuntimeError("nonfinite validation evidence")
    rows = []
    prediction = evidence["nominal_logits"].argmax(1)
    for group, mask in evidence["groups"].items():
        n = int(mask.sum())
        for certificate, cert in evidence["certificates"].items():
            correct = prediction == evidence["labels"]
            rows.append(dict(group=group, certificate=certificate, n=n,
                             macro_f1=macro_f1(evidence["labels"][mask].numpy(), prediction[mask].numpy()),
                             correct_n=int((mask & correct).sum()), certified_n=int((mask & cert).sum()),
                             certified_correct_n=int((mask & cert & correct).sum()),
                             certified_flip_n=int((mask & cert & evidence["flip"]).sum()),
                             category_flip_n=int((mask & evidence["flip"]).sum()),
                             softmax_change_ge_005_n=int((mask & (evidence["softmax_change"] >= .05)).sum()),
                             certified_ratio=float(cert[mask].double().mean()) if n else None,
                             certified_accuracy=float((cert & correct)[mask].double().mean()) if n else None,
                             category_flip_fraction=float(evidence["flip"][mask].double().mean()) if n else None,
                             softmax_change_fraction=float((evidence["softmax_change"][mask] >= .05).double().mean()) if n else None,
                             mean_radius=float(evidence["radius"][mask].mean()) if n else None))
    bound_values = dict(hidden_bound_max_excess=max_hidden_excess,
                        symmetric_logit_bound_max_excess=max_logit_excess,
                        cone_margin_bound_max_excess=max_cone_excess) if name == "observed_p" else {}
    reliability = dict(certified_flip_nodes=sorted(set(violations)), bound_violations=sum(v > tol for v in bound_values.values()),
                       numeric_tolerance=tol, **bound_values)
    evidence["reliability"] = reliability
    return rows, evidence, dict(certification_diagnostic_seconds=time.monotonic() - start, reliability=reliability)
