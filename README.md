# FedBCert

Code for **FedBCert: Endpoint-Blind Completion Certification for Federated
Polynomial Graph Predictors**.

FedBCert checks whether a frozen model's prediction for a node remains unchanged
under every admissible completion of hidden cross-client connections. It uses
the local induced graph, valid per-node upper bounds on missing neighbor counts,
and bounded hidden representations. It does not reconstruct remote endpoints or
retrain the predictor. Class stability is not a guarantee of correct prediction.

This repository corresponds to the October 17, 2026 manuscript revision. It
includes the core verifier, on-demand aggregation extension, fixed protocols,
tests and compact reference results. Original benchmark datasets and the full
trained-model archive are not included.

## Installation

Use Linux or WSL2 with Python 3.9–3.11. The published timing environment used
Python 3.9.25, PyTorch 2.8.0 and PyTorch Geometric 2.6.1. CPU is sufficient for
the checks below. Historical replay uses POSIX `resource`, so native Windows is
not a supported replay platform.

```bash
git clone https://github.com/sdxy-ai/FedBCert.git
cd FedBCert
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-no-torch.txt
python -m pip install -e . --no-deps
python -m pip check
```

`requirements-no-torch.txt` includes PyG and all other fixed dependencies;
it excludes only PyTorch. For CUDA, use the corresponding PyTorch 2.8.0 command
from the [official guide](https://pytorch.org/get-started/previous-versions/#v280).
Do not upgrade numerical dependencies when comparing saved results. See
[DEPENDENCIES.md](DEPENDENCIES.md) for versions and optional comparisons.

## Quick checks: no data download or training

From the repository root:

```bash
python scripts/smoke_certificate.py
python scripts/verify_review_artifact.py
python scripts/verify_reference_results.py
python scripts/verify_gap_reference_results.py
python scripts/verify_on_demand_reference.py
python -m pytest -q -m 'not optional'
```

The first command checks a small synthetic graph. Reference checks validate
saved hashes and recompute included statistics. Tests cover certificates, input
checks, serialization and replay control flow. None of these commands retrains
the reported models or reruns the original timing experiment. Passing them is
not a complete audit of the omitted checkpoint archive.

Local results and their environment are recorded in
[docs/VALIDATION.md](docs/VALIDATION.md). An optional CPU CI configuration is
provided in `docs/ci-tests.yml.example`; it is not enabled in this release.

## Code layout

| Location | Contents |
|---|---|
| `openfgl/flcore/fedbcert/certificate.py` | Bounded polynomial predictor, retained-weight bounds and class-difference certificates |
| `openfgl/flcore/fedbcert/federated.py` | Client training, FedAvg and evaluation |
| `openfgl/flcore/fedbcert/deployment.py` | Frozen snapshots and Certified / Deferred / Unavailable outcomes |
| `openfgl/flcore/fedbcert/geometry_baselines.py` | Nonnegative-domain symmetric baselines |
| `openfgl/flcore/fedbcert/on_demand.py` | Exact fallback, serialized messages and batch-local caching |
| `configs/` | Fixed settings and seed lists |
| `reference_results/` | Saved benchmark, exact-small-graph, synthetic and timing results |
| `results_boundary_certified/fedbcert_on_demand_20261015/` | Saved on-demand records and per-node diagnostics |
| `tests/` | Portable tests; external solver tests are marked `optional` |

## Paper results and reproduction

The [reproduction guide](docs/REPRODUCING.md) maps results to scripts, inputs and
reference files. The included evidence supports the following checks:

| Result | Saved evidence | Check without original checkpoints |
|---|---|---|
| Coverage: 80.03% → 91.31% | `reference_results/main_confirmation/` | Saved values and checksums |
| Exact recall: 82.05%; precision: 100% | `reference_results/exact_worst_case/` | Saved statistics; optional finite-oracle rerun |
| Synthetic exposed coverage: 65.19% | `reference_results/e28b_confirmation/` | Paired seed statistics and tests |
| Certificate-only CPU cost | `reference_results/gaps/g2/` | Statistics from 1,200 saved measurements |
| On-demand payload reduction: 22.09% | `results_boundary_certified/fedbcert_on_demand_20261015/pilot/` | All 240 saved records and per-seed comparisons |

The on-demand study is a retrospective, single-process protocol replay on eight
frozen synthetic graph seeds. All output classes match the specified complete
graph. Serialized application payload falls by 22.09%, but CPU time increases
by about 23.8% relative to direct complete-graph inference. This is not a real
network deployment or an overall speedup. Repeating its computations requires
original graph bundles and selected states that are not included here.

Optional historical synthetic retraining:

```bash
python scripts/replay_fedbcert_e28b.py --full-replay \
  --output /tmp/fedbcert-historical-replay-new
```

This runs **96 graphs and 288 full 100-round trainings**, not a quick check.
The output directory must not exist. It reuses published seeds, not new
independent confirmation. Bitwise-identical GPU retraining across platforms
is not promised. No full retraining was performed to prepare this release.

## Limitations

- Count bounds must not underestimate actual cross-client counts. Hashes and
  signatures do not prove that a supplied bound is true.
- The theory covers finite-order row-stochastic polynomial propagation with
  bounded representations, not arbitrary nonlinear GNNs. The nonnegative rule
  requires nonnegative local and remote representations.
- The numerical acceptance threshold is not an outward-rounded floating-point
  proof. Finite tests do not replace the mathematical argument.
- Parameter-matched real-graph controls did not establish an independent
  propagation benefit; the teacher extension retains its inconclusive
  graph-complementarity result.
- Missing historical inputs limit reconstruction of some benchmark and external
  verifier comparisons. These cases are identified in the reproduction guide.

## License and provenance

Included OpenFGL components retain their [MIT license](LICENSE) and
[attribution](NOTICE). Optional repositories and datasets have their own terms.
[docs/PROVENANCE.md](docs/PROVENANCE.md) records the source archives, preserved
manifests and development disclosure.

For bug reports, provide the command, package versions and a minimal example in
GitHub Issues. Do not upload private data or credentials.
