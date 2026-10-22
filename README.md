# FedBCert

Code for **FedBCert: Endpoint-Blind Completion Certification for Federated
Polynomial Graph Predictors**.

FedBCert provides sufficient conditions for certifying that a frozen
predictor's class for a node is invariant under all admissible structural
completions. It uses a local induced subgraph, valid per-node count bounds and
bounded hidden representations. It neither reconstructs remote endpoints nor
retrains the predictor. An inconclusive certificate does not establish
instability, and class invariance does not guarantee prediction correctness.

Release 1.2.0 accompanies the October 21, 2026 manuscript. Its scientific
sources include the fusion and snapshot-contract fixes and the RG-1.0 runner
from commit `e99878f48ef441346011f27e5bdd63df3e568f63`. Compact historical
results and all 32 RG-1.0 case records are included. Datasets and trained
checkpoints are not bundled.

## Installation

Use Linux or WSL2. The validated environment uses Python **3.9.25**,
PyTorch **2.8.0+cpu** and PyTorch Geometric **2.6.1**. Package metadata supports
Python 3.9–3.11; the other interpreter versions have not received the same
validation. CPU execution is sufficient for the commands below. Native Windows
is not supported by historical replay programs, which use POSIX `resource`.

Download the source archive from the repository page you are viewing and
extract it. When using an anonymous review copy, obtain the files from that
copy rather than cloning a separate public repository. Open a terminal in the
extracted repository root (the directory containing `pyproject.toml`), then run:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip>=25.1,<26' setuptools==79.0.1 wheel==0.48.0
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-no-torch.txt
python -m pip install -e . --no-deps
python -m pip check
```

Select a Python 3.9.25 interpreter when matching the validated environment.
See [DEPENDENCIES.md](DEPENDENCIES.md) for package versions, optional solvers
and the scope of the retained environment lock.

## Quick review: no data download or benchmark training

Run from the repository root after installation:

```bash
python scripts/smoke_certificate.py
python scripts/demo_fedbcert_deployment.py --output /tmp/fedbcert-demo-new.json
python scripts/verify_rg1_reference.py
python scripts/verify_review_artifact.py
python scripts/verify_reference_results.py
python scripts/verify_gap_reference_results.py
python scripts/verify_on_demand_reference.py
python -m pytest -q -m 'not optional and not archive'
```

The demo output path must not already exist. The smoke and deployment examples
use tiny offline graphs. Reference commands check file integrity and recompute
statistics from included records; they do not rerun the trained models or the
original timing measurements. The test selection excludes external solvers
(`optional`) and missing parent-project archives (`archive`). Actual release
validation results are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).
Training-path tests use only tiny synthetic inputs and a few optimization steps.

A shorter implementation-focused test selection is:

```bash
python -m pytest -q tests/test_fedbcert_b1_fusion_contract.py \
  tests/test_fedbcert_b2_snapshot_contract.py tests/test_fedbcert_on_demand.py \
  tests/test_fedbcert_rg1_evaluation.py
```

These tests exercise rejection of unsupported configurations, snapshot/cache
checks, exact fallback and the six inference strategies on toy inputs.

## Repository layout

| Path | Contents |
| --- | --- |
| `openfgl/flcore/fedbcert/certificate.py` | Bounded polynomial predictors and class-margin certificates |
| `openfgl/flcore/fedbcert/model_contract.py` | Supported-model checks and inference-semantic fingerprints |
| `openfgl/flcore/fedbcert/deployment.py` | Frozen inference and Certified / Deferred / Unavailable outcomes |
| `openfgl/flcore/fedbcert/geometry_baselines.py` | Nonnegative-domain symmetric baselines |
| `openfgl/flcore/fedbcert/on_demand.py` | Exact multi-hop fallback, message codecs and batch-local caching |
| `scripts/`, `configs/`, `protocols/` | Experiment programs and fixed protocols |
| `reference_results/` | Historical benchmark, finite-oracle, synthetic and timing records |
| `results_boundary_certified/fedbcert_on_demand_20261015/` | Historical synthetic on-demand records |
| `rg1_data/` | All 32 real-graph cases, measurements and frozen statistics |
| `tests/` | Offline tests and marked archive/external checks |

## Paper results and reproduction scope

| Study | Included result | Review entry point |
| --- | --- | --- |
| Six-dataset same-predictor comparison | Coverage 80.03% → 91.31% | `verify_reference_results.py` |
| Finite small-graph oracle | Recall 82.05%; precision 100% | `verify_reference_results.py` |
| Controlled synthetic comparison | Exposed-node coverage 65.19% | `verify_review_artifact.py` |
| Historical synthetic on-demand replay | Paired payload reduction 22.09% | `verify_on_demand_reference.py` |
| RG-1.0: Cora and CiteSeer | P−I Macro-F1 +0.80 pp; S/D/N coverage 87.02% / 90.44% / 94.77% | `verify_rg1_reference.py` |

RG-1.0 contains two fixed graphs, two partitions and eight paired seeds:
32 cases and 96 trained models. Certificates are compared on the same frozen
P model. About 0.575% of test predictions differ between local and specified
complete-graph inference. For the full test batch, N reduces application payload
by **53.23%** on average across paired cases, at **3.16 times** the CPU time of F.
For 32-node batches, the paired relative reduction is **−281.83%**, while the
mean absolute N−F byte difference is approximately **−103,995 B**. N increases
bytes in the eight CiteSeer/METIS cases and decreases them in the other 24.
Mean paired relative changes and differences of mean bytes are distinct
statistics; small F denominators explain their different signs. See
[rg1_data/README.md](rg1_data/README.md) for units, fields and all adverse results.

The historical synthetic replay is a separate study: its original records show
22.09% paired payload reduction and about **23.8%** additional CPU time. A
retained-input replay after the contract fixes measured about **22.43%**
additional CPU time with unchanged message counts. These are different
implementation/timing records, not interchangeable measurements or evidence of
a causal speed change. The fixed-input replay timing is not bundled here.

The [reproduction guide](docs/REPRODUCING.md) distinguishes runnable offline
checks, experiments requiring public data, and archive-dependent replay.
**The full 96-model RG-1.0 runner cannot be rerun from this compact repository
alone.** Its original parent-project inputs, environment/approval manifests
and checkpoints are not bundled; the frozen runner's integrity gates are not
relaxed. Saved-record verification remains independently runnable.

## Interpretation and license

Count bounds must not underestimate true cross-client counts. The model and
representation assumptions must hold; arbitrary nonlinear GNNs are not covered.
The numerical acceptance guard and finite tests do not establish strict
floating-point soundness. Fixed-graph bootstrap intervals are descriptive, not
guarantees over independent graphs. Serialized application bytes do not measure
real network delay or deployment cost, and lower payload is not overall
inference acceleration.

OpenFGL components retain their [MIT license](LICENSE) and [attribution](NOTICE).
Optional repositories and datasets retain their own terms. Source and release
mapping are documented in [docs/PROVENANCE.md](docs/PROVENANCE.md).
