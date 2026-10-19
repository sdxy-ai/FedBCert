# Reproducing the results

There are three different tasks: checking included reference results, rerunning
an experiment, and reproducing a historical computation with its original
inputs. The quick commands in the README do the first task and unit testing.
They do not imply the other two have been completed.

## Reference checks

| Command | What it checks |
|---|---|
| `python scripts/smoke_certificate.py` | Certificates on a small offline graph, without training |
| `python scripts/verify_review_artifact.py` | Release hashes, preserved scientific sources and synthetic paired statistics |
| `python scripts/verify_reference_results.py` | Saved benchmark, finite-oracle and related headline values |
| `python scripts/verify_gap_reference_results.py` | Saved timing measurements and external-verifier availability records |
| `python scripts/verify_on_demand_reference.py` | Source/evidence bindings, 240 saved records and their paired bootstrap summaries |

The on-demand reference command does not run the original analyzer CLI. That
CLI first verifies all 137 original input files and refuses to overwrite an
existing analysis directory. Those guards are retained in the original source.
The new command recomputes statistics without loading graph/model tensors; it
does not claim to have verified absent raw inputs. Its numerical analysis uses
the unchanged original analyzer functions.

## Experiment map

Paths below are relative to the repository root. Experiment identifiers in
filenames are retained so that frozen manifests and saved outputs still match.

| Paper component | Protocol / entry point | Saved results |
|---|---|---|
| Six-dataset same-predictor comparison | `configs/fedbcert_e18_cone_protocol.json`; `bash scripts/reproduce.sh confirmation` | `reference_results/main_confirmation/` |
| Graph-weight trade-off and ablations | `configs/fedbcert_e15_protocol.json`; `bash scripts/reproduce.sh frontier` | `reference_results/alpha_frontier/` |
| Finite nonnegative small-graph oracle | `configs/fedbcert_e25_exact_cone_oracle_protocol.json`; `bash scripts/run_fedbcert_e25_exact_cone_oracle.sh --output-root /tmp/fedbcert-exact-new` | `reference_results/exact_worst_case/` |
| Count-bound granularity | `configs/fedbcert_e27_ledger_granularity_protocol.json`; `bash scripts/reproduce.sh ledger` | `reference_results/ledger_granularity/` |
| Synthetic P / I / MLP comparison | `configs/fedbcert_e28b_confirmation_protocol.json`; portable retraining command below | `reference_results/e28b_confirmation/` |
| Frozen certificate-only timing | `provenance/gap_source_scripts/run_fedbcert_g2_cost_20261005.py` (archive-dependent) | `reference_results/gaps/g2/` |
| On-demand complete-graph fallback | `configs/fedbcert_on_demand_20261015.json`; original `scripts/run_fedbcert_on_demand_20261015.py` (archive-dependent) | `results_boundary_certified/fedbcert_on_demand_20261015/pilot/` |
| Teacher / auxiliary-classifier extension | `configs/fedbcert_e24_invariant_cover_protocol.json`; `bash scripts/reproduce.sh safe-cover` | `reference_results/safe_cover/` |
| Known-endpoint solver cross-check | `configs/fedbcert_e26_known_universe_verifier_bridge.json`; `bash scripts/reproduce.sh gnnev` | `reference_results/gnnev_bridge/` |
| ogbn-arxiv scalability | `configs/fedbcert_e18c_ogb_cone_scale_protocol.json`; `bash scripts/reproduce.sh ogb` | `reference_results/ogbn_arxiv/` |

Experiment commands may download public datasets, train models and write large
outputs. Read the relevant protocol before running them. `reproduce.sh smoke`
includes a Cora download and short training; it is not the offline quick check.
GNNev needs an external solver and license; the teacher comparison needs the
pinned FedLap+ repository. See [DEPENDENCIES.md](../DEPENDENCIES.md).

## Optional synthetic historical retraining

```bash
python scripts/replay_fedbcert_e28b.py --full-replay \
  --output /tmp/fedbcert-historical-replay-new
```

The wrapper generates the fixed synthetic protocol: eight whole-graph seeds,
twelve grid conditions and three models, totaling 288 training runs. Every
selected checkpoint is fixed before test evaluation. Outputs are written to a
new directory; even an existing empty directory is rejected. Published seeds
are reused, so this is historical reproduction, not independent confirmation.

This release has not rerun that training budget. Reproducing the original
checkpoint bytes or original hardware timings is not promised. Do not substitute
regenerated graphs for missing archived inputs while describing a run as an
exact replay of the saved models.

## Missing inputs and interpretation

- Benchmark datasets are not bundled. Saved split/score information does not
  establish bitwise identity with every historical input; the paper discloses
  the reconstructed exposure-analysis masks separately.
- Original synthetic graph bundles and selected model states used for the
  on-demand replay are not included. Its per-node diagnostic tensors are included,
  but cannot replace those inputs. The original audit/runner entry points will
  not run to completion from this compact repository alone.
- Certificate-only timing measurements can be summarized from saved records;
  timing the original checkpoints again needs the omitted payloads. The OGB
  timing does not isolate certification cost.
- The planned RobLight comparison was unavailable because required historical
  inputs were not saved. No numerical superiority claim is made for that case.
- A routed auxiliary classifier's certificate is not a guarantee that the raw
  teacher remains invariant on every completion.

Source, count-bound, model and feature assumptions must hold for a certificate.
Observed zero violations in finite checks do not establish strict floating-point
soundness or prediction correctness.
