# Reproducing and checking results

This repository separates three tasks: executing the verifier on toy inputs,
checking included saved records, and rerunning research experiments. The first
two are the default review path. They do not imply that missing historical
models or original hardware timings have been reproduced.

## Offline review

Install the CPU environment from the root README and run commands there.
All paths below are relative to the repository root.

| Command | Scope |
| --- | --- |
| `python scripts/smoke_certificate.py` | A small offline graph; no training |
| `python scripts/demo_fedbcert_deployment.py --output /tmp/fedbcert-demo-new.json` | Frozen inference and status examples; output must be new |
| `python scripts/verify_rg1_reference.py` | RG-1.0 case/measurement integrity and frozen paired statistics |
| `python scripts/verify_review_artifact.py` | Release hashes and historical synthetic paired statistics |
| `python scripts/verify_reference_results.py` | Historical benchmark and finite-oracle headline values |
| `python scripts/verify_gap_reference_results.py` | Saved certificate timing and external-verifier availability records |
| `python scripts/verify_on_demand_reference.py` | All 240 historical synthetic on-demand records and paired summaries |
| `python -m pytest -q -m 'not optional and not archive'` | Portable suite, excluding external solvers and missing archives |

For a short implementation review:

```bash
python -m pytest -q tests/test_fedbcert_b1_fusion_contract.py \
  tests/test_fedbcert_b2_snapshot_contract.py tests/test_fedbcert_on_demand.py \
  tests/test_fedbcert_rg1_evaluation.py
```

The B1/B2 tests check fusion semantics, fingerprints, cache invalidation and
rejection of unsupported changes. Toy on-demand tests cover signed polynomial
coefficients, complete normalization, remote-to-remote dependencies, message
serialization and equality to a centralized reference. The RG-1.0 toy
evaluation covers L/F/E/S/D/N without formal data or training. See
[VALIDATION.md](VALIDATION.md) for actual release test results.

## RG-1.0 saved records

The scientific sources correspond to commit
`e99878f48ef441346011f27e5bdd63df3e568f63`. RG-1.0 used Cora and CiteSeer,
METIS and balanced random partitions, and eight seeds 42020101–42020108:
32 paired cases and 96 P/I/MLP models, each trained for 100 rounds. All selected
checkpoints were sealed before formal evaluation. The three certificates use
the same frozen P predictor within each case.

The [record guide](../rg1_data/README.md) describes all files and units:

- `case_metrics.json` contains all 32 cases, utility, structural sensitivity
  and certificate coverage.
- `all_records.json` contains 1,536 formal repeated measurements, including
  controls. Timing repeats, nodes and clients are not independent units.
- `analysis.json` retains the frozen dataset-stratified paired bootstrap with
  10,000 draws and RNG seed 42028616.
- CSV files expose case, workload, strategy and group comparisons, including
  negative effects and relative and absolute payload summaries.

Within each case, workload and strategy, CPU time is the median of three formal
repeats. Bytes and counts must agree across repeats. Ratios or reductions are
computed within cases before equal-weight aggregation. A mean of relative
reductions is not the reduction computed from mean byte counts. The 32-node
workload has a mean paired reduction of −281.83% but a mean absolute N−F byte
difference near −103,995 B: eight CiteSeer/METIS cases have small F denominators
and increased N bytes, whereas the other 24 cases decrease bytes. No cases are
dropped. Descriptive fixed-graph intervals are not guarantees over independently
sampled real graphs.

### Full RG-1.0 rerun: archive-dependent

The frozen programs `scripts/run_fedbcert_rg1.py`,
`fedbcert_rg1_common.py`, `fedbcert_rg1_training.py`,
`fedbcert_rg1_evaluation.py` and `fedbcert_rg1_analysis.py` are included.
**Their formal configure/prepare/train/evaluate pipeline is not standalone
from this compact repository.** It expects the original parent-project layout,
preflight/data archives, environment metadata, source-helper/protocol paths
and approved run manifests. Its manifests bind paths, source hashes and inputs.

Those machine-specific records, the raw data cache and all 96 checkpoints are
not bundled. Original integrity/approval gates have not been bypassed or
relaxed. Do not run the formal runner expecting a successful independent
96-model reproduction from this checkout. Similar regenerated inputs do not
establish identity with the original run. Offline saved-record verification
and toy inference are the supported independent review paths.

## Historical experiment map

These entry points are not quick checks. They may download datasets, train
models, require optional repositories, or depend on omitted input archives.
Read the fixed protocol before running one; use a new output directory.

| Paper component | Protocol / entry point | Included evidence |
| --- | --- | --- |
| Six-dataset same-predictor comparison | `configs/fedbcert_e18_cone_protocol.json`; `bash scripts/reproduce.sh confirmation` | `reference_results/main_confirmation/` |
| Graph-weight trade-off and ablations | `configs/fedbcert_e15_protocol.json`; `bash scripts/reproduce.sh frontier` | `reference_results/alpha_frontier/` |
| Finite nonnegative small-graph oracle | `configs/fedbcert_e25_exact_cone_oracle_protocol.json`; `bash scripts/run_fedbcert_e25_exact_cone_oracle.sh --output-root /tmp/fedbcert-exact-new` | `reference_results/exact_worst_case/` |
| Count-bound granularity | `configs/fedbcert_e27_ledger_granularity_protocol.json`; `bash scripts/reproduce.sh ledger` | `reference_results/ledger_granularity/` |
| Synthetic P/I/MLP comparison | `configs/fedbcert_e28b_confirmation_protocol.json`; command below | `reference_results/e28b_confirmation/` |
| Certificate-only timing | Historical measurement source; original inputs required | `reference_results/gaps/g2/` |
| Synthetic on-demand replay | `configs/fedbcert_on_demand_20261015.json`; `scripts/run_fedbcert_on_demand_20261015.py` (archive-dependent) | `results_boundary_certified/fedbcert_on_demand_20261015/pilot/` |
| Fixed-route teacher extension | `configs/fedbcert_e24_invariant_cover_protocol.json`; `bash scripts/reproduce.sh safe-cover` | `reference_results/safe_cover/` |
| Known-endpoint solver cross-check | `configs/fedbcert_e26_known_universe_verifier_bridge.json`; `bash scripts/reproduce.sh gnnev` | `reference_results/gnnev_bridge/` |
| ogbn-arxiv scalability | `configs/fedbcert_e18c_ogb_cone_scale_protocol.json`; `bash scripts/reproduce.sh ogb` | `reference_results/ogbn_arxiv/` |

`reproduce.sh smoke` downloads Cora and performs short training; it is not the
offline `smoke_certificate.py` command. The solver cross-check needs Gurobi and
a license. The teacher comparison needs the pinned FedLap+ repository. See
[DEPENDENCIES.md](../DEPENDENCIES.md).

### Optional historical synthetic retraining

```bash
python scripts/replay_fedbcert_e28b.py --full-replay \
  --output /tmp/fedbcert-historical-replay-new
```

This runs **96 generated graphs and 288 full 100-round model trainings** across
twelve conditions and eight whole-graph seeds. The output directory must not
exist, even as an empty directory. Every selected checkpoint is fixed before
test evaluation. Published seeds are reused; this is historical reproduction,
not new independent confirmation. It does not establish reproduction of the
original checkpoint bytes or hardware timings. None of the release quick
checks runs this budget.

## Historical timing versions and missing inputs

Included original synthetic on-demand records report 22.09% paired payload
reduction and about 23.8% additional CPU time. A separate retained-input replay
after the contract fixes reported about 22.43% additional CPU time with
unchanged message counts. That newer timing payload is not bundled here.
These noncontemporaneous measurements are not interchangeable results or a
causal speed comparison. RG-1.0 is a separate real-graph experiment.

The historical reference checker recomputes statistics without loading omitted
graph/model tensors. The original analyzer checks 137 archived input files and
refuses to overwrite an analysis directory; it cannot complete from result
records alone. Per-node diagnostics do not replace those inputs. Similarly,
certificate-only timing can be summarized from records, but remeasuring the
original models needs omitted payloads. Historical OGB timing does not isolate
certification cost.

Benchmark data are not bundled. Reconstructed historical split/score information
does not establish bitwise identity with every original input. The planned
RobLight comparison could not run because historical inputs were missing;
no numerical superiority claim follows. A routed auxiliary classifier's
certificate is not a guarantee for the raw teacher on every completion.

Valid count bounds and supported model/representation assumptions remain
prerequisites. Finite observed zero violations and the numerical acceptance
guard do not establish strict floating-point soundness or prediction
correctness. Application-message bytes do not measure real network delay,
security overhead or deployment cost.
