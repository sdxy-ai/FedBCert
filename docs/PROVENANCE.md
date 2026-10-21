# Source provenance

The current scientific sources match revision
`e99878f48ef441346011f27e5bdd63df3e568f63` from October 20, 2026. This includes
the B1 fusion-contract fix, B2 snapshot-contract fix, shared model contract,
and the RG-1.0 training, evaluation and analysis programs.
`FROZEN_SOURCE_SHA256.json` records the original file hashes.

The earlier public release used two October 17 manuscript archives:

- `FedBCert_Anonymous_Code_20261013.zip`: model, certificate, federation,
  deployment, fixed protocols, portable review tools and compact results.
- `FedBCert_OnDemand_Supplement.zip`: on-demand aggregation, geometric baselines,
  tests and saved application records.

The earlier source pins and measurements remain unchanged. The current runtime
includes the subsequently verified B1/B2 corrections; it is not claimed to be
byte-identical to the earlier runtime. `HISTORICAL_SOURCE_LOCATIONS.json` points
to archived copies of earlier certificate/deployment/on-demand files for old record
checks. Public helper adaptations resolve these historical pins to those
copies; they do not alter the saved measurements or rewrite old hashes.

The root README, NOTICE, packaging metadata and release index are updated for
public installation. Original documentation and indexes are preserved under
`provenance/submission_20261017/`. The original supplement index describes its
archive, not this combined layout. `PACKAGE_SHA256.json` is the current release
index and should not be confused with an original research freeze.

Tests that require unpublished manuscript snapshots are kept under
`provenance/workspace_tests_not_portable/`, outside default collection. This does
not remove a failing mathematical test: the moved test checks editorial text and
table formatting against two absent Chinese manuscript files.

## Result versions and public checks

The original synthetic replay measured 22.09% paired payload reduction and about
23.8% additional CPU time. The later retained-input B1/B2 replay measured about
22.43% additional CPU time with unchanged message counts. This does not replace
the original timing or establish a causal effect of the fix. RG-1.0 is a
separate fixed-real-graph experiment and is not pooled with the synthetic data.

`rg1_data/` preserves all 32 real-graph cases and 1,536 formal records, including
negative outcomes. `verify_rg1_reference.py` calls the frozen analysis on saved
JSON, without models, inference or new measurements. The original bodies of the
three adapted historical record helpers remain under `provenance/`.

The full RG-1.0 runner still needs the original parent-project preflight,
environment and authorization archives. Its sources are included for review;
they are not advertised as a standalone 96-model rerun. Tests marked `archive`
need those omitted inputs; tests marked `optional` need external software.
See [RG1_ARCHIVED_RUNNER.md](RG1_ARCHIVED_RUNNER.md).

OpenFGL attribution and its original MIT license are preserved. Optional external
code and datasets are obtained from their official sources and retain their own
license terms.
