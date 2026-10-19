# Source provenance

The release uses two archives from the October 17, 2026 manuscript package:

- `FedBCert_Anonymous_Code_20261013.zip`: model, certificate, federation,
  deployment, fixed protocols, portable review tools and compact results.
- `FedBCert_OnDemand_Supplement.zip`: on-demand aggregation, geometric baselines,
  tests and saved application records.

The three overlapping source files in these archives were byte-identical. No
mathematical implementation, training rule, fixed protocol or saved experiment
value was changed to prepare this public repository. Source pins in the
historical manifests are retained. New code provides a saved-record checker and
release validation, not a new experiment or a different certificate.

The root README, NOTICE, packaging metadata and release index are updated for
public installation. Original documentation and indexes are preserved under
`provenance/submission_20261017/`. The original supplement index describes its
archive, not this combined layout. `PACKAGE_SHA256.json` is the current release
index and should not be confused with an original research freeze.

Tests that require unpublished manuscript snapshots are kept under
`provenance/workspace_tests_not_portable/`, outside default collection. This does
not remove a failing mathematical test: the moved test checks editorial text and
table formatting against two absent Chinese manuscript files.

## Development disclosure

OpenAI ChatGPT/Codex assisted English documentation, code organization,
refactoring and consistency checks during development and release preparation.
This assistance is not authorship or a substitute for scientific validation.
The manuscript contains the corresponding preparation disclosure. Frozen source
and result records remain available for inspection rather than being rewritten
to hide their development history.

OpenFGL attribution and its original MIT license are preserved. Optional external
code and datasets are obtained from their official sources and retain their own
license terms.
