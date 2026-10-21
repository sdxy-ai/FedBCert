# Release contents

Release 1.2.0 adds the verified B1/B2 corrections and RG-1.0 source and records.
Scientific model, training, evaluation and analysis files match commit
`e99878f48ef441346011f27e5bdd63df3e568f63`. Public packaging and record-check
helper adaptations are documented separately in `docs/PROVENANCE.md`.

`PACKAGE_SHA256.json` describes this release. Original documentation and indexes
are retained under `provenance/submission_20261017/`; those indexes describe the
older archives, not the new layout. `HISTORICAL_SOURCE_SHA256.json` retains the
original scientific-source pins. `HISTORICAL_SOURCE_LOCATIONS.json` locates
the archived pre-fix source copies. `FROZEN_SOURCE_SHA256.json` pins the current
scientific implementation. `rg1_data/` contains all 32 real-graph cases and
1,536 formal measurements, without changing their values.

One manuscript-only regression test is preserved under
`provenance/workspace_tests_not_portable/`. It requires two Chinese manuscript
snapshots not distributed with the code and is not part of the portable suite.
Other workspace-only tests were already archived in that directory.

The on-demand diagnostic tensors are not original graph bundles or trained
checkpoints. Those larger archives, downloaded datasets, optional third-party
checkouts and new training outputs are not included. The saved on-demand result
directory is an explicit exception to the generated-results ignore rule.
No manuscript PDF or author title page is included.
