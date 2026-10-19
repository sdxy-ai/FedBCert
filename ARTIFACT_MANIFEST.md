# Release contents

This release combines the main code archive and on-demand supplement supplied
with the October 17, 2026 manuscript. Scientific sources, fixed protocols and
saved results retain their submitted bytes. Documentation, package metadata,
portable release checks and the package hash index are updated for GitHub.

`PACKAGE_SHA256.json` describes this release. Original documentation and indexes
are retained under `provenance/submission_20261017/`; those indexes describe the
older archives, not the new layout. `HISTORICAL_SOURCE_SHA256.json` retains the
original scientific-source pins.

One manuscript-only regression test is preserved under
`provenance/workspace_tests_not_portable/`. It requires two Chinese manuscript
snapshots not distributed with the code and is not part of the portable suite.
Other workspace-only tests were already archived in that directory.

The on-demand diagnostic tensors are not original graph bundles or trained
checkpoints. Those larger archives, downloaded datasets, optional third-party
checkouts and new training outputs are not included. The saved on-demand result
directory is an explicit exception to the generated-results ignore rule.
No manuscript PDF or author title page is included.
