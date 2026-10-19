# Anonymous FedBCert artifact: final editorial submission (2026-10-05)

This release accompanies the optimized final manuscript. It changes only this
README and `ARTIFACT_MANIFEST.md` from the portable claim-closure package;
implementation, protocol and reference-result bytes are preserved. No new
experiment, training, timing replay or post-hoc tuning is part of this release.
`PACKAGE_SHA256.json` identifies the exact portable payload. The submission
bundle supplies the corresponding manuscript separately from this anonymous
code archive.

This package is self-contained for offline verifier tests and compact reference
checks, and preserves optional historical E28b replay tools. It does
not require the original author's workspace or private historical directories.
No benchmark dataset, trained checkpoint, or third-party solver is bundled.
The preserved legacy reproduction scripts may require their external datasets
and optional tools; they are not the entry point for the portable E28b replay.

## Environment and quick checks

Use Linux/WSL, Python 3.9--3.11, PyTorch 2.8.0 and PyG 2.6.1. Install PyTorch
and PyG for your platform, then `pip install -r requirements-no-torch.txt`.
GPU is optional. The frozen engine uses the POSIX `resource` module.

From this package directory:

```bash
python scripts/verify_review_artifact.py
python scripts/verify_reference_results.py
python scripts/verify_gap_reference_results.py
python -m pytest -q -m 'not optional'
```

Only reference/unit checks were executed for this closure publication. No new
experiments, timing replay, smoke training, full historical training, seed search
or post-hoc tuning was performed. The optional smoke entry point uses 120 nodes,
engineering seed 28000, two rounds, CPU, and
validation targets only. It is explicitly not scientific evidence or a repeat
of independent confirmation. Outputs must not exist; existing results are never
overwritten. The main package manifest checks sources and included references.
The legacy reference verifier checks its own preserved `SHA256SUMS`; the new
review verifier additionally checks E28a/E28b results and paired statistics.

## Optional full historical replay

```bash
python scripts/replay_fedbcert_e28b.py --full-replay \
  --output /tmp/fedbcert-historical-replay-new
```

This reuses published seeds 28101--28108: 96 graphs, 288 complete 100-round
trainings, all checkpoints committed before any test evaluation. It does not
constitute new independent confirmation, and must not be used for post-hoc
hyperparameter search. The frozen engine and protocol bytes are preserved.
The wrapper changes only protection scope (included package files rather than
private historic directories) and output location. `replay_manifest.json` and
`replay_summary.json` label the result historical reproduction. Original raw
`phase` strings are retained as legacy storage-schema values and do not override
that label. Full original freeze/audit scripts are included for provenance;
their original entry points require the omitted historical research archive.
Backend/version changes can affect training numerics; bitwise identical GPU
retraining is not promised. Resource/reliability failures are retained and stop
the run without hidden retries, seed replacement, or budget reduction.

The full replay and smoke training were NOT executed for this closure package:
the published independent confirmation was already complete. Reference
statistics, source hashes, extracted-package tests and replay gate logic are
checked without regenerating any confirmation graph. Optional replay tools
remain for explicitly requested reproduction, not as additional experiment
tasks or a reason to reopen the frozen claims.

## Closed gap records (G1/G2)

`reference_results/gaps/g1/gate_audit.json` records an official available
RobLight artifact and a failed independent archival-input gate. All 40 E26
models (160 completions) are pre-execution Unavailable because original
model/features and complete per-completion logits were not archived. Numeric
comparison/translation counts are zero; precision/solved ratio are null, not
zero performance. No external solver ran. This is not evidence of unsupported
models or an external-verifier superiority claim.

`reference_results/gaps/g2/` includes the already completed frozen timing replay,
its execution/start manifests and all eight raw timing JSON records. G2 uses
the eight committed E28b primary-cell models, not new training. The median of
per-case combined certificate times is 18.12 ms; forward is 13.02 ms and the
same-case certificate/forward ratio is 1.391. This describes certificate-only
cost, not an overall 1.391x pipeline or only 39% extra cost. Sampled RSS
increments are neither total memory nor instantaneous peak guarantees. Three
historic OGB cases are Unavailable; no cert-only large-graph overhead is claimed.

`verify_gap_reference_results.py` recomputes all 1,200 included measurements'
statistics and checks case/stage completeness, status semantics, manifest
binding and bundled source hashes. This is a read-only compact-reference
check, not a replay, a full checkpoint audit or strict floating-point proof.
Large label-free payloads, original graph bundles and checkpoints are omitted.
`provenance/gap_source_scripts/` preserves original research scripts byte for
byte for provenance; these are NOT portable entry points and require the
omitted archive. They must not be run as publication quick checks.
Two E26 files retain the prior package's portability edits (reviewer-provided
license wording and configurable external dependency locations). Their
derivative hashes are checked separately, not reported as byte-exact original
research-source pins. The G1 record preserves the original identities.

## Reference data and provenance

`reference_results/e28b_confirmation/` contains the complete descriptive grid,
per-seed paired metrics, primary tests, and full original source/commit/audit
fingerprints. These reference manifests point to omitted large checkpoint and
graph artifacts; checking compact reference hashes is not repeating the full
original archive audit. `reference_results/e28a_exposure/` contains retrospective
aggregate and unit-level results. Its audit input hashes are not prospective
historical E18 input hashes. Missing historical split bundles are disclosed.
`HISTORICAL_SOURCE_SHA256.json` records the source hashes that can be independently
checked in this compact package. `PACKAGE_SHA256.json` covers the package files.

The no-graph MLP and identity controls are structurally invariant. The MLP is a
fixed capacity control, not an optimized representative of all possible MLPs.
Primary MLP inference is noninferiority with margin -1 pp, not a retrospective
superiority hypothesis. Eight whole-graph seeds are the independent units;
nodes, clients, and grid cells sharing a seed are not additional replicates.
Other grid cells are descriptive. Sign tests and bootstrap mean intervals have
different estimands. Existing null graph-attribution results are not reversed.

## Deployment contract

`openfgl/flcore/fedbcert/deployment.py` and its tests demonstrate bounded-ledger
and explicit universal-row modes. Inputs, preprocessing, weights, unit edges,
and snapshot must be fixed; bounds must not underestimate actual counts.
Architecture-level row bounds must apply to all admissible remote inputs;
cone verification needs nonnegativity. Hashes and receipts do not prove ledger
truth. The backend supports CPU float32/float64 and its two bounded ReLU model
families. Certified/Deferred/Unavailable concern an added class-stability
certificate, not correctness or a business abstention rule. A wrong class can
be stable. There is no privacy, poisoning, invariant-logit, or end-to-end safety
guarantee. A selective route certifies the routed safe expert, not a raw teacher
evaluated on every completion.

The package contains the preserved upstream license and third-party notices.
No optional solver license or private dependency is redistributed.
