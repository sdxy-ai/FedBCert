# Artifact manifest

Final editorial release: 2026-10-05. Only package documentation and the derived
payload hash index change relative to the portable claim-closure artifact.
Scientific implementation, frozen protocols and reference outputs do not.
The manuscript clarifies completion domains, cohort denominators and the
scope of historical timing; it does not alter any reported experiment.

This repository is the executable code artifact accompanying the FedBCert
submission. It contains the implementation, frozen protocol files, experiment
and analysis scripts, tests, and compact copies of the reported result tables
and figures.

## Included

- The FedBCert certificate implementation and the model/federated utilities
  needed by the reported experiments.
- Frozen JSON manifests for the utility--certificate frontier, independent
  confirmation, ledger-granularity study, exact finite-completion evaluation,
  GNNev cross-check, ogbn-arxiv scale audit, and FedLap+ selective cover.
- Reproduction, analysis, and plotting scripts for those protocols.
- Unit tests and an offline synthetic certificate check.
- Compact CSV, JSON, PDF, and PNG outputs under `reference_results/`, with
  SHA-256 digests in `reference_results/SHA256SUMS`.

## Generated or downloaded on demand

- Public benchmark datasets are downloaded into `data/fedbcert/` by the data
  preparation and experiment scripts.
- Raw run directories and model checkpoints are regenerated under
  `results_boundary_certified/`; they are excluded from version control because
  of their size.
- The pinned FedLap+ and GNNev repositories are fetched into `third_party/` by
  their setup scripts and remain subject to their upstream licenses.
- GNNev additionally requires a reviewer-provided Gurobi installation and
  license.

No manuscript source, author information, local machine path, private dataset,
or license credential is included in this repository.
