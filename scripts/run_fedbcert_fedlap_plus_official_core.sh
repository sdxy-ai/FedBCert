#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FEDLAP_ROOT="${FEDLAP_ROOT:-third_party/fedbcert_e8/FedLap}"
BUNDLE_ROOT="${BUNDLE_ROOT:-results_boundary_certified/boundary_certified_spectral_e8_q1/official_bridges/bundles}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_boundary_certified/boundary_certified_spectral_e8_q1/official_bridges/fedlap_plus_core}"

# Frozen paper-supported core: four datasets x two matched E8 memberships x
# three untouched confirmation seeds = 24 official-code matched runs.
# Computers is not in the FedLap paper and is intentionally excluded.
for dataset in Cora CiteSeer PubMed Photo; do
  for partition in metis random; do
    for seed in 21 22 23; do
      "$PYTHON_BIN" scripts/prepare_fedbcert_e8_official_bundle.py \
        --dataset "$dataset" --partition "$partition" \
        --num-clients 3 --seed "$seed" --output-root "$BUNDLE_ROOT"
      "$PYTHON_BIN" scripts/run_fedbcert_fedlap_plus_official.py \
        --dataset "$dataset" --partition "$partition" \
        --num-clients 3 --seed "$seed" \
        --bundle-root "$BUNDLE_ROOT" --fedlap-root "$FEDLAP_ROOT" \
        --output-root "$OUTPUT_ROOT"
    done
  done
done

"$PYTHON_BIN" scripts/analyze_fedbcert_fedlap_plus_official.py \
  --result-root "$OUTPUT_ROOT" --require-complete
