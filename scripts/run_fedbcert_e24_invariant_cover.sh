#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="results_boundary_certified/boundary_certified_spectral_e24_invariant_cover"
read -r -a DATASETS <<<"${E24_DATASETS:-Cora CiteSeer PubMed Photo}"
read -r -a PARTITIONS <<<"${E24_PARTITIONS:-metis random}"
read -r -a SEEDS <<<"${E24_SEEDS:-128 129 130}"
COMPONENTS=(feature_ce feature_kd_cad spectral_identity_cad spectral_cad)
FORCE_ARGS=()
if [[ "${E24_FORCE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

for dataset in "${DATASETS[@]}"; do
  for partition in "${PARTITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      "$PYTHON_BIN" scripts/prepare_fedbcert_e8_official_bundle.py \
        --dataset "$dataset" --partition "$partition" --num-clients 3 --seed "$seed" \
        --output-root "$ROOT/bundles"
      "$PYTHON_BIN" scripts/run_fedbcert_fedlap_plus_official.py \
        --dataset "$dataset" --partition "$partition" --num-clients 3 --seed "$seed" \
        --bundle-root "$ROOT/bundles" --output-root "$ROOT/teachers" \
        --export-aligned-logits --skip-test-evaluation
      for component in "${COMPONENTS[@]}"; do
        "$PYTHON_BIN" scripts/run_fedbcert_e22_teacher_shield.py \
          --dataset "$dataset" --partition "$partition" --num-clients 3 --seed "$seed" \
          --variant "$component" --bundle-root "$ROOT/bundles" \
          --teacher-root "$ROOT/teachers" --output-root "$ROOT/components" \
          --protocol-name "FedBCert E24 frozen invariant-cover component" \
          --evaluation-scope validation_only "${FORCE_ARGS[@]}"
      done
      "$PYTHON_BIN" scripts/run_fedbcert_e24_invariant_cover.py \
        --dataset "$dataset" --partition "$partition" --num-clients 3 --seed "$seed" \
        --bundle-root "$ROOT/bundles" --teacher-root "$ROOT/teachers" \
        --component-root "$ROOT/components" --output-root "$ROOT/evaluation" \
        "${FORCE_ARGS[@]}"
    done
  done
done

"$PYTHON_BIN" scripts/analyze_fedbcert_e24.py --require-complete
