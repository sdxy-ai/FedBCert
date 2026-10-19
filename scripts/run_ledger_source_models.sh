#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results_boundary_certified}"
if [[ "$RESULT_ROOT" != "results_boundary_certified" ]]; then
  echo "The frozen E27 manifest refers to results_boundary_certified/." >&2
  echo "Unset RESULT_ROOT when reproducing the paper's ledger experiment." >&2
  exit 2
fi
BUNDLE_ROOT="$RESULT_ROOT/boundary_certified_spectral_e8_q1/official_bridges/bundles"
BASE_EXP="boundary_certified_spectral_e10c/e8_bundle_paired"
read -r -a DATASETS <<<"${LEDGER_DATASETS:-Cora CiteSeer PubMed Computers Photo}"
read -r -a PARTITIONS <<<"${LEDGER_PARTITIONS:-metis random}"
read -r -a SEEDS <<<"${LEDGER_SEEDS:-21 22 23}"

variant_for_dataset() {
  case "$1" in
    Computers|Photo) printf '%s\n' "dual_h64_d2_s05_sep025 dual_feature_spectral 2" ;;
    Cora|CiteSeer|PubMed) printf '%s\n' "e8_linear_h64_d1 linear_relu 1" ;;
    *) echo "unsupported ledger dataset: $1" >&2; return 2 ;;
  esac
}

lambda_for_dataset() {
  case "$1" in
    CiteSeer) printf '%s\n' 1.0 ;;
    Cora|PubMed|Computers|Photo) printf '%s\n' 0.1 ;;
    *) return 2 ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  read -r variant encoder_type encoder_depth <<<"$(variant_for_dataset "$dataset")"
  robust_lambda="$(lambda_for_dataset "$dataset")"
  for partition in "${PARTITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="${dataset}_${partition}_c3_seed${seed}"
      bundle_dir="$BUNDLE_ROOT/$run_name"
      "$PYTHON_BIN" scripts/prepare_fedbcert_e8_official_bundle.py \
        --dataset "$dataset" --partition "$partition" --num-clients 3 \
        --seed "$seed" --output-root "$BUNDLE_ROOT"
      "$PYTHON_BIN" scripts/run_boundary_certified_spectral_e1.py \
        --exp-name "$BASE_EXP/$variant" \
        --protocol-name "FedBCert frozen source model for ledger audit" \
        --result-root "$RESULT_ROOT" \
        --dataset "$dataset" --partition "$partition" \
        --num-clients 3 --seed "$seed" \
        --immutable-split-bundle "$bundle_dir/split_bundle.pt" \
        --immutable-split-manifest "$bundle_dir/manifest.json" \
        --robust-lambda "$robust_lambda" \
        --robust-mode adaptive_projected --suite pair \
        --projection-ratio-cap 0.25 \
        --boundary-activation-threshold 0.20 \
        --boundary-budget-source local_stub_ledger \
        --boundary-aggregation-mode pairwise_masked \
        --lambda-source "frozen source model; no ledger-mode selection" \
        --rounds 100 --local-epochs 1 --hidden-dim 64 \
        --encoder-type "$encoder_type" --encoder-depth "$encoder_depth" \
        --encoder-residual-scale 0.5 --order-count 3 --dropout 0.2 \
        --lr 0.01 --weight-decay 0.0005 --certificate-margin 0.05 \
        --evaluation-scope full --force "${LEDGER_FORCE:-0}"
    done
  done
done
