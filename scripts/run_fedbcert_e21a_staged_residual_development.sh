#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results_boundary_certified}"
BASE_EXP="boundary_certified_spectral_e21_staged_residual/development"
read -r -a DATASETS <<<"${E21A_DATASETS:-Cora CiteSeer PubMed Computers Photo CS}"
read -r -a SEEDS <<<"${E21A_SEEDS:-101 102}"
read -r -a ALPHAS <<<"${E21A_ALPHAS:-0.1 0.25 0.5}"

PYTHONPATH=. "$PYTHON_BIN" scripts/check_fedbcert_e20_identity_null.py
PYTHONPATH=. "$PYTHON_BIN" scripts/check_fedbcert_e21_staged_residual.py

alpha_label() {
  "$PYTHON_BIN" -c "print(str(float('$1')).replace('.', 'p'))"
}

for alpha in "${ALPHAS[@]}"; do
  label="alpha$(alpha_label "$alpha")"
  for shift_mode in observed identity; do
    variant="${shift_mode}_${label}"
    for dataset in "${DATASETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        "$PYTHON_BIN" scripts/run_boundary_certified_spectral_e1.py \
          --exp-name "$BASE_EXP/$variant" \
          --protocol-name "FedBCert E21a staged residual validation-only development" \
          --result-root "$RESULT_ROOT" \
          --dataset "$dataset" --partition random --num-clients 3 --seed "$seed" \
          --robust-lambda 0.1 --robust-mode adaptive_projected --suite fedavg_only \
          --projection-ratio-cap 0.25 --boundary-activation-threshold 0.20 \
          --boundary-budget-source local_stub_ledger \
          --boundary-aggregation-mode pairwise_masked \
          --lambda-source "unused: staged standard FedAvg structural correction" \
          --rounds 150 --feature-pretrain-rounds 100 \
          --identity-null-training-mode staged_residual \
          --local-epochs 1 --hidden-dim 64 \
          --encoder-type dual_feature_identity_null_spectral --encoder-depth 2 \
          --encoder-residual-scale 0.5 --spectral-fusion-scale "$alpha" \
          --spectral-shift-mode "$shift_mode" \
          --uncertainty-recurrence retained_mass \
          --certificate-margin-mode pairwise_difference \
          --include-cone-certificate-audit 1 \
          --order-count 3 --dropout 0.2 --lr 0.01 --weight-decay 0.0005 \
          --certificate-margin 0.05 --evaluation-scope validation_only \
          --force "${E21A_FORCE:-0}"
      done
    done
  done
done

"$PYTHON_BIN" scripts/analyze_fedbcert_e21a_staged_residual.py --require-complete
