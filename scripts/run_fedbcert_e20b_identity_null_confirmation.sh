#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results_boundary_certified}"
ROOT="boundary_certified_spectral_e20_identity_null"
POLICY="${E20_POLICY:-$RESULT_ROOT/$ROOT/analysis/frozen_identity_null_policy.json}"
BASE_EXP="$ROOT/confirmation"

"$PYTHON_BIN" scripts/analyze_fedbcert_e20a_identity_null.py --require-complete
read -r decision alpha <<<"$("$PYTHON_BIN" -c "import json; p=json.load(open('$POLICY')); print(p['decision'], p['selected_alpha'])")"
if [[ "$decision" != "proceed_to_e20b" || "$alpha" == "None" ]]; then
  echo "E20b not authorized by the frozen E20a decision: $decision $alpha" >&2
  exit 2
fi

PYTHONPATH=. "$PYTHON_BIN" scripts/check_fedbcert_e20_identity_null.py
read -r -a DATASETS <<<"${E20B_DATASETS:-Cora CiteSeer PubMed Computers Photo CS}"
read -r -a CLIENTS <<<"${E20B_CLIENTS:-3 10}"
read -r -a PARTITIONS <<<"${E20B_PARTITIONS:-metis random}"
read -r -a SEEDS <<<"${E20B_SEEDS:-87 88 89}"

for shift_mode in observed identity; do
  variant="${shift_mode}_selected_alpha"
  for dataset in "${DATASETS[@]}"; do
    for clients in "${CLIENTS[@]}"; do
      for partition in "${PARTITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          "$PYTHON_BIN" scripts/run_boundary_certified_spectral_e1.py \
            --exp-name "$BASE_EXP/$variant" \
            --protocol-name "FedBCert E20b frozen identity-null independent confirmation" \
            --result-root "$RESULT_ROOT" \
            --dataset "$dataset" --partition "$partition" \
            --num-clients "$clients" --seed "$seed" \
            --robust-lambda 0.1 --robust-mode adaptive_projected --suite fedavg_only \
            --projection-ratio-cap 0.25 --boundary-activation-threshold 0.20 \
            --boundary-budget-source local_stub_ledger \
            --boundary-aggregation-mode pairwise_masked \
            --lambda-source "unused: E20a-frozen standard FedAvg structural residual" \
            --rounds 100 --local-epochs 1 --hidden-dim 64 \
            --encoder-type dual_feature_identity_null_spectral --encoder-depth 2 \
            --encoder-residual-scale 0.5 --spectral-fusion-scale "$alpha" \
            --spectral-shift-mode "$shift_mode" \
            --uncertainty-recurrence retained_mass \
            --certificate-margin-mode pairwise_difference \
            --include-cone-certificate-audit 1 \
            --order-count 3 --dropout 0.2 --lr 0.01 --weight-decay 0.0005 \
            --certificate-margin 0.05 --evaluation-scope full \
            --force "${E20B_FORCE:-0}"
        done
      done
    done
  done
done

"$PYTHON_BIN" scripts/analyze_fedbcert_e20b_identity_null.py --require-complete
"$PYTHON_BIN" scripts/plot_fedbcert_e20.py
