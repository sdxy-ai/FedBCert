#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results_boundary_certified}"
POLICY="${E18_POLICY:-results_boundary_certified/boundary_certified_spectral_e18_cone/analysis/frozen_cone_policy.json}"
BASE_EXP="boundary_certified_spectral_e18_cone/confirmation"

"$PYTHON_BIN" scripts/analyze_fedbcert_e18a_cone.py --require-complete
read -r decision selected <<<"$("$PYTHON_BIN" -c "import json; p=json.load(open('$POLICY')); print(p['decision'], p['selected_variant'])")"
if [[ "$decision" != "proceed_to_e18b" || "$selected" == "None" ]]; then
  echo "E18b not authorized by the frozen E18a decision: $decision $selected" >&2
  exit 2
fi

variant_arguments() {
  case "$1" in
    feature_alpha0) printf '%s\n' "feature_only_mlp 2 0.0" ;;
    dual_alpha010) printf '%s\n' "dual_feature_spectral 2 0.1" ;;
    dual_alpha025) printf '%s\n' "dual_feature_spectral 2 0.25" ;;
    dual_alpha050) printf '%s\n' "dual_feature_spectral 2 0.5" ;;
    dual_alpha100) printf '%s\n' "dual_feature_spectral 2 1.0" ;;
    *) echo "unknown E18 variant: $1" >&2; return 2 ;;
  esac
}

read -r -a DATASETS <<<"${E18B_DATASETS:-Cora CiteSeer PubMed Computers Photo CS}"
read -r -a CLIENTS <<<"${E18B_CLIENTS:-3 10}"
read -r -a PARTITIONS <<<"${E18B_PARTITIONS:-metis random}"
read -r -a SEEDS <<<"${E18B_SEEDS:-73 74 75}"

for variant in feature_alpha0 "$selected"; do
  read -r encoder_type encoder_depth alpha <<<"$(variant_arguments "$variant")"
  for dataset in "${DATASETS[@]}"; do
    for clients in "${CLIENTS[@]}"; do
      for partition in "${PARTITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          "$PYTHON_BIN" scripts/run_boundary_certified_spectral_e1.py \
            --exp-name "$BASE_EXP/$variant" \
            --protocol-name "FedBCert E18b frozen ReLU-cone independent confirmation" \
            --result-root "$RESULT_ROOT" \
            --dataset "$dataset" --partition "$partition" \
            --num-clients "$clients" --seed "$seed" \
            --robust-lambda 0.1 --robust-mode adaptive_projected --suite fedavg_only \
            --projection-ratio-cap 0.25 --boundary-activation-threshold 0.20 \
            --boundary-budget-source local_stub_ledger \
            --boundary-aggregation-mode pairwise_masked \
            --lambda-source "unused: E18a-frozen standard FedAvg predictor $variant" \
            --rounds 100 --local-epochs 1 --hidden-dim 64 \
            --encoder-type "$encoder_type" --encoder-depth "$encoder_depth" \
            --encoder-residual-scale 0.5 --spectral-fusion-scale "$alpha" \
            --uncertainty-recurrence retained_mass \
            --certificate-margin-mode pairwise_difference \
            --include-cone-certificate-audit 1 \
            --order-count 3 --dropout 0.2 --lr 0.01 --weight-decay 0.0005 \
            --certificate-margin 0.05 --evaluation-scope full \
            --force "${E18B_FORCE:-0}"
        done
      done
    done
  done
done

"$PYTHON_BIN" scripts/analyze_fedbcert_e18b_cone.py --require-complete
