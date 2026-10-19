#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results_boundary_certified}"
BASE_EXP="boundary_certified_spectral_e15/confirmation"
read -r -a DATASETS <<<"${E15_DATASETS:-Cora CiteSeer PubMed Computers Photo CS}"
read -r -a CLIENTS <<<"${E15_CLIENTS:-3 10}"
read -r -a PARTITIONS <<<"${E15_PARTITIONS:-metis random}"
read -r -a SEEDS <<<"${E15_SEEDS:-50 51 52}"
read -r -a VARIANTS <<<"${E15_VARIANTS:-feature_alpha0 dual_alpha010 dual_alpha025 dual_alpha050 dual_alpha100}"

variant_arguments() {
  case "$1" in
    feature_alpha0) printf '%s\n' "feature_only_mlp 2 0.0" ;;
    dual_alpha010) printf '%s\n' "dual_feature_spectral 2 0.1" ;;
    dual_alpha025) printf '%s\n' "dual_feature_spectral 2 0.25" ;;
    dual_alpha050) printf '%s\n' "dual_feature_spectral 2 0.5" ;;
    dual_alpha100) printf '%s\n' "dual_feature_spectral 2 1.0" ;;
    *) echo "unknown E15 variant: $1" >&2; return 2 ;;
  esac
}

for variant in "${VARIANTS[@]}"; do
  read -r encoder_type encoder_depth alpha <<<"$(variant_arguments "$variant")"
  for dataset in "${DATASETS[@]}"; do
    for clients in "${CLIENTS[@]}"; do
      for partition in "${PARTITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          "$PYTHON_BIN" scripts/run_boundary_certified_spectral_e1.py \
            --exp-name "$BASE_EXP/$variant" \
            --protocol-name "FedBCert E15 fixed utility-certificate Pareto independent confirmation" \
            --result-root "$RESULT_ROOT" \
            --dataset "$dataset" --partition "$partition" \
            --num-clients "$clients" --seed "$seed" \
            --robust-lambda 0.1 --robust-mode adaptive_projected \
            --suite fedavg_only \
            --projection-ratio-cap 0.25 \
            --boundary-activation-threshold 0.20 \
            --boundary-budget-source local_stub_ledger \
            --boundary-aggregation-mode pairwise_masked \
            --lambda-source "unused in E15 standard-FedAvg-only confirmation" \
            --rounds 100 --local-epochs 1 \
            --hidden-dim 64 \
            --encoder-type "$encoder_type" --encoder-depth "$encoder_depth" \
            --encoder-residual-scale 0.5 \
            --spectral-fusion-scale "$alpha" \
            --uncertainty-recurrence retained_mass \
            --certificate-margin-mode pairwise_difference \
            --order-count 3 --dropout 0.2 \
            --lr 0.01 --weight-decay 0.0005 --certificate-margin 0.05 \
            --evaluation-scope full --force "${E15_FORCE:-0}"
        done
      done
    done
  done
done

if [[ "${E15_SKIP_FINAL_ANALYSIS:-0}" != "1" ]]; then
  E15_ROOT="$RESULT_ROOT/boundary_certified_spectral_e15"
  "$PYTHON_BIN" scripts/reaudit_fedbcert_e15_numerics.py --root "$E15_ROOT"
  "$PYTHON_BIN" scripts/analyze_fedbcert_e15.py --root "$E15_ROOT" --require-complete
  "$PYTHON_BIN" scripts/plot_fedbcert_e15.py --root "$E15_ROOT"
fi
