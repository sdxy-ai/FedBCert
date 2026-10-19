#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results_boundary_certified}"
POLICY="${E18_POLICY:-results_boundary_certified/boundary_certified_spectral_e18_cone/analysis/frozen_cone_policy.json}"
CONFIRMATION="${E18_CONFIRMATION:-results_boundary_certified/boundary_certified_spectral_e18_cone/analysis/confirmation_summary.json}"
BASE_EXP="boundary_certified_spectral_e18c_ogb_cone_scale/runs"

"$PYTHON_BIN" scripts/analyze_fedbcert_e18a_cone.py --require-complete
"$PYTHON_BIN" scripts/analyze_fedbcert_e18b_cone.py --require-complete
read -r policy_decision selected alpha <<<"$("$PYTHON_BIN" -c "import json; p=json.load(open('$POLICY')); print(p['decision'], p['selected_variant'], p['selected_alpha'])")"
confirmation_decision="$("$PYTHON_BIN" -c "import json; print(json.load(open('$CONFIRMATION'))['decision'])")"
if [[ "$policy_decision" != "proceed_to_e18b" || "$selected" != "dual_alpha025" || "$alpha" != "0.25" ]]; then
  echo "E18c requires the frozen dual_alpha025 E18 policy, found: $policy_decision $selected $alpha" >&2
  exit 2
fi
if [[ "$confirmation_decision" != "confirmation_complete_claim_supported" ]]; then
  echo "E18c requires a successful E18b confirmation, found: $confirmation_decision" >&2
  exit 2
fi

variant_arguments() {
  case "$1" in
    feature_alpha0) printf '%s\n' "feature_only_mlp 2 0.0" ;;
    dual_alpha025) printf '%s\n' "dual_feature_spectral 2 0.25" ;;
    *) echo "unknown E18c variant: $1" >&2; return 2 ;;
  esac
}

read -r -a SEEDS <<<"${E18C_SEEDS:-76 77 78}"
read -r -a VARIANTS <<<"${E18C_VARIANTS:-feature_alpha0 dual_alpha025}"

for variant in "${VARIANTS[@]}"; do
  read -r encoder_type encoder_depth frozen_alpha <<<"$(variant_arguments "$variant")"
  for seed in "${SEEDS[@]}"; do
    "$PYTHON_BIN" scripts/run_boundary_certified_spectral_e1.py \
      --exp-name "$BASE_EXP/$variant" \
      --protocol-name "FedBCert E18c frozen cone-hybrid ogbn-arxiv scale audit" \
      --result-root "$RESULT_ROOT" \
      --dataset ogbn-arxiv --partition random --num-clients 3 --seed "$seed" \
      --robust-lambda 0.1 --robust-mode adaptive_projected \
      --suite fedavg_only --projection-ratio-cap 0.25 \
      --boundary-activation-threshold 0.20 \
      --boundary-budget-source local_stub_ledger \
      --boundary-aggregation-mode pairwise_masked \
      --lambda-source "unused: E18-frozen standard FedAvg OGB cone scale audit" \
      --rounds 100 --local-epochs 1 --hidden-dim 64 \
      --encoder-type "$encoder_type" --encoder-depth "$encoder_depth" \
      --encoder-residual-scale 0.5 --spectral-fusion-scale "$frozen_alpha" \
      --uncertainty-recurrence retained_mass \
      --certificate-margin-mode pairwise_difference \
      --include-cone-certificate-audit 1 \
      --order-count 3 --dropout 0.2 --lr 0.01 --weight-decay 0.0005 \
      --certificate-margin 0.05 --evaluation-scope full \
      --force "${E18C_FORCE:-0}"
  done
done

"$PYTHON_BIN" scripts/analyze_fedbcert_e18c_ogb_cone_scale.py --require-complete
