#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-results_boundary_certified}"

if [[ ! -f data/fedbcert/Cora/linqs_data.pt ]]; then
  "$PYTHON_BIN" scripts/prepare_linqs_data.py --download --datasets Cora
fi

"$PYTHON_BIN" scripts/run_boundary_certified_spectral_e1.py \
  --exp-name smoke/dual_alpha025 \
  --protocol-name "FedBCert artifact smoke test" \
  --result-root "$RESULT_ROOT" \
  --dataset Cora --partition random --num-clients 3 --seed 18 \
  --robust-lambda 0.1 --robust-mode adaptive_projected --suite fedavg_only \
  --projection-ratio-cap 0.25 --boundary-activation-threshold 0.20 \
  --boundary-budget-source local_stub_ledger \
  --boundary-aggregation-mode pairwise_masked \
  --lambda-source "unused: smoke test" \
  --rounds 2 --local-epochs 1 --hidden-dim 16 \
  --encoder-type dual_feature_spectral --encoder-depth 2 \
  --encoder-residual-scale 0.5 --spectral-fusion-scale 0.25 \
  --uncertainty-recurrence retained_mass \
  --certificate-margin-mode pairwise_difference \
  --include-cone-certificate-audit 1 \
  --order-count 3 --dropout 0.2 --lr 0.01 --weight-decay 0.0005 \
  --certificate-margin 0.05 --evaluation-scope full --use-cuda 0 --force 1

completion="$RESULT_ROOT/smoke/dual_alpha025/Cora_random_c3_seed18/run_complete.json"
"$PYTHON_BIN" - "$completion" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "complete":
    raise SystemExit(f"incomplete smoke run: {path}")
print(f"[OK] {path}")
PY
