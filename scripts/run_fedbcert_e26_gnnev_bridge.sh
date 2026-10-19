#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GNNEV_ROOT="${GNNEV_ROOT:-$ROOT/third_party/GNNEV}"
COMPAT="${GUROBI_COMPAT:-$ROOT/.vendor/gurobi12_compat}"

if [[ ! -d "$COMPAT/gurobipy" ]]; then
  echo "[ERROR] isolated gurobipy 12.0.3 is missing: $COMPAT" >&2
  echo "Install with: $PYTHON_BIN -m pip install --no-deps --target '$COMPAT' gurobipy==12.0.3" >&2
  exit 2
fi

cd "$ROOT"
GNNEV_ROOT="$GNNEV_ROOT" GUROBI_COMPAT="$COMPAT" \
PYTHONPATH="$COMPAT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/run_fedbcert_e26_gnnev_bridge.py --require-complete "$@"
