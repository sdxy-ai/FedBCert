#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GNNEV_ROOT="${GNNEV_ROOT:-$ROOT/third_party/GNNEV}"
GUROBI_COMPAT="${GUROBI_COMPAT:-$ROOT/.vendor/gurobi12_compat}"
COMMIT="09295460e28f240cc840013cd745fa0f4e903434"

mkdir -p "$(dirname "$GNNEV_ROOT")" "$GUROBI_COMPAT"
if [[ ! -d "$GNNEV_ROOT/.git" ]]; then
  git clone https://github.com/minghao-liu/GNNEV "$GNNEV_ROOT"
fi
git -C "$GNNEV_ROOT" fetch --all --tags
git -C "$GNNEV_ROOT" checkout "$COMMIT"

"$PYTHON_BIN" -m pip install --no-deps --target "$GUROBI_COMPAT" gurobipy==12.0.3

echo "GNNev: $GNNEV_ROOT"
echo "gurobipy compatibility directory: $GUROBI_COMPAT"
echo "A valid Gurobi license is required before running the bridge."

