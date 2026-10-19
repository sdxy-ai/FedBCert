#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/run_fedbcert_e25_exact_cone_oracle.py --require-complete "$@"
