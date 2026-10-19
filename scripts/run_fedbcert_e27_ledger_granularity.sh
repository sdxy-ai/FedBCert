#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/run_fedbcert_e27_ledger_granularity.py "$@"
"${PYTHON_BIN}" scripts/analyze_fedbcert_e27_ledger_granularity.py --require-complete
"${PYTHON_BIN}" scripts/plot_fedbcert_e27_ledger_granularity.py
