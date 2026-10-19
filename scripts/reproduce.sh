#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHON_BIN

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce.sh STAGE

Stages:
  smoke          offline certificate check plus a short Cora run
  confirmation   validation selection followed by independent confirmation
  frontier       five-point alpha utility-certificate frontier
  ledger         source models plus boundary-information granularity audit
  exact          exact finite-completion audit
  ogb            ogbn-arxiv scale audit (needs confirmation)
  safe-cover     FedLap+ teacher and selective certification
  gnnev          optional GNNev cross-check (needs Gurobi)
  all-core       confirmation, frontier, ledger, exact, and ogb
EOF
}

stage="${1:-}"
case "$stage" in
  smoke)
    "$PYTHON_BIN" scripts/smoke_certificate.py
    bash scripts/run_smoke.sh
    ;;
  confirmation)
    bash scripts/run_fedbcert_e18a_cone_development.sh
    bash scripts/run_fedbcert_e18b_cone_confirmation.sh
    "$PYTHON_BIN" scripts/plot_fedbcert_e18.py
    ;;
  frontier)
    bash scripts/run_fedbcert_e15_confirmation.sh
    ;;
  ledger)
    bash scripts/run_ledger_source_models.sh
    bash scripts/run_fedbcert_e27_ledger_granularity.sh
    ;;
  exact)
    bash scripts/run_fedbcert_e25_exact_cone_oracle.sh
    ;;
  ogb)
    bash scripts/run_fedbcert_e18c_ogb_cone_scale.sh
    "$PYTHON_BIN" scripts/plot_fedbcert_e18c.py
    ;;
  safe-cover)
    bash scripts/run_fedbcert_e24_invariant_cover.sh
    ;;
  gnnev)
    bash scripts/run_fedbcert_e26_gnnev_bridge.sh
    ;;
  all-core)
    bash scripts/reproduce.sh confirmation
    bash scripts/reproduce.sh frontier
    bash scripts/reproduce.sh ledger
    bash scripts/reproduce.sh exact
    bash scripts/reproduce.sh ogb
    ;;
  *)
    usage
    exit 2
    ;;
esac
