#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEDLAP_ROOT="${FEDLAP_ROOT:-$ROOT/third_party/fedbcert_e8/FedLap}"
COMMIT="652fba0bb2e9a660a31ce7e8abcf728c225d188c"

mkdir -p "$(dirname "$FEDLAP_ROOT")"
if [[ ! -d "$FEDLAP_ROOT/.git" ]]; then
  git clone https://github.com/JavadAliakbari/FedLap "$FEDLAP_ROOT"
fi
git -C "$FEDLAP_ROOT" fetch --all --tags
git -C "$FEDLAP_ROOT" checkout "$COMMIT"

echo "FedLap checkout ready at $FEDLAP_ROOT"
echo "Install any additional requirements declared by the pinned repository."
