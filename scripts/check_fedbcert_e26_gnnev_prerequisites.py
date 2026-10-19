from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fedbcert_e26_known_universe_verifier_bridge.json"
DEFAULT_REPO = Path(
    os.environ.get("GNNEV_ROOT", ROOT / "third_party" / "GNNEV")
).resolve()
DEFAULT_COMPAT = Path(
    os.environ.get("GUROBI_COMPAT", ROOT / ".vendor" / "gurobi12_compat")
).resolve()
DEFAULT_SUMMARY = (
    ROOT
    / "results_boundary_certified"
    / "boundary_certified_spectral_e26_known_universe_verifier"
    / "analysis"
    / "bridge_summary.json"
)

if DEFAULT_COMPAT.exists() and str(DEFAULT_COMPAT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_COMPAT))


def git_head(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def gurobi_probe() -> tuple[bool, str | None]:
    if importlib.util.find_spec("gurobipy") is None:
        return False, "gurobipy is not importable"
    try:
        import gurobipy as gp

        model = gp.Model()
        model.dispose()
        return True, None
    except Exception as exc:  # GurobiError is expected when the license is unusable.
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_REPO
    observed_commit = git_head(repo)
    gurobi_spec = importlib.util.find_spec("gurobipy")
    gurobi_usable, gurobi_error = gurobi_probe()
    bridge_summary = (
        json.loads(DEFAULT_SUMMARY.read_text(encoding="utf-8"))
        if DEFAULT_SUMMARY.exists()
        else None
    )
    license_candidates = [
        repo / "gurobi.lic",
        Path.home() / "gurobi.lic",
        Path.home() / ".gurobi" / "gurobi.lic",
    ]
    report = {
        "protocol": config["protocol"],
        "official_repository_present": repo.exists(),
        "official_commit_expected": config["official_commit"],
        "official_commit_observed": observed_commit,
        "official_commit_verified": observed_commit == config["official_commit"],
        "gurobipy_importable": gurobi_spec is not None,
        "gurobi_solver_usable": gurobi_usable,
        "gurobi_solver_error": gurobi_error,
        "gurobi_license_file_detected": any(path.exists() for path in license_candidates),
        "isolated_compatibility_directory": str(DEFAULT_COMPAT),
        "model_translation_frozen": config.get("status")
        == "frozen_before_numeric_execution",
        "task_translation_frozen": config.get("status")
        == "frozen_before_numeric_execution",
        "numeric_execution_allowed": bool(
            gurobi_usable and observed_commit == config["official_commit"]
        ),
        "bridge_complete": bool(bridge_summary and bridge_summary.get("complete")),
        "decision": (
            "external_verifier_bridge_complete"
            if bridge_summary and bridge_summary.get("complete")
            else "ready_for_numeric_execution"
            if gurobi_usable and observed_commit == config["official_commit"]
            else "bridge_blocked_no_external_numeric_claim"
        ),
        "test_labels_used": False,
        "model_or_hyperparameter_selection_performed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
