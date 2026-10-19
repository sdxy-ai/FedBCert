"""Archive a failed historical-input gate; never regenerate models or rank tools."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_boundary_certified/boundary_certified_spectral_e26_known_universe_verifier"
OUT = ROOT / "results_boundary_certified/fedbcert_g1_comparability_20261005"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("exclusive audit exists; no overwrite")
    analysis = SOURCE / "analysis"
    files = sorted(p for p in analysis.rglob("*") if p.is_file())
    with (analysis/"paper_data_e26_gnnev_bridge_trials.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    protocol = json.loads((ROOT/"configs/fedbcert_e26_known_universe_verifier_bridge.json").read_text())
    expected = list(map(int,protocol["trial_seeds"]))
    if [int(r["seed"]) for r in rows] != expected or len(rows) != 40:
        raise ValueError("historical all-case manifest mismatch")
    tensors = [p for p in SOURCE.rglob("*") if p.is_file() and p.suffix in (".pt",".pth",".npz",".npy")]
    # Official library examples cannot establish provenance of the E26 trials.
    trial_tensors = [p for p in tensors if "official_bridges" not in p.parts]
    if trial_tensors or any("logits" in c for c in columns):
        raise ValueError("potential new provenance anchor found; manual equivalence review required")
    OUT.mkdir()
    result = dict(task="G1",status="terminal_not_comparable_historical_input_gate_failed",
        failed_gate="No stored E26 model/features or per-completion logits to establish exact frozen-input reconstruction (atol 1e-8). Aggregate max-error/gap/change values are not a full-logit anchor.",
        official_candidate_available=True,
        official_sources=[dict(paper="https://arxiv.org/html/2510.18591v2",
            original_artifact="https://zenodo.org/records/17485304",
            latest_official_version="https://zenodo.org/records/18946199",
            version="v2",software_heritage_directory="swh:1:dir:56341488b456df0a207e42f5a655983f46fefe30",
            published_md5="82abb73d6f5b6c78e6b271c9dfc6ca50",
            downloaded_or_executed=False)],
        claim="This is a failed archival-input gate, not evidence that RobLight cannot encode the models or is inferior.",
        external_source_and_dependency_numeric_pin="not attempted: earlier independent input-provenance gate failed; no numeric execution",
        historical_analysis_inventory={str(p.relative_to(ROOT)):sha(p) for p in files},
        historical_columns=columns,stored_e26_trial_tensors=[],
        trial_status=[dict(seed=s,known_completions=4,status="pre_execution_unavailable") for s in expected],
        numeric_comparisons=0, translated_cases=0, precision=None,solved_instance_ratio=None,
        no_regeneration=True,no_training=True,no_easy_case_search=True,
        historical_bridge_retained=True, same_threat_superiority_claim_allowed=False,
        pins={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"configs/fedbcert_gap_tasks_20261005.json",
            ROOT/"configs/fedbcert_e26_known_universe_verifier_bridge.json",ROOT/"scripts/run_fedbcert_e26_gnnev_bridge.py"]})
    with (OUT/"gate_audit.json").open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write("\n")
    print("G1 terminal: official artifact exists, but 40 frozen-input provenance gates cannot be checked; no numeric comparison.")


if __name__ == "__main__": main()
