"""Portable wrapper gates; no confirmation data generation or training."""
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from replay_fedbcert_e28b import package_index, reserve_output
from fedbcert_e28b_common import sha256


def test_exclusive_output(tmp_path):
    output = tmp_path / "replay"
    assert reserve_output(output) == output
    with pytest.raises(FileExistsError):
        reserve_output(output)


def test_missing_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="generated anonymous"):
        package_index(tmp_path)


def test_changed_source_rejected(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original\n")
    manifest = {"source.py": sha256(source)}
    (tmp_path / "PACKAGE_SHA256.json").write_text(json.dumps(manifest))
    assert package_index(tmp_path) == manifest
    source.write_text("changed\n")
    with pytest.raises(RuntimeError, match="package changed"):
        package_index(tmp_path)


def test_manifest_path_escape(tmp_path):
    (tmp_path / "PACKAGE_SHA256.json").write_text(json.dumps({"../outside": "unused"}))
    with pytest.raises(RuntimeError, match="unsafe manifest path"):
        package_index(tmp_path)


def test_full_replay_orchestration_is_labelled_historical(tmp_path, monkeypatch):
    # Test orchestration only: no graph, model, training, or target is generated.
    import replay_fedbcert_e28b as replay
    import run_fedbcert_e28b_confirmation as engine
    from fedbcert_e28b_confirmation_common import load_protocol, matrix
    import torch
    spec, base = load_protocol()
    calls = []

    def fake_train(specification, base_protocol, output, device):
        frozen = json.loads((output / "freeze_manifest.json").read_text())
        assert frozen["independent_confirmation"] is False
        assert frozen["experiment"] == "E28b-historical-reproduction"
        assert len(frozen["matrix"]) == 288
        calls.append("train_all")

    def fake_test(specification, base_protocol, output):
        assert calls == ["train_all"]
        for unit in matrix(specification):
            folder = output / "test_evaluation" / unit["graph_id"] / unit["model"]
            folder.mkdir(parents=True)
            (folder / "test_metrics.json").write_text(json.dumps(unit))
        calls.append("test_all")

    def fake_paired(p, identity, mlp):
        assert p["graph_id"] == identity["graph_id"] == mlp["graph_id"]
        return {h["metric"]: .03 for h in spec["statistics"]["primary_family"]}

    monkeypatch.setattr(engine, "train_all", fake_train)
    monkeypatch.setattr(engine, "test_all", fake_test)
    monkeypatch.setattr(replay, "paired_metrics", fake_paired)
    monkeypatch.setattr(replay, "package_index", lambda: {})
    output = reserve_output(tmp_path / "full_mock")
    replay.historical_replay(spec, base, output, torch.device("cpu"), {})
    summary = json.loads((output / "replay_summary.json").read_text())
    assert calls == ["train_all", "test_all"]
    assert summary["independent_confirmation"] is False
    assert summary["phase"] == "historical_reproduction"
    assert len(summary["cells"]) == 12 and len(summary["per_seed"]) == 96
    assert summary["decision"] == "replay_completed_not_new_confirmatory_evidence"
