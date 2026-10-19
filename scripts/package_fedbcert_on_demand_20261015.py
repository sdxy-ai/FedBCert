"""Create an exclusive implementation/results extension, not a submission ZIP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submissions/FedBCert_OnDemand_Implementation_20261015.zip"
FILES = [
    "docs/FedBCert：联邦多项式图预测器的端点盲补全认证 (1).md",
    "docs/snapshots/FEDBCERT_BEFORE_ON_DEMAND_APPLICATION_20261015.md",
    "docs/FEDBCERT_ON_DEMAND_IMPLEMENTATION_20261015_ZH.md",
    "docs/FEDBCERT_ON_DEMAND_PROTOCOL_20261015_ZH.md",
    "docs/FEDBCERT_CERTIFICATE_DOMINANCE_20261015_ZH.md",
    "configs/fedbcert_on_demand_20261015.json",
    "openfgl/flcore/fedbcert/geometry_baselines.py",
    "openfgl/flcore/fedbcert/on_demand.py",
    "scripts/audit_fedbcert_on_demand_inputs_20261015.py",
    "scripts/run_fedbcert_on_demand_20261015.py",
    "scripts/analyze_fedbcert_on_demand_20261015.py",
    "scripts/package_fedbcert_on_demand_20261015.py",
    "tests/test_fedbcert_geometry_baselines.py",
    "tests/test_fedbcert_on_demand.py",
    "tests/test_fedbcert_on_demand_inputs_20261015.py",
    "tests/test_fedbcert_on_demand_runner_20261015.py",
    "tests/test_fedbcert_on_demand_analysis_20261015.py",
    "tests/test_fedbcert_on_demand_paper_20261015.py",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    results = ROOT / "results_boundary_certified/fedbcert_on_demand_20261015"
    if not (results / "pilot/analysis/summary.json").is_file() or (results / "pilot/failure.json").exists():
        raise RuntimeError("complete successful exclusive analysis required")
    paths = [ROOT / name for name in FILES]
    paths += [path for path in results.rglob("*") if path.is_file()]
    paths = sorted(set(paths))
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    manifest = dict(schema_version=1, date="2026-10-15", files=hashes,
        scope="implementation and controlled-replay results extension; not an English submission package or standalone reproduction repository",
        required_existing_inputs="original OpenFGL environment plus hash-pinned E28b artifacts listed in input_audit.json; not bundled",
        no_original_inputs_or_submission_archives_overwritten=True)
    with OUTPUT.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("README_ZH.md", (ROOT / "docs/FEDBCERT_ON_DEMAND_IMPLEMENTATION_20261015_ZH.md").read_text(encoding="utf-8"))
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("archive integrity failure")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(Path(name).as_posix())).hexdigest() != expected:
                raise RuntimeError(f"archive member hash mismatch: {name}")
    print(json.dumps(dict(archive=str(OUTPUT), bytes=OUTPUT.stat().st_size,
                          files=len(paths), sha256=sha256(OUTPUT)), ensure_ascii=False))


if __name__ == "__main__":
    main()
