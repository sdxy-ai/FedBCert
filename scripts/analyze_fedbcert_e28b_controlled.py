#!/usr/bin/env python
"""Point-estimate E28b development decisions, only after all 24 runs finish."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fedbcert_e28b_common import PROTOCOL, sha256, write_json
from run_fedbcert_e28b_controlled import verify_manifest


def select_row(record, group, certificate=None):
    matches = [row for row in record["rows"] if row["group"] == group
               and (certificate is None or row["certificate"] == certificate)]
    if not matches:
        raise ValueError("missing group/certificate")
    return matches[0]


def paired_values(p, identity, mlp):
    exposed = select_row(p, "exposed", "cone_hybrid")
    symmetric = select_row(p, "exposed", "retained_pairwise")
    all_p = select_row(p, "all", "cone_hybrid")
    if exposed["n"] == 0:
        raise ValueError("development corner unexpectedly has no exposure")
    return dict(p_macro_f1=all_p["macro_f1"], i_macro_f1=select_row(identity, "all")["macro_f1"],
                mlp_macro_f1=select_row(mlp, "all")["macro_f1"],
                p_minus_i_all_macro_f1=all_p["macro_f1"] - select_row(identity, "all")["macro_f1"],
                p_minus_i_exposed_macro_f1=exposed["macro_f1"] - select_row(identity, "exposed")["macro_f1"],
                p_minus_mlp_all_macro_f1=all_p["macro_f1"] - select_row(mlp, "all")["macro_f1"],
                validation_exposure_fraction=exposed["n"] / all_p["n"],
                exposed_softmax_change_fraction=exposed["softmax_change_fraction"],
                exposed_category_flip_fraction=exposed["category_flip_fraction"],
                hybrid_exposed_certified_ratio=exposed["certified_ratio"],
                symmetric_exposed_certified_ratio=symmetric["certified_ratio"],
                hybrid_minus_symmetric_exposed_ratio=exposed["certified_ratio"] - symmetric["certified_ratio"])


def cell_gates(values, thresholds):
    return dict(
        A_graph_utility=(values["p_minus_i_all_macro_f1"] >= thresholds["p_minus_i_all_macro_f1_at_least"]
                         and values["p_minus_i_exposed_macro_f1"] >= thresholds["p_minus_i_exposed_macro_f1_at_least"]),
        A_simple_noninferiority=values["p_minus_mlp_all_macro_f1"] >= thresholds["p_minus_mlp_all_macro_f1_at_least"],
        B1_exposure=values["validation_exposure_fraction"] >= thresholds["validation_exposure_fraction_at_least"],
        B2_actual_sensitivity=(values["exposed_softmax_change_fraction"] >= thresholds["exposed_softmax_change_fraction_at_least_OR"]
                              or values["exposed_category_flip_fraction"] >= thresholds["exposed_category_flip_fraction_at_least_OR"]),
        C1_nontrivial_coverage=values["hybrid_exposed_certified_ratio"] >= thresholds["hybrid_exposed_certified_ratio_at_least"],
        C2_certificate_gain=values["hybrid_minus_symmetric_exposed_ratio"] >= thresholds["hybrid_minus_symmetric_exposed_ratio_at_least"],
    )


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / protocol["artifacts"]["output_root"]
    manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    verify_manifest(manifest)
    completion_path = output / "development/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["completed_training_runs"] != len(manifest["matrix"]) or completion["test_evaluated"]:
        raise ValueError("complete validation-only matrix required")
    records, raw_rows, input_hashes = {}, [], {}
    for unit in manifest["matrix"]:
        folder = output / "development" / unit["graph_id"] / unit["model"]
        path = folder / "validation_metrics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["phase"] != "development" or record["split"] != "validation" or record["protocol_sha256"] != sha256(PROTOCOL):
            raise ValueError("phase/split/protocol mismatch")
        for filename, field in (("selected_state.pt", "checkpoint_sha256"), ("validation_evidence.pt", "evidence_sha256")):
            if sha256(folder / filename) != record[field]:
                raise ValueError("artifact hash mismatch")
        if record["reliability"]["bound_violations"] or record["reliability"]["certified_flip_nodes"]:
            raise ValueError("reliability hard-stop event")
        records[(unit["seed"], unit["h"], unit["rho"], unit["model"])] = record
        raw_rows += [dict(seed=unit["seed"], h=unit["h"], rho=unit["rho"], model=unit["model"], **row) for row in record["rows"]]
        input_hashes[str(path.relative_to(ROOT))] = sha256(path)
    seed_rows, cells = [], []
    for h in protocol["development"]["homophily"]:
        for rho in protocol["development"]["cross_client_mass"]:
            rows = []
            for seed in protocol["seeds"]["development"]:
                values = paired_values(*(records[(seed, h, rho, model)] for model in ("observed_p", "identity", "strong_mlp")))
                row = dict(seed=seed, h=h, rho=rho, **values)
                rows.append(row)
                seed_rows.append(row)
            means = {key: sum(row[key] for row in rows) / len(rows) for key in values}
            gates = cell_gates(means, protocol["stopping"]["development_cell_gate"])
            gates["C3_reliability"] = True
            gates["R_resource"] = True  # Runner enforces resource caps; all units completed.
            cells.append(dict(h=h, rho=rho, seed_count=len(rows), means=means, gates=gates, eligible=all(gates.values())))
    eligible = [cell for cell in cells if cell["eligible"]]
    decision = "eligible_for_separately_authorized_confirmation" if eligible else "stop_after_e28b_development"
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    summary = dict(experiment="E28b", phase="development", decision=decision, training_runs=len(records),
                   confidence_intervals=None, test_evaluated=False, confirmation_started=False,
                   cells=cells, per_seed=seed_rows, input_sha256=input_hashes,
                   freeze_manifest_sha256=sha256(output / "freeze_manifest.json"),
                   interpretation="controlled synthetic validation-only development; no independent confirmation or change to old decisions")
    write_json(analysis / "summary.json", summary)
    for filename, rows in (("validation_subgroups.csv", raw_rows), ("paired_seed_metrics.csv", seed_rows)):
        with (analysis / filename).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# E28b 开发结果（仅验证集）", "", f"决策：`{decision}`。24 次训练完成；测试未评估，确认未启动。", "",
             "两个整图种子的点估计，不报告置信区间或显著性。P/I/MLP 使用同一 observed-P 暴露分组。", "",
             "| h | ρ | P−I F1（百分点） | P−MLP F1（百分点） | 暴露率 | 暴露翻转率 | 组合覆盖率（暴露） | 覆盖增益（百分点） | 未通过门槛 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for cell in cells:
        v = cell["means"]
        failed = ", ".join(k for k, passed in cell["gates"].items() if not passed) or "无"
        lines.append(f"| {cell['h']} | {cell['rho']} | {100*v['p_minus_i_all_macro_f1']:.2f} | {100*v['p_minus_mlp_all_macro_f1']:.2f} | {100*v['validation_exposure_fraction']:.2f}% | {100*v['exposed_category_flip_fraction']:.2f}% | {100*v['hybrid_exposed_certified_ratio']:.2f}% | {100*v['hybrid_minus_symmetric_exposed_ratio']:.2f} | {failed} |")
    lines += ["", "完整分组、每个种子、负结果、资源和可靠性记录保留在 JSON/CSV 和各训练目录。", "",
              "停止不意味着理论无效：图效用、诊断敏感性和可认证性是不同门槛；单一完整图的零违反不能替代补全集理论。",
              "", "下一步：若开发门槛通过，另行授权并冻结确认分析后运行保留种子全网格；若未通过，保留负结果，不自动扩展网格或修改特征信号。"]
    with (analysis / "RESULTS_ZH.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    verify_manifest(manifest)
    print(json.dumps({"decision": decision, "cells": cells, "confirmation_started": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
