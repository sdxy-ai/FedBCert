#!/usr/bin/env python3
"""Summarize the full retrospective E28a audit without selecting a predictor."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_fedbcert_e28_exposure import (
    CONFIG, GROUPS, check_output_root, read_csv, read_json, safe_div,
    sha256, source_units, verify_hashes, write_csv, write_json, write_new_text,
)


def mean(values) -> float | None:
    data = [float(value) for value in values if value is not None and value != ""]
    return float(statistics.mean(data)) if data else None


def summarise_groups(rows: list[dict], dataset: str = "all_datasets") -> list[dict]:
    keyed = {(row["identity"], row["group"], row["certificate"]): row for row in rows}
    identities = sorted({row["identity"] for row in rows})
    result = []
    for group in GROUPS:
        current = [keyed[(identity, group, "combined")] for identity in identities]
        counts = [int(row["node_count"]) for row in current]
        record = {
            "dataset": dataset, "group": group, "registered_units": len(identities),
            "defined_units": sum(count > 0 for count in counts),
            "evaluation_occurrences": sum(counts),
            "mean_node_fraction": mean(
                safe_div(int(keyed[(identity, group, "combined")]["node_count"]),
                         int(keyed[(identity, "all", "combined")]["node_count"]))
                for identity in identities
            ),
            "mean_local_macro_f1": mean(row["local_macro_f1"] for row in current),
            "mean_local_accuracy": mean(row["local_accuracy"] for row in current),
        }
        for name in ("symmetric", "cone_raw", "combined"):
            certificate_rows = [keyed[(identity, group, name)] for identity in identities]
            for metric in ("certified_ratio", "certified_accuracy", "accuracy_given_certified"):
                record[f"mean_{name}_{metric}"] = mean(row[metric] for row in certificate_rows)
            certified = sum(int(row["certified_count"]) for row in certificate_rows)
            correct_certified = sum(int(row["correct_certified_count"]) for row in certificate_rows)
            record[f"count_weighted_{name}_certified_ratio"] = safe_div(certified, sum(counts))
            record[f"count_weighted_{name}_certified_accuracy"] = safe_div(correct_certified, sum(counts))
            record[f"count_weighted_{name}_accuracy_given_certified"] = safe_div(correct_certified, certified)
            record[f"{name}_certified_completion_flip_count"] = sum(
                int(row["certified_completion_flip_count"]) for row in certificate_rows
            )
        record["mean_combined_minus_symmetric_ratio"] = mean(
            float(keyed[(identity, group, "combined")]["certified_ratio"])
            - float(keyed[(identity, group, "symmetric")]["certified_ratio"])
            for identity in identities if int(keyed[(identity, group, "combined")]["node_count"])
        )
        record["completion_flip_count"] = sum(int(row["completion_flip_count"]) for row in current)
        record["mean_absolute_completion_logit_change"] = mean(
            row["mean_absolute_completion_logit_change"] for row in current
        )
        result.append(record)
    return result


def coverage_decomposition(rows: list[dict]) -> list[dict]:
    keyed = {(row["identity"], row["group"], row["certificate"]): row for row in rows}
    result = []
    for identity in sorted({row["identity"] for row in rows}):
        all_row = keyed[(identity, "all", "combined")]
        denominator = int(all_row["node_count"])
        record = {key: all_row[key] for key in ("identity", "dataset", "num_clients", "partition", "seed")}
        record["all_test_nodes"] = denominator
        for group in ("exposed", "non_exposed"):
            combined = keyed[(identity, group, "combined")]
            symmetric = keyed[(identity, group, "symmetric")]
            record[f"{group}_node_fraction"] = safe_div(int(combined["node_count"]), denominator)
            record[f"{group}_combined_coverage_mass"] = safe_div(int(combined["certified_count"]), denominator)
            record[f"{group}_new_certified_count"] = int(combined["certified_count"]) - int(symmetric["certified_count"])
            record[f"{group}_coverage_gain_mass"] = safe_div(record[f"{group}_new_certified_count"], denominator)
        all_symmetric = keyed[(identity, "all", "symmetric")]
        record["overall_coverage_gain"] = (
            float(all_row["certified_ratio"]) - float(all_symmetric["certified_ratio"])
        )
        mass = record["exposed_coverage_gain_mass"] + record["non_exposed_coverage_gain_mass"]
        if abs(mass - record["overall_coverage_gain"]) > 1e-12:
            raise RuntimeError("exposed/non-exposed coverage decomposition is inconsistent")
        result.append(record)
    return result


def percent(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.2f}%"


def render_report(summary: dict, table: list[dict]) -> str:
    labels = {"all": "全部", "exposed": "暴露组", "direct": "直接边界", "indirect": "间接暴露", "non_exposed": "非暴露"}
    lines = [
        "# FedBCert E28a 冻结预测器边界暴露审计", "",
        "> 回顾性审计；不重新训练、不选择模型、不构成新的独立效用确认。", "",
        f"完成 {summary['completed_units']}/{summary['expected_units']} 个 E18 非零图预测器单元。",
        "模型包含 P^0、P^1、P^2；暴露组为精确计数边界节点及其一跳本地邻居。", "",
        "下表是单元比例的非加权均值，与 E18 主表的聚合口径一致。空分组为 NA；",
        "各组 defined_units 不一定相同，不能把平均占比与平均覆盖率直接相乘。", "",
        "| 分组 | 有效单元 | 节点占比 | 对称覆盖 | 组合覆盖 | 组合增益 | 组合认证准确率 | 已认证条件正确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {labels[row['group']]} | {row['defined_units']} | {percent(row['mean_node_fraction'])} | "
            f"{percent(row['mean_symmetric_certified_ratio'])} | {percent(row['mean_combined_certified_ratio'])} | "
            f"{percent(row['mean_combined_minus_symmetric_ratio'])} | {percent(row['mean_combined_certified_accuracy'])} | "
            f"{percent(row['mean_combined_accuracy_given_certified'])} |"
        )
    lines.extend([
        "", "## 总体覆盖的来源", "",
        f"组合总体覆盖中，暴露组贡献 {percent(summary['mean_exposed_combined_coverage_mass'])} 的总测试节点，",
        f"非暴露组贡献 {percent(summary['mean_non_exposed_combined_coverage_mass'])}。",
        f"相对对称证书的总体覆盖增益，有 {percent(summary['fraction_of_mean_gain_from_exposed'])} 来自暴露组。", "",
        "这些是先在每个单元以全部测试节点为分母计算，再取均值的质量分解。", "",
        "## 可靠性与来源", "",
        "全部单元的名义分类及各证书总体指标与原冻结指标逐项对齐；认证计算未改变 logits 或模型参数。",
        "合法完整图诊断中认证翻转与解析界违反均为零；该诊断仅检查一个实际补全，不能替代理论或穷举所有补全。",
        "源检查点、划分、协议、指标、缓存以及受保护文件的 SHA-256 在审计前后保持一致。", "",
        "E18 未保存不可变 split bundle 和名义 logits。因此本次采用保存的 membership、现有缓存和原确定性掩码规则重建输入，",
        "并核对图规模、切边数、测试分类与证书指标。当前输入和 logits 的哈希是本次审计指纹，不能倒称为原实验预先冻结的哈希。", "",
        "## 结论边界", "",
        "暴露表示结构上可能受影响，不表示已经翻转；认证只保证给定补全集上的类别不变，不保证类别正确。",
        "CA 的分母为组内全部节点；已认证条件正确率的分母为该组已认证节点，二者都只是有标签的离线评价。",
        "汇总节点数是重复评估出现次数，不是独立样本或不同节点总数。",
        "本审计没有 P=I 效用归因，不修改 E19--E22 停止结论或 E24 图互补降级。",
        "不根据这次旧测试分析重新选择 alpha、证书或有利数据集。", "",
        "## 产物", "",
        "- paper_table_e28a_overall.csv：总体分组与两种聚合口径。",
        "- paper_table_e28a_by_dataset.csv：逐数据集完整分组。",
        "- paper_data_e28a_units.csv：全部逐单元/分组/证书数据。",
        "- coverage_decomposition.csv：逐单元总体覆盖及增益来源。",
        "- audit_summary.json 与 ../freeze_manifest.json：完整性、原指标对齐、来源和代码指纹。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    root = check_output_root(args.root or ROOT / config["result_root"])
    manifest = read_json(root / "freeze_manifest.json")["inventory"]
    if manifest["protocol_sha256"] != sha256(args.config):
        raise SystemExit("audit config changed after inventory")
    if manifest["analysis_code_sha256"] != sha256(Path(__file__)):
        raise SystemExit("analysis code changed after inventory")
    verify_hashes(manifest["source_files"])
    rows = []
    errors = []
    complete_units = 0
    for unit in source_units(config):
        target = root / "units" / unit["identity"]
        complete_path = target / "run_complete.json"
        if not complete_path.exists():
            errors.append(f"missing:{unit['identity']}")
            continue
        completed = read_json(complete_path)
        if completed.get("status") != "complete" or not completed.get("source_metrics_reproduced"):
            errors.append(f"source_mismatch:{unit['identity']}")
            continue
        verify_hashes(completed["signature"]["source_file_hashes"])
        verify_hashes(completed["output_hashes"])
        if (completed["signature"]["protocol_sha256"] != manifest["protocol_sha256"]
                or completed["signature"]["audit_code_sha256"] != manifest["audit_code_sha256"]):
            errors.append(f"signature:{unit['identity']}")
            continue
        current = read_csv(target / "exposure_metrics.csv")
        expected_keys = {(group, name) for group in GROUPS for name in config["certificates"]}
        if (len(current) != len(expected_keys)
                or {(row["group"], row["certificate"]) for row in current} != expected_keys
                or any(row["identity"] != unit["identity"] for row in current)):
            errors.append(f"table_keys:{unit['identity']}")
            continue
        if any(int(row["certified_completion_flip_count"]) for row in current):
            errors.append(f"certified_flip:{unit['identity']}")
            continue
        rows.extend(current)
        complete_units += 1
    analysis = root / "analysis"
    if errors or complete_units != config["expected_units"]:
        value = {"complete": False, "completed_units": complete_units,
                 "expected_units": config["expected_units"], "errors": errors, "no_training": True}
        write_json(analysis / "incomplete_summary.json", value)
        print(json.dumps(value, indent=2))
        if args.require_complete:
            raise SystemExit("audit incomplete: no paper table emitted")
        return
    overall = summarise_groups(rows)
    by_dataset = []
    for dataset in config["datasets"]:
        by_dataset.extend(summarise_groups([row for row in rows if row["dataset"] == dataset], dataset))
    decomposition = coverage_decomposition(rows)
    source = read_csv(ROOT / config["source_overall_table"])[0]
    source_checks = {}
    all_row = overall[0]
    for target_metric, source_metric in (
        ("mean_local_macro_f1", "selected_macro_f1"),
        ("mean_local_accuracy", "selected_accuracy"),
        ("mean_symmetric_certified_ratio", "symmetric_certified_ratio"),
        ("mean_combined_certified_ratio", "cone_hybrid_certified_ratio"),
        ("mean_symmetric_certified_accuracy", "symmetric_certified_accuracy"),
        ("mean_combined_certified_accuracy", "cone_hybrid_certified_accuracy"),
    ):
        difference = abs(all_row[target_metric] - float(source[source_metric]))
        source_checks[source_metric] = {"absolute_difference": difference, "passed": difference <= 1e-9}
    if not all(check["passed"] for check in source_checks.values()):
        write_json(analysis / "overall_source_mismatch.json", source_checks)
        raise SystemExit("reconstructed overall table does not match frozen E18")
    exposed_mass = mean(row["exposed_coverage_gain_mass"] for row in decomposition)
    non_exposed_mass = mean(row["non_exposed_coverage_gain_mass"] for row in decomposition)
    summary = {
        "complete": True, "completed_units": complete_units, "expected_units": config["expected_units"],
        "decision": "retrospective_audit_complete_no_new_graph_utility_claim",
        "source_overall_checks": source_checks, "certificate_violations": 0,
        "registered_sources_unchanged": True, "no_training": True,
        "new_independent_confirmation": False, "historical_stop_decisions_unchanged": True,
        "mean_exposed_combined_coverage_mass": mean(row["exposed_combined_coverage_mass"] for row in decomposition),
        "mean_non_exposed_combined_coverage_mass": mean(row["non_exposed_combined_coverage_mass"] for row in decomposition),
        "mean_exposed_coverage_gain_mass": exposed_mass,
        "mean_non_exposed_coverage_gain_mass": non_exposed_mass,
        "fraction_of_mean_gain_from_exposed": safe_div(exposed_mass, exposed_mass + non_exposed_mass),
        "total_evaluation_occurrences": all_row["evaluation_occurrences"],
        "not_distinct_independent_node_count": True,
    }
    write_csv(analysis / "paper_table_e28a_overall.csv", overall)
    write_csv(analysis / "paper_table_e28a_by_dataset.csv", by_dataset)
    write_csv(analysis / "paper_data_e28a_units.csv", rows)
    write_csv(analysis / "coverage_decomposition.csv", decomposition)
    write_json(analysis / "audit_summary.json", summary)
    write_new_text(analysis / "report_zh.md", render_report(summary, overall))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
