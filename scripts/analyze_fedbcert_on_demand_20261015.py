"""Exclusive engineering analysis of frozen on-demand protocol replay records.

Reads JSON records and hashes files; never loads graph tensors or labels.
Pilot thresholds describe payload feasibility, not independent confirmation,
real network performance, downstream correctness, or journal acceptability.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("configs/fedbcert_on_demand_20261015.json")
STRATEGIES = ("full_reference", "exposure_only", "ball_pairwise", "nonnegative_diameter", "cone")
WORKLOADS = {"all_test": 2400, "small_batch": 32}
SEEDS = tuple(range(28101, 28109))
REPETITIONS = (0, 1, 2)
COUNTS = ("target_n", "certified_n", "fallback_n", "metadata_wire_bytes", "aggregate_request_bytes",
          "aggregate_response_bytes", "aggregate_wire_bytes", "total_wire_bytes", "remote_vectors",
          "request_response_packets", "class_mismatches", "certified_class_mismatches",
          "nominal_vs_full_class_mismatches")
TIMES = ("cpu_seconds", "service_setup_seconds", "nominal_seconds", "metadata_seconds", "certificate_seconds", "fallback_seconds")
INVARIANT = COUNTS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_index(root, mapping):
    root = Path(root).resolve()
    for relative, expected in mapping.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), f"hash target outside workspace: {relative}")
        require(sha256(path) == expected, f"hash mismatch: {relative}")


def validate_integrity(root, pilot, runs):
    root, pilot = Path(root), Path(pilot)
    spec = read_json(root / PROTOCOL)
    freeze_path = pilot / "freeze_manifest.json"
    freeze = read_json(freeze_path)
    require(runs["schema_version"] == 1, "unsupported runs schema")
    require(tuple(spec["seeds"]) == SEEDS and tuple(spec["strategies"]) == STRATEGIES, "protocol matrix changed")
    require({item["name"]: item["count"] for item in spec["workloads"]} == WORKLOADS, "protocol workloads changed")
    require(spec["timing"]["repetitions"] == 3, "protocol repetitions changed")
    protocol_hash = sha256(root / PROTOCOL)
    audit_path = root / spec["input_audit"]
    audit_hash = sha256(audit_path)
    require(protocol_hash == freeze["protocol_sha256"] == runs["protocol_sha256"], "protocol hash mismatch")
    require(audit_hash == freeze["input_audit_sha256"] == runs["input_audit_sha256"], "input audit hash mismatch")
    require(sha256(freeze_path) == runs["freeze_manifest_sha256"], "freeze manifest hash mismatch")
    audit = read_json(audit_path)
    require(audit["passed"] is True and audit["source_hashes_unchanged_after_audit"] is True, "input audit did not pass")
    require(runs["frozen_input_sha256"] == freeze["frozen_input_sha256"] == audit["verified_file_sha256"],
            "frozen input hash index mismatch")
    require(runs["inputs_unchanged_after"] is True and runs["implementation_unchanged_after"] is True,
            "runner integrity check failed")
    require(bool(freeze["source_sha256"]), "empty frozen implementation index")
    verify_index(root, freeze["frozen_input_sha256"])
    verify_index(root, freeze["source_sha256"])
    return spec, freeze


def validate_records(records):
    expected = {(seed, workload, method, repetition)
                for seed in SEEDS for workload in WORKLOADS for method in STRATEGIES for repetition in REPETITIONS}
    require(isinstance(records, list) and len(records) == 240, "expected exactly 240 measured records")
    indexed = {}
    for row in records:
        key = (row["seed"], row["workload"], row["strategy"], row["repetition"])
        require(key in expected and key not in indexed, f"missing/unexpected/duplicate matrix cell: {key}")
        require(row["graph_id"] == f"h0.90_rho0.40_seed{row['seed']}", "wrong fixed graph cell")
        require(row.get("warmup", False) is False, "warmup record included in measurement matrix")
        for name in COUNTS:
            value = row[name]
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    f"invalid nonnegative integer {name}: {key}")
        for name in TIMES + ("fallback_max_abs_error",):
            value = row[name]
            require(isinstance(value, (float, int)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0, f"invalid finite value {name}: {key}")
        require(row["cpu_seconds"] > 0, "CPU duration must be positive")
        require(row["target_n"] == WORKLOADS[row["workload"]], "target workload count mismatch")
        require(row["certified_n"] + row["fallback_n"] == row["target_n"], "certified/fallback partition mismatch")
        require(row["nominal_vs_full_class_mismatches"] <= row["target_n"], "invalid nominal mismatch count")
        require(row["aggregate_request_bytes"] + row["aggregate_response_bytes"] == row["aggregate_wire_bytes"],
                "aggregate wire identity failed")
        require(row["metadata_wire_bytes"] + row["aggregate_wire_bytes"] == row["total_wire_bytes"],
                "total wire identity failed")
        require(row["class_mismatches"] == row["certified_class_mismatches"] == 0, "class mismatch: hard stop")
        require(row["fallback_max_abs_error"] <= 1e-10, "fallback logit error: hard stop")
        indexed[key] = row
    require(set(indexed) == expected, "incomplete measured matrix")
    grouped = {}
    for seed in SEEDS:
        for workload in WORKLOADS:
            for strategy in STRATEGIES:
                group = [indexed[seed, workload, strategy, rep] for rep in REPETITIONS]
                for name in INVARIANT:
                    require(all(row[name] == group[0][name] for row in group),
                            f"repetition changed fixed count/wire/output {name}: {(seed, workload, strategy)}")
                value = {"seed": seed, "graph_id": group[0]["graph_id"], "workload": workload, "strategy": strategy}
                value.update({name: group[0][name] for name in INVARIANT})
                value["fallback_max_abs_error"] = max(row["fallback_max_abs_error"] for row in group)
                value.update({f"median_{name}": statistics.median(row[name] for row in group) for name in TIMES})
                value["certified_ratio"] = value["certified_n"] / value["target_n"]
                grouped[seed, workload, strategy] = value
    return grouped


def estimate(values, indices):
    vector = np.asarray(values, dtype=np.float64)
    require(vector.shape == (8,) and np.isfinite(vector).all(), "expected eight complete finite seed values")
    means = vector[indices].mean(axis=1)
    return {"mean": float(vector.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "n_seeds": 8, "interval_scope": "descriptive paired seed bootstrap, not simultaneous or confirmatory"}


def ideal_break_even(baseline_bytes, cone_bytes, baseline_cpu, cone_cpu):
    saved = baseline_bytes - cone_bytes
    added = cone_cpu - baseline_cpu
    if saved > 0 and added > 0:
        return {"bandwidth_bps": 8 * saved / added, "regime": "cone_wins_only_below_threshold_in_ideal_serial_model"}
    if saved > 0 and added <= 0:
        return {"bandwidth_bps": None, "regime": "cone_has_lower_payload_and_no_added_cpu_in_ideal_model"}
    return {"bandwidth_bps": None, "regime": "no_positive_payload_saving_for_this_comparison"}


def analyze_records(records):
    grouped = validate_records(records)
    indices = np.random.default_rng(28616).integers(0, 8, size=(10000, 8))
    summaries, comparisons = {}, {}
    for workload in WORKLOADS:
        methods = {}
        for strategy in STRATEGIES:
            rows = [grouped[seed, workload, strategy] for seed in SEEDS]
            methods[strategy] = {"total_wire_bytes": estimate([row["total_wire_bytes"] for row in rows], indices),
                                 "mean_seed_median_cpu_seconds": estimate([row["median_cpu_seconds"] for row in rows], indices),
                                 "median_seed_median_cpu_seconds": statistics.median(row["median_cpu_seconds"] for row in rows),
                                 "certified_ratio": estimate([row["certified_ratio"] for row in rows], indices),
                                 "remote_vectors": estimate([row["remote_vectors"] for row in rows], indices),
                                 "request_response_packets": estimate([row["request_response_packets"] for row in rows], indices)}
        summaries[workload] = methods
        comparisons[workload] = {}
        for baseline in STRATEGIES[:-1]:
            baselines = [grouped[seed, workload, baseline] for seed in SEEDS]
            cones = [grouped[seed, workload, "cone"] for seed in SEEDS]
            require(all(row["total_wire_bytes"] > 0 for row in baselines), "zero baseline bytes cannot define primary relative reduction")
            reduction = [1 - cone["total_wire_bytes"] / base["total_wire_bytes"] for cone, base in zip(cones, baselines)]
            cpu_ratio = [cone["median_cpu_seconds"] / base["median_cpu_seconds"] for cone, base in zip(cones, baselines)]
            comparisons[workload][baseline] = {
                "relative_payload_reduction": estimate(reduction, indices),
                "positive_payload_saving_seeds": sum(value > 0 for value in reduction),
                "cone_to_baseline_cpu_ratio": estimate(cpu_ratio, indices),
                "ideal_break_even_by_seed": [{"seed": seed, **ideal_break_even(
                    base["total_wire_bytes"], cone["total_wire_bytes"], base["median_cpu_seconds"], cone["median_cpu_seconds"])}
                    for seed, base, cone in zip(SEEDS, baselines, cones)]}
    primary = comparisons["all_test"]
    gates = {"complete_matrix_and_zero_mismatch": True,
             "cone_vs_exposure_mean_payload_reduction_at_least_5pct": primary["exposure_only"]["relative_payload_reduction"]["mean"] >= .05,
             "cone_vs_exposure_positive_in_at_least_6_of_8": primary["exposure_only"]["positive_payload_saving_seeds"] >= 6,
             "cone_vs_nonnegative_diameter_mean_payload_reduction_positive": primary["nonnegative_diameter"]["relative_payload_reduction"]["mean"] > 0,
             "cone_vs_full_reference_mean_payload_reduction_positive": primary["full_reference"]["relative_payload_reduction"]["mean"] > 0}
    return {"schema_version": 1, "analysis_date": "2026-10-15", "measured_records": 240,
            "replicate_unit": "whole-graph seed; repetitions, clients and nodes are not independent samples",
            "bootstrap": {"draws": 10000, "rng_seed": 28616, "paired": True, "descriptive_only": True},
            "workloads": summaries, "cone_comparisons": comparisons, "engineering_gates": gates,
            "engineering_feasibility_passed": all(gates.values()),
            "decision_if_negative": "stop application expansion; retain certification-paper positioning; no cell, seed or threshold replacement",
            "ideal_network_model": "T = measured CPU seconds + 8 * serialized application bytes / B; serial, no compute/transport overlap, no latency/TCP/TLS/access-control overhead; not measured network performance",
            "claim_limits": ["historical synthetic checkpoints, not new independent confirmation or real deployment",
                             "payload reduction does not imply CPU speedup or real communication-cost savings",
                             "class agreement is with one specified complete-graph predictor, not true-label correctness",
                             "engineering thresholds are not inferential significance tests or acceptance criteria"]}, list(grouped.values())


def csv_text(rows):
    stream = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def summary_rows(report, seed_rows):
    indexed = {(row["seed"], row["workload"], row["strategy"]): row for row in seed_rows}
    indices = np.random.default_rng(28616).integers(0, 8, size=(10000, 8))
    result = []
    for workload in WORKLOADS:
        for strategy in STRATEGIES:
            values = report["workloads"][workload][strategy]
            baseline = [indexed[seed, workload, "full_reference"] for seed in SEEDS]
            current = [indexed[seed, workload, strategy] for seed in SEEDS]
            relative = estimate([1 - a["total_wire_bytes"] / b["total_wire_bytes"] for a, b in zip(current, baseline)], indices)
            cpu_ratio = estimate([a["median_cpu_seconds"] / b["median_cpu_seconds"] for a, b in zip(current, baseline)], indices)
            result.append({"workload": workload, "strategy": strategy, "n_seeds": 8,
                           "mean_total_wire_bytes": values["total_wire_bytes"]["mean"],
                           "wire_mean_ci95_low": values["total_wire_bytes"]["ci95"][0],
                           "wire_mean_ci95_high": values["total_wire_bytes"]["ci95"][1],
                           "mean_seed_median_cpu_seconds": values["mean_seed_median_cpu_seconds"]["mean"],
                           "median_seed_median_cpu_seconds": values["median_seed_median_cpu_seconds"],
                           "mean_certified_ratio": values["certified_ratio"]["mean"],
                           "mean_remote_vectors": values["remote_vectors"]["mean"],
                           "mean_request_response_packets": values["request_response_packets"]["mean"],
                           "mean_seed_relative_payload_reduction_vs_full": relative["mean"],
                           "payload_reduction_vs_full_ci95_low": relative["ci95"][0],
                           "payload_reduction_vs_full_ci95_high": relative["ci95"][1],
                           "mean_seed_cpu_ratio_vs_full": cpu_ratio["mean"]})
    return result


def markdown_report(report):
    passed = report["engineering_feasibility_passed"]
    lines = ["# FedBCert 按需授权聚合的冻结输入协议重放", "",
             "本报告汇总8个既有合成整图种子、2种工作负载、5种策略与3次正式重复，共240条记录。未重新训练或选模型；不是新的独立应用确认。", "",
             "## 主要结果与工程决策", "",
             ("预先定义的工程可行性门槛通过。该结果只支持继续完善这一受控服务场景，不代表期刊接收、现实网络提速或工业效益已经确立。"
              if passed else "预先定义的工程可行性门槛未通过。按原协议停止扩大应用主张，保留认证方法论文定位；不替换工作点、种子或阈值。"), "",
             "| all_test：cone相对基线 | 平均逐种子载荷减幅 | 描述性95%区间 | 载荷降低种子 | 平均逐种子CPU比 |",
             "|---|---:|---:|---:|---:|"]
    for baseline in STRATEGIES[:-1]:
        item = report["cone_comparisons"]["all_test"][baseline]
        value = item["relative_payload_reduction"]
        lines.append(f"| {baseline} | {value['mean'] * 100:.2f}% | [{value['ci95'][0] * 100:.2f}, {value['ci95'][1] * 100:.2f}]% | {item['positive_payload_saving_seeds']}/8 | {item['cone_to_baseline_cpu_ratio']['mean']:.3f} |")
    lines += ["", "载荷统计包含计数/存在性摘要的应用层序列化请求与响应；CPU时间包含协议规定的服务准备、本地推断、认证与回退。字节与时间分别报告，不把二者相加。", "",
              "## 全部工作负载", "",
              "| 工作负载 | 策略 | 平均应用载荷（字节） | 每种子中位CPU时间的均值（ms） | 平均认证比例 |",
              "|---|---|---:|---:|---:|"]
    for workload, methods in report["workloads"].items():
        for strategy, item in methods.items():
            lines.append(f"| {workload} | {strategy} | {item['total_wire_bytes']['mean']:.1f} | {1000 * item['mean_seed_median_cpu_seconds']['mean']:.3f} | {100 * item['certified_ratio']['mean']:.2f}% |")
    lines += ["", "## 理想串行网络模型的临界带宽", "",
              "下表仅在cone字节更少但CPU更长时定义 B*=8×(基线字节−cone字节)/(cone CPU−基线CPU)。模型总时间为CPU秒数加8×字节/B。因而只有带宽低于B*时，该简化模型预测cone总时间更短；带宽高于B*时额外CPU可能抵消字节节省。", "",
              "该模型不含计算与传输重叠、往返延迟、TCP/TLS及访问控制开销；不是实测网络性能，不应将临界带宽等同于真实部署结论。空缺表示该种子不满足字节更少且CPU更长的定义条件。", "",
              "| 工作负载 | 基线 | 种子 | 临界带宽（Mbit/s） | 情形 |", "|---|---|---:|---:|---|"]
    for workload, comparisons in report["cone_comparisons"].items():
        for baseline, item in comparisons.items():
            for record in item["ideal_break_even_by_seed"]:
                bandwidth = record["bandwidth_bps"]
                text_value = "NA" if bandwidth is None else f"{bandwidth / 1e6:.6f}"
                lines.append(f"| {workload} | {baseline} | {record['seed']} | {text_value} | {record['regime']} |")
    lines += ["", "## 完整性、统计口径与范围", "",
              "全部矩阵、逐重复固定计数/字节、零类别不一致、回退logit误差不超过1e-10，以及协议、实现、输入hash均须通过。分析前后再次检查冻结文件。", "",
              "CPU先在每个种子/工作负载/策略内取3次重复中位数；减幅与CPU比先逐种子计算再取均值，不使用汇总值之比。10000次配对整图种子自助抽样使用RNG 28616，区间仅为8个种子的描述性区间。节点、客户端、重复不是独立样本；不报告新的确认性显著性。", "",
              "一致性目标是同一冻结模型在一张指定完整图上的预测类别，不是真实标签正确性。回退需要按需授权的远端精确聚合服务；不能将该接口描述为永远无法访问远端关系的部署。", "",
              "载荷下降不意味着CPU下降。若载荷或CPU出现负结果，应同时保留；不得以纯计算时间、向量数或排除摘要成本的口径替换预先定义的总载荷主指标。", ""]
    return "\n".join(lines)


def write_analysis(folder, report, seed_rows):
    folder = Path(folder)
    folder.mkdir(parents=False, exist_ok=False)
    payloads = {"summary.json": json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                "per_seed.csv": csv_text(seed_rows), "workload_summary.csv": csv_text(summary_rows(report, seed_rows)),
                "RESULTS_ZH.md": markdown_report(report)}
    for name, content in payloads.items():
        with (folder / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)


def main():
    spec = read_json(ROOT / PROTOCOL)
    pilot = ROOT / spec["output_root"]
    analysis = pilot / "analysis"
    if analysis.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {analysis}")
    runs_path = pilot / "runs.json"
    before_runs = sha256(runs_path)
    runs = read_json(runs_path)
    _, freeze = validate_integrity(ROOT, pilot, runs)
    report, seed_rows = analyze_records(runs["records"])
    validate_integrity(ROOT, pilot, runs)
    require(sha256(runs_path) == before_runs, "runs.json changed during analysis")
    report["integrity"] = {"passed": True, "protocol_sha256": runs["protocol_sha256"],
                           "input_audit_sha256": runs["input_audit_sha256"],
                           "freeze_manifest_sha256": runs["freeze_manifest_sha256"],
                           "runs_sha256": before_runs, "frozen_input_files": len(freeze["frozen_input_sha256"]),
                           "frozen_implementation_files": len(freeze["source_sha256"]),
                           "checked_before_and_after": True, "labels_loaded": False}
    report["engineering_gates"]["protocol_input_implementation_integrity"] = True
    report["engineering_feasibility_passed"] = all(report["engineering_gates"].values())
    write_analysis(analysis, report, seed_rows)
    print(json.dumps({"engineering_feasibility_passed": report["engineering_feasibility_passed"],
                      "measured_records": 240, "output": str(analysis)}, indent=2))


if __name__ == "__main__":
    main()
