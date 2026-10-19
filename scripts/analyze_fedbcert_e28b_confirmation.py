#!/usr/bin/env python
"""Locked primary confirmation and full-grid descriptive reporting."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import numpy as np
from fedbcert_e28b_common import sha256, write_json
from fedbcert_e28b_confirmation_common import (
    PROTOCOL, check_index, estimate, exact_sign_p, holm_adjust, load_protocol,
    paired_metrics, primary_decision, require_test_unlock, resample_indices,
)
from run_fedbcert_e28b_confirmation import verified_manifest


def format_estimate(est, scale=100, digits=2):
    if est["mean"] is None:
        return "—"
    mean, (lo, hi) = est["mean"], est["ci95"]
    return f"{scale*mean:.{digits}f} [{scale*lo:.{digits}f}, {scale*hi:.{digits}f}]"


def plot_grid(cells, spec, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    h_values, rhos = spec["confirmation"]["homophily"], spec["confirmation"]["cross_client_mass"]
    lookup = {(cell["h"], cell["rho"]): cell for cell in cells}
    panels = [("p_minus_i_all_macro_f1", "P - I Macro-F1 (percentage points)", "coolwarm"),
              ("exposed_category_flip_fraction", "Exposed category flips (%)", "viridis"),
              ("hybrid_exposed_certified_ratio", "Hybrid coverage on exposed nodes (%)", "viridis"),
              ("hybrid_minus_symmetric_exposed_ratio", "Hybrid - symmetric coverage (pp)", "viridis")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title, cmap) in zip(axes.flat, panels):
        values = np.array([[np.nan if lookup[(h, rho)]["estimates"][metric]["mean"] is None else 100*lookup[(h, rho)]["estimates"][metric]["mean"] for rho in rhos] for h in h_values])
        limits = {}
        if metric == "p_minus_i_all_macro_f1":
            bound = max(float(np.nanmax(np.abs(values))), 1)
            limits = dict(vmin=-bound, vmax=bound)
        color = plt.get_cmap(cmap).copy()
        color.set_bad("#eeeeee")
        im = axis.imshow(values, cmap=color, aspect="auto", **limits)
        axis.set(xticks=range(len(rhos)), xticklabels=[str(rho) for rho in rhos],
                 yticks=range(len(h_values)), yticklabels=[str(h) for h in h_values],
                 xlabel="Cross-client edge mass rho", ylabel="Same-class mass h", title=title)
        for i in range(len(h_values)):
            for j in range(len(rhos)):
                label = "N/A" if np.isnan(values[i, j]) else f"{values[i,j]:.2f}"
                axis.text(j, i, label, ha="center", va="center", color="black", fontsize=10,
                          bbox=dict(facecolor="white", alpha=.7, edgecolor="none", pad=2))
        axis.add_patch(Rectangle((2.5, 1.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5))
        fig.colorbar(im, ax=axis, shrink=.85)
    fig.suptitle("E28b independent confirmation: 8 whole-graph seeds, test only\nBlack outline = locked primary cell; other cells descriptive; zero-cut exposure N/A", fontsize=12)
    fig.savefig(output / "fig_controlled_grid.png", dpi=180)
    plt.close(fig)


def main():
    spec, base = load_protocol()
    output = ROOT / spec["artifacts"]["output_root"]
    manifest = verified_manifest(spec, output)
    commit = require_test_unlock(output, spec)
    audit_path = output / "artifact_integrity_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["all_checks_passed"] or audit["training_runs"] != 288 or audit["checkpoint_commit_sha256"] != sha256(output / "checkpoint_commit.json"):
        raise RuntimeError("complete independent artifact audit required")
    check_index(output, audit["test_artifact_sha256"])
    records, raw, seeds, indices = {}, [], [], resample_indices(spec)
    for unit in manifest["matrix"]:
        path = output / "test_evaluation" / unit["graph_id"] / unit["model"] / "test_metrics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["split"] != "test" or record["protocol_sha256"] != sha256(PROTOCOL):
            raise RuntimeError("split/protocol mismatch")
        records[(unit["seed"], unit["h"], unit["rho"], unit["model"])] = record
        raw += [dict(split="test", seed=unit["seed"], h=unit["h"], rho=unit["rho"], model=unit["model"], **row) for row in record["rows"]]
    cells = []
    for h in spec["confirmation"]["homophily"]:
        for rho in spec["confirmation"]["cross_client_mass"]:
            values = []
            for seed in spec["confirmation"]["seeds"]:
                metrics = paired_metrics(*(records[(seed, h, rho, model)] for model in ("observed_p", "identity", "strong_mlp")))
                values.append(metrics)
                seeds.append(dict(split="test", seed=seed, h=h, rho=rho, **metrics))
            estimates = {key: estimate([value[key] for value in values], indices) for key in values[0]}
            cells.append(dict(h=h, rho=rho, estimates=estimates, role="primary" if dict(h=h, rho=rho) == spec["confirmation"]["primary_cell"] else "descriptive"))
    primary = next(cell for cell in cells if cell["role"] == "primary")
    primary_seed = [r for r in seeds if r["h"] == primary["h"] and r["rho"] == primary["rho"]]
    tests = []
    for hypothesis in spec["statistics"]["primary_family"]:
        metric, margin = hypothesis["metric"], hypothesis["null_margin"]
        vector = [entry[metric] for entry in primary_seed]
        pvalue, successes = exact_sign_p(vector, margin)
        tests.append(dict(**hypothesis, seed_values=vector, successes=successes, n=8,
                          raw_p=pvalue, mean=primary["estimates"][metric]["mean"], ci95=primary["estimates"][metric]["ci95"]))
    for test, adjusted in zip(tests, holm_adjust([test["raw_p"] for test in tests])):
        test["holm_p"] = adjusted
    gates, positive = primary_decision(primary, tests, base)
    decision = spec["decision"]["positive"] if positive else spec["decision"]["negative"]
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    summary = dict(experiment=spec["experiment"], phase="independent_confirmation", split="test", decision=decision,
                   training_runs=288, graph_units=96, independent_seeds=8, primary_cell=spec["confirmation"]["primary_cell"],
                   primary_tests=tests, primary_gates=gates, cells=cells, per_seed=seeds,
                   confidence_interpretation=spec["statistics"]["bootstrap"]["interval"],
                   sign_test_estimand=spec["statistics"]["primary_test_estimand"],
                   test_targets_used_in_training=False, development_test_evaluated=False,
                   old_decisions_changed=False, freeze_manifest_sha256=sha256(output / "freeze_manifest.json"),
                   checkpoint_commit_sha256=sha256(output / "checkpoint_commit.json"),
                   artifact_integrity_audit_sha256=sha256(audit_path),
                   resource_summary=dict(training_elapsed_seconds=commit["training_elapsed_seconds"],
                       test_elapsed_seconds=json.loads((output / "test_evaluation/completion.json").read_text())["test_elapsed_seconds"],
                       artifact_audit_seconds=audit["artifact_audit_seconds"],
                       process_peak_rss_gib=max(commit["process_peak_rss_gib"], audit["process_peak_rss_gib"], json.loads((output / "test_evaluation/completion.json").read_text())["process_peak_rss_gib"]),
                       cuda_peak_gib=max(json.loads((output / "training" / unit["graph_id"] / unit["model"] / "training_record.json").read_text())["cuda_peak_gib"] for unit in manifest["matrix"])))
    write_json(analysis / "summary.json", summary)
    for filename, rows in (("test_subgroups.csv", raw), ("paired_seed_metrics.csv", seeds), ("primary_tests.csv", tests)):
        with (analysis / filename).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_grid(cells, spec, analysis)
    lines = ["# E28b 独立确认结果（新种子的原测试集）", "", f"决策：`{decision}`。96 张图、288 次训练/测试完成；每格 8 个独立整图种子。", "",
             "开发选择的唯一主点 h=0.90、ρ=0.40 已在确认前锁定。其他格只描述适用范围，不作显著性声明。", "",
             "| h | ρ | P−I F1（pp，95% CI） | P−MLP F1（pp，95% CI） | 暴露翻转率（%，95% CI） | 暴露组合覆盖率（%，95% CI） | 覆盖增益（pp，95% CI） |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for cell in cells:
        e = cell["estimates"]
        metrics = ("p_minus_i_all_macro_f1", "p_minus_mlp_all_macro_f1", "exposed_category_flip_fraction", "hybrid_exposed_certified_ratio", "hybrid_minus_symmetric_exposed_ratio")
        lines.append(f"| {cell['h']} | {cell['rho']} | " + " | ".join(format_estimate(e[k]) for k in metrics) + " |")
    lines += ["", "## 主检验（四项 Holm 校正）", "",
              "| 配对指标 | 成功种子数/8 | raw p | Holm p | 均值（pp） | 95% CI（pp） |", "|---|---:|---:|---:|---:|---|"]
    for test in tests:
        lines.append(f"| {test['metric']} | {test['successes']}/8 | {test['raw_p']:.6f} | {test['holm_p']:.6f} | {100*test['mean']:.2f} | [{100*test['ci95'][0]:.2f}, {100*test['ci95'][1]:.2f}] |")
    lines += ["", "主门槛：`" + json.dumps(gates, ensure_ascii=False) + "`。", "",
              "符号检验检验种子差超过门槛的概率是否大于 1/2，不是分布无关的均值检验。区间为 8 个整图种子的 10,000 次配对 percentile bootstrap，非同时区间；不将节点当独立重复。", "",
              "零认证翻转/解析界违反是允许完整图上的实现诊断，不是补全集穷举或预测正确性证明。ρ=0 暴露为空，保留 —。", "",
              "没有使用开发数据封存测试、确认测试调参或替换主点；旧真实图负结果与主稿/投稿包未改动。", "",
              "![确认全网格](fig_controlled_grid.png)", "",
              "下一步：阳性仅可作为受控合成证据，另准备可合入主稿的限定性段落；阴性则按协议停止，不继续同确认数据后验调参。"]
    with (analysis / "RESULTS_ZH.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    from run_fedbcert_e28b_controlled import verify_manifest
    verify_manifest(manifest)
    print(json.dumps(dict(decision=decision, primary=primary, primary_tests=tests, primary_gates=gates,
                          resource_summary=summary["resource_summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
