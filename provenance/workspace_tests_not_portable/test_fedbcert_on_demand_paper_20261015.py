"""Editorial safeguards; not evidence of mathematical or network soundness."""
from pathlib import Path
import csv
import io

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/FedBCert：联邦多项式图预测器的端点盲补全认证 (1).md"
BEFORE = ROOT / "docs/snapshots/FEDBCERT_BEFORE_ON_DEMAND_APPLICATION_20261015.md"


def test_old_table_rows_and_reference_text_preserved():
    old, new = BEFORE.read_text(encoding="utf-8"), PAPER.read_text(encoding="utf-8")
    assert old.split("## 参考文献\n", 1)[1] == new.split("## 参考文献\n", 1)[1]
    for line in old.splitlines():
        if line.startswith("|"):
            assert line in new


def test_new_scope_and_negative_cpu_result_are_explicit():
    paper = PAPER.read_text(encoding="utf-8")
    assert "### 3.10. 应用扩展" in paper
    assert "### 4.8. 受控应用回放" in paper
    assert "### 4.9. 讨论" in paper
    assert "### E.2. 按需授权聚合" in paper
    assert "未认证节点回退后可能改变本地预测" in paper
    assert "不是新的独立确认或真实网络部署" in paper
    assert "CPU 比值均值为 1.238" in paper
    assert "CPU 比值均值为 2.069" in paper


def test_application_table_matches_exclusive_results():
    paper = PAPER.read_text(encoding="utf-8")
    results = ROOT / "results_boundary_certified/fedbcert_on_demand_20261015/pilot/analysis/workload_summary.csv"
    rows = list(csv.DictReader(io.StringIO(results.read_text(encoding="utf-8"))))
    labels = dict(full_reference="全部目标精确完整图推理", exposure_only="仅结构暴露筛选与回退",
        ball_pairwise="原对称球证书与回退", nonnegative_diameter="非负域直径感知对称证书与回退",
        cone="非负区间证书与回退")
    for row in rows:
        if row["workload"] != "all_test":
            continue
        line = (f"| {labels[row['strategy']]} | {float(row['mean_total_wire_bytes']) / 1e6:.4f} | "
                f"{1000 * float(row['mean_seed_median_cpu_seconds']):.2f} | "
                f"{100 * float(row['mean_certified_ratio']):.2f} |")
        assert line in paper
