"""Transparent editorial erratum; old seals and experiment outputs are immutable."""
import argparse
import json
from pathlib import Path
from audit_fedbcert_gap_closure_20261005 import ROOT, read, sha, require, audit

OUT=ROOT/"results_boundary_certified/fedbcert_gap_closure_20261005_v2"
V1=ROOT/"FedBCert：端点盲补全集认证（实验收口候选稿20261005）.md"
V2=ROOT/"FedBCert：端点盲补全集认证（实验收口候选稿20261005v2）.md"
NOTE=ROOT/"docs/FEDBCERT_GAP_CLOSURE_V2_ERRATUM_20261005_ZH.md"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    prior=ROOT/"results_boundary_certified/fedbcert_gap_closure_20261005/manifest.json"
    old=read(prior)
    for path,expected in old["artifact_sha256"].items():
        require(sha(ROOT/path)==expected,"v1 frozen artifact changed: "+path)
    result=audit()
    expected=V1.read_text(encoding="utf-8").replace(
        "> 版本：实验收口候选稿；", "> 版本：实验收口候选稿 v2（导航表状态文字勘误）；").replace(
        "| 在有限资源下能否计算？ | 复杂度分析、E18c 大图运行 | 尚未隔离认证单独开销；不作低附加成本排名 |",
        "| 在有限资源下能否计算？ | 复杂度分析、E18c 大图运行、G2 八冻结模型计时 | G2 已隔离受控小图开销；大图认证单独开销仍无证据，不作低成本排名 |"
    ).replace("fedbcert_gap_closure_20261005/manifest.json","fedbcert_gap_closure_20261005_v2/manifest.json")
    expected += "\nv2 只勘误第 4 节导航表残留的待计时文字，未改数值、实验或已封存输出；说明见\n`docs/FEDBCERT_GAP_CLOSURE_V2_ERRATUM_20261005_ZH.md`。\n"
    require(V2.read_text(encoding="utf-8")==expected,"unapproved editorial/scientific change")
    result.update(dict(editorial_revision="navigation status erratum only", prior_seal_sha256=sha(prior),
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in [prior,V2,NOTE,Path(__file__)]}))
    if args.verify:
        require(read(OUT/"manifest.json")==result,"v2 seal mismatch")
    else:
        require(not OUT.exists(),"no overwrite")
        OUT.mkdir()
        with (OUT/"manifest.json").open("x",encoding="utf-8") as handle:
            json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.write("\n")
    print("PASS: v2 textual erratum only; v1 25-file seal and all task outputs unchanged.")


if __name__=="__main__": main()
