# FedBCert 按需聚合：本轮实现、结果与交接

本轮按照用户确认的顺序完成基础修正、冻结输入核查、按需精确回退、强对照成本实验和据实写入中文稿。没有重新训练、重选检查点、修改历史实验数字、覆盖旧投稿包或提交 Git。

## 已完成的修改

1. 最新中文稿区分独立纯特征 MLP、同模型 alpha=0 与参数匹配 P=I；原表格数字和参考文献保留。非负区间界在同一非负补全集上的支配关系写入正文及附录，完整推导见同目录的证书支配说明。
2. 新增非负域直径感知对称基线和严格的同预测器测试。原 certificate.py、federated.py 和模型训练实现未被本轮改写。
3. 对八个历史主设置校验137个输入文件。本地及完整图float64 logits逐位复现，最大差异均为0。输入审计结果采用排他创建。
4. 实现所有者侧精确聚合回放：完整度归一化、多跳远端依赖、真实消息序列化/解码、批次内去重、计时的本地编码复用和快照核查。认证策略不读取远端边端点、标签或完整图参照预测。
5. 正式成本执行前冻结协议与实现；完成八种子、两负载、五策略、三次正式重复，共240条记录。全部类别与指定完整图参照一致。协议、实现、输入及八份新增逐节点证据均通过hash复核。
6. 最新中文稿新增第3.10节、第4.8节、表7及附录E.2；讨论顺延至4.9。增加应用内容前的稿件保存在 docs/snapshots/FEDBCERT_BEFORE_ON_DEMAND_APPLICATION_20261015.md。

## 主要结果

| 比较 | 全测试批次平均载荷减幅 | 描述性95%区间 | 逐种子平均CPU比 |
| --- | ---: | ---: | ---: |
| FedBCert / 直接完整图推理 | 22.09% | [19.75%, 24.75%] | 1.238 |
| FedBCert / 结构暴露筛选 | 22.49% | [20.16%, 25.13%] | 0.947 |
| FedBCert / 非负直径强对称基线 | 13.82% | [12.86%, 14.81%] | 0.988 |

上述载荷比较均为8/8种子下降。全测试批次平均总应用层字节从9,661,434降至7,526,460；CPU时间的种子内中位数均值从202.70ms增至250.80ms。32节点次要批次相对直接完整图推理载荷下降60.71%，但逐种子CPU比均值为2.069。

结果通过事先固定的工程门槛。准确解释是：在该授权服务接口及这些历史合成模型上，以额外本地计算减少远程聚合应用消息量。不是普遍加速、实际网络成本下降、人工复核节省、现实业务验证或新的独立确认。按需策略保持完整图类别，不保证所有节点原本的本地类别；未认证节点回退后可能改变本地预测。没有额外训练收益或严格机器级浮点证明。

## 文件入口

最终相关测试与历史回归为 **165 passed**，其中包含新增表7逐项对应正式CSV、旧表格行保留及参考文献原文不变的检查。测试结果不替代数学证明或真实网络评估。

- 最新中文稿：docs/FedBCert：联邦多项式图预测器的端点盲补全认证 (1).md。
- 完整证书证明：docs/FEDBCERT_CERTIFICATE_DOMINANCE_20261015_ZH.md。
- 预执行协议：configs/fedbcert_on_demand_20261015.json 与 docs/FEDBCERT_ON_DEMAND_PROTOCOL_20261015_ZH.md。
- 输入审计：scripts/audit_fedbcert_on_demand_inputs_20261015.py。
- 服务引擎：openfgl/flcore/fedbcert/on_demand.py；几何基线：geometry_baselines.py。
- 运行与分析：scripts/run_fedbcert_on_demand_20261015.py、scripts/analyze_fedbcert_on_demand_20261015.py。
- 结果根目录：results_boundary_certified/fedbcert_on_demand_20261015/。
- 结果报告：该目录 pilot/analysis/RESULTS_ZH.md；正式记录：pilot/runs.json；原始证据：pilot/cases/。

## 运行约定

本工作区已有 input_audit.json、pilot/freeze_manifest.json、runs.json 和分析结果。所有正式输出均排他创建，不要在这里重复运行或删除旧结果。若需重新执行，应先复制代码及所需历史输入到独立工作区，保留原目录；新工作区不预放本轮已生成的输出。在该独立副本中按顺序运行：

```bash
python scripts/audit_fedbcert_on_demand_inputs_20261015.py
python scripts/run_fedbcert_on_demand_20261015.py prepare
python scripts/run_fedbcert_on_demand_20261015.py run
python scripts/analyze_fedbcert_on_demand_20261015.py
```

依赖沿用原 OpenFGL 环境。输入审计会要求原E28b提交记录、八个主设置的图/模型/历史证据及其冻结实现；不能只凭随机种子重建数据冒充历史快照。正式测量一次预热、三次重复，CPU单线程float64；原始字节应可复算，墙钟时间受运行环境影响。

本轮实现与结果包是供审阅、复核的工作区扩展包，不是已经重编译的英文投稿包。旧英文LaTeX、PDF和匿名代码包尚未同步新增应用内容；正式上传前需要单独进行英文同步、编译和版面检查。

扩展包由 scripts/package_fedbcert_on_demand_20261015.py 排他生成，文件为 submissions/FedBCert_OnDemand_Implementation_20261015.zip。包内含本轮新增源码、协议、中文修订稿、应用修订前快照、完整新结果及逐文件SHA-256清单；不包含原E28b大图束、旧检查点和项目依赖，不是独立可运行的完整仓库。
