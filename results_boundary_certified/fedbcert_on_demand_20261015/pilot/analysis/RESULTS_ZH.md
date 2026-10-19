# FedBCert 按需授权聚合的冻结输入协议重放

本报告汇总8个既有合成整图种子、2种工作负载、5种策略与3次正式重复，共240条记录。未重新训练或选模型；不是新的独立应用确认。

## 主要结果与工程决策

预先定义的工程可行性门槛通过。该结果只支持继续完善这一受控服务场景，不代表期刊接收、现实网络提速或工业效益已经确立。

| all_test：cone相对基线 | 平均逐种子载荷减幅 | 描述性95%区间 | 载荷降低种子 | 平均逐种子CPU比 |
|---|---:|---:|---:|---:|
| full_reference | 22.09% | [19.75, 24.75]% | 8/8 | 1.238 |
| exposure_only | 22.49% | [20.16, 25.13]% | 8/8 | 0.947 |
| ball_pairwise | 18.49% | [17.06, 19.99]% | 8/8 | 0.968 |
| nonnegative_diameter | 13.82% | [12.86, 14.81]% | 8/8 | 0.988 |

载荷统计包含计数/存在性摘要的应用层序列化请求与响应；CPU时间包含协议规定的服务准备、本地推断、认证与回退。字节与时间分别报告，不把二者相加。

## 全部工作负载

| 工作负载 | 策略 | 平均应用载荷（字节） | 每种子中位CPU时间的均值（ms） | 平均认证比例 |
|---|---|---:|---:|---:|
| all_test | full_reference | 9661434.0 | 202.697 | 0.00% |
| all_test | exposure_only | 9710580.0 | 264.974 | 0.01% |
| all_test | ball_pairwise | 9229458.0 | 259.177 | 23.67% |
| all_test | nonnegative_diameter | 8730234.0 | 253.742 | 39.21% |
| all_test | cone | 7526460.0 | 250.799 | 65.19% |
| small_batch | full_reference | 1056486.0 | 93.679 | 0.00% |
| small_batch | exposure_only | 1105632.0 | 159.169 | 0.00% |
| small_batch | ball_pairwise | 903888.0 | 166.936 | 25.00% |
| small_batch | nonnegative_diameter | 706020.0 | 148.807 | 43.75% |
| small_batch | cone | 416378.5 | 156.309 | 70.31% |

## 理想串行网络模型的临界带宽

下表仅在cone字节更少但CPU更长时定义 B*=8×(基线字节−cone字节)/(cone CPU−基线CPU)。模型总时间为CPU秒数加8×字节/B。因而只有带宽低于B*时，该简化模型预测cone总时间更短；带宽高于B*时额外CPU可能抵消字节节省。

该模型不含计算与传输重叠、往返延迟、TCP/TLS及访问控制开销；不是实测网络性能，不应将临界带宽等同于真实部署结论。空缺表示该种子不满足字节更少且CPU更长的定义条件。

| 工作负载 | 基线 | 种子 | 临界带宽（Mbit/s） | 情形 |
|---|---|---:|---:|---|
| all_test | full_reference | 28101 | 215.470081 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | full_reference | 28102 | 466.496592 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | full_reference | 28103 | 398.805930 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | full_reference | 28104 | 260.199655 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | full_reference | 28105 | 266.398658 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | full_reference | 28106 | 340.424549 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | full_reference | 28107 | 708.988719 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | full_reference | 28108 | 402.223192 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | exposure_only | 28101 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | exposure_only | 28102 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | exposure_only | 28103 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | exposure_only | 28104 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | exposure_only | 28105 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | exposure_only | 28106 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | exposure_only | 28107 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | exposure_only | 28108 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28101 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28102 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28103 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28104 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28105 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28106 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28107 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | ball_pairwise | 28108 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | nonnegative_diameter | 28101 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | nonnegative_diameter | 28102 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | nonnegative_diameter | 28103 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | nonnegative_diameter | 28104 | 6258.173292 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | nonnegative_diameter | 28105 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | nonnegative_diameter | 28106 | 8162.487770 | cone_wins_only_below_threshold_in_ideal_serial_model |
| all_test | nonnegative_diameter | 28107 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| all_test | nonnegative_diameter | 28108 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | full_reference | 28101 | 73.907785 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | full_reference | 28102 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | full_reference | 28103 | 42.130350 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | full_reference | 28104 | 69.822938 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | full_reference | 28105 | 31.821097 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | full_reference | 28106 | 80.043563 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | full_reference | 28107 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | full_reference | 28108 | 70.388909 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | exposure_only | 28101 | 734.427544 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | exposure_only | 28102 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | exposure_only | 28103 | 626.306756 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | exposure_only | 28104 | 67.400549 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | exposure_only | 28105 | 672.451080 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | exposure_only | 28106 | 1171.812079 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | exposure_only | 28107 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | exposure_only | 28108 | 650.056785 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | ball_pairwise | 28101 | 839.542957 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | ball_pairwise | 28102 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | ball_pairwise | 28103 | 545.347980 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | ball_pairwise | 28104 | 553.421334 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | ball_pairwise | 28105 | 376.107780 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | ball_pairwise | 28106 | 455.289481 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | ball_pairwise | 28107 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | ball_pairwise | 28108 | 617.588674 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | nonnegative_diameter | 28101 | 31.665799 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | nonnegative_diameter | 28102 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | nonnegative_diameter | 28103 | 259.901616 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | nonnegative_diameter | 28104 | 124.677846 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | nonnegative_diameter | 28105 | 211.405177 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | nonnegative_diameter | 28106 | 30.368558 | cone_wins_only_below_threshold_in_ideal_serial_model |
| small_batch | nonnegative_diameter | 28107 | NA | cone_has_lower_payload_and_no_added_cpu_in_ideal_model |
| small_batch | nonnegative_diameter | 28108 | 357.809585 | cone_wins_only_below_threshold_in_ideal_serial_model |

## 完整性、统计口径与范围

全部矩阵、逐重复固定计数/字节、零类别不一致、回退logit误差不超过1e-10，以及协议、实现、输入hash均须通过。分析前后再次检查冻结文件。

CPU先在每个种子/工作负载/策略内取3次重复中位数；减幅与CPU比先逐种子计算再取均值，不使用汇总值之比。10000次配对整图种子自助抽样使用RNG 28616，区间仅为8个种子的描述性区间。节点、客户端、重复不是独立样本；不报告新的确认性显著性。

一致性目标是同一冻结模型在一张指定完整图上的预测类别，不是真实标签正确性。回退需要按需授权的远端精确聚合服务；不能将该接口描述为永远无法访问远端关系的部署。

载荷下降不意味着CPU下降。若载荷或CPU出现负结果，应同时保留；不得以纯计算时间、向量数或排除摘要成本的口径替换预先定义的总载荷主指标。
