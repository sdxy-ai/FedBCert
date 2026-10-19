# FedBCert：基线身份与非负区间证书支配关系复核

日期：2026-10-15。

本文档给出本次稿件修订使用的实验身份核查和完整代数证明。它不报告新训练、覆盖率、查询成本或应用实验，也不改变既有实验数字。对应稿件为同目录的《FedBCert：联邦多项式图预测器的端点盲补全认证 (1).md》。

## 1. 已核实的纯特征基线身份

以下冻结协议均将 feature_alpha0 映射为 encoder_type=feature_only_mlp、encoder_depth=2、alpha=0：

- configs/fedbcert_e15_protocol.json 的 variants。
- configs/fedbcert_e18_cone_protocol.json 的 variants。
- scripts/run_fedbcert_e15_confirmation.sh 的 variant_arguments。
- scripts/run_fedbcert_e18b_cone_confirmation.sh 的 variant_arguments。

已完成运行记录 results_boundary_certified/boundary_certified_spectral_e18_cone/confirmation/feature_alpha0/Computers_metis_c10_seed75/protocol.json 的 arguments.encoder_type 同样为 feature_only_mlp。当前模型实现的 feature_only_encoding 与该分支对应；这不是将一个冻结双分支模型的融合系数事后改为零。

因此必须区分：

1. 式 (7) 中同一双分支预测器在数学上取 alpha=0 的情形。
2. 表 1、验证选择及图 2 零点使用的独立训练纯特征 MLP。
3. 另外训练的参数匹配 P=I 对照。

现有总体表 paper_table_e18b_overall.csv 将 feature_macro_f1 与 selected_macro_f1 分别保存，且 cone_raw_certified_ratio 与 cone_hybrid_certified_ratio 相等。后一观察与下述理论一致，但不是理论证明本身。修订只纠正比较对象及因果措辞，不将独立 MLP 的旧数字解释为同模型置零结果。

## 2. 共同设置与保证目标

固定客户端、本地诱导图、行随机矩阵 P、本地初始隐藏矩阵 H、模型参数、多项式系数 theta、非负融合系数 alpha，以及逐节点有效边数上界 b。设

$$
m_v=\frac{d_v}{d_v+b_v},\qquad M=\operatorname{diag}(m),\qquad S=MP.
$$

所有向量与矩阵不等式均按元素比较；这里的矩阵不等式不是半正定序。P 非负且 P1=1，因而 S 非负、S1=m 且 S 不大于 P。

比较统一在同一个非负补全集上进行。本地与远端初始表示均属于

$$
\mathcal C=\{h\in\mathbb R^p:h\ge0,\ \|h\|_2\le1\}.
$$

非负行随机传播保持该凸集。只限制远端表示非负而允许本地表示含负分量，不足以使用本证明。单纯换成对称归一化矩阵或带符号传播，也不满足这里的行随机条件。

固定待认证节点 v 的名义预测类 a 和一个竞争类 c。令 q=w_{s,a}-w_{s,c} 为列向量，名义类别差为 Delta_{v,a,c}。传播同一个类别差方向 q 时，不能在不同节点或阶数改用邻居各自的预测类方向。

定义理论目标

$$
\gamma_{v,a,c}
=\inf_{\omega\in\overline\Omega_{\mathrm{cone}}(b)}
\bigl(\ell_{v,a}(\omega)-\ell_{v,c}(\omega)\bigr).
$$

使用下确界，而不要求任意远端节点族上的最小值必然达到。任何有效下界 B_{v,a,c} 均满足 B_{v,a,c}<=gamma_{v,a,c}；所有竞争类的下界严格为正才给出类别不变性证书。以下支配关系比较的是这些充分下界，不是改变真实最坏间隔 gamma。

## 3. 保留质量递推的闭式

欧氏单位球误差递推为

$$
u^{(0)}=0,\qquad
u^{(k)}=\min\{Su^{(k-1)}+2(\mathbf1-m),\,2\mathbf1\}.
$$

定义 tau^{(k)}=1-S^k1。由于 S 非负且行和至多为 1，有 0<=S^k1<=1。假设 u^{(k-1)}=2(1-S^{k-1}1)，则未裁剪的一步为

$$
2S\mathbf1-2S^k\mathbf1+2(\mathbf1-m)
=2(\mathbf1-S^k\mathbf1).
$$

该值已位于 [0,2]，因此裁剪不会改变结果。由初值归纳得到

$$
u^{(k)}=2\tau^{(k)}
=2(\mathbf1-S^k\mathbf1),\qquad k\ge0.
$$

S^k1 可解释为在每步使用已知连接最小保留质量时仍留在已知传播路径上的质量。该解释不把闭式当作真实补全误差的精确值，也不把单步紧致性升级为多阶全局紧致性。

对允许正负号的多项式系数，欧氏单位球半径为

$$
r_v=\sum_{k=1}^{K-1}|\theta_k|u_v^{(k)}.
$$

零阶项没有结构误差。

## 4. 非负投影区间的闭式

记

$$
l_q=-\|q_-\|_2,\qquad U_q=\|q_+\|_2,\qquad
D_q=U_q-l_q=\|q_+\|_2+\|q_-\|_2.
$$

这是线性投影在 C 上的精确区间和跨度。由于 H 的各行属于 C，Hq 在 [l_q,U_q] 内。

稿件递推中的下界候选为 PL^{(k-1)} 及其与 l_q 的凸组合；上界候选为 PU^{(k-1)} 及其与 U_q 的凸组合。归纳可知所有前序区间端点均在 [l_q,U_q] 内，故

$$
m\odot PL^{(k-1)}+(\mathbf1-m)l_q\ \preceq\ PL^{(k-1)},
$$

而对应的上界候选不小于 PU^{(k-1)}。因此，在这里的精确算术与共同表示域前提下，原最小值/最大值递推化为

$$
L^{(k)}=SL^{(k-1)}+(\mathbf1-m)l_q,\qquad
U^{(k)}=SU^{(k-1)}+(\mathbf1-m)U_q.
$$

由 L^{(0)}=U^{(0)}=Hq 及 S1=m，得到

$$
\begin{aligned}
L^{(k)}
&=l_q\mathbf1+S^k(Hq-l_q\mathbf1),\\
U^{(k)}
&=U_q\mathbf1+S^k(Hq-U_q\mathbf1).
\end{aligned}
$$

这些闭式只是已有递推的代数等价式，不表示已实施新的快速计算或测得运行时间降低。

## 5. 方向跨度误差与逐竞争类支配

因为 0<=S<=P 且两者非负，逐阶相乘保持序关系，故

$$
P^k-S^k\succeq0,\qquad
(P^k-S^k)\mathbf1=\tau^{(k)}.
$$

结合 l_q1<=Hq<=U_q1，便有

$$
l_q\tau^{(k)}
\preceq(P^k-S^k)Hq
\preceq U_q\tau^{(k)}.
$$

代入前节闭式，得到两侧投影误差

$$
\begin{aligned}
0&\preceq P^kHq-L^{(k)}
\preceq D_q\tau^{(k)}=\frac{D_q}{2}u^{(k)},\\
0&\preceq U^{(k)}-P^kHq
\preceq D_q\tau^{(k)}=\frac{D_q}{2}u^{(k)}.
\end{aligned}
$$

令

$$
\underline s_v(q)
=\sum_{\theta_k\ge0}\theta_kL_v^{(k)}
+\sum_{\theta_k<0}\theta_kU_v^{(k)}.
$$

对正系数使用下端点，对负系数使用上端点。因此即使 theta 带符号，也有

$$
\underline s_v(q)
\ge\sum_k\theta_k[P^kHq]_v-\frac{D_q}{2}r_v.
$$

乘以 alpha>=0，并加入不变特征分支的类别差和图分类头偏置，得到非负区间类别差下界

$$
B^{\mathrm{interval}}_{v,a,c}
\ge\Delta_{v,a,c}-\alpha\frac{D_q}{2}r_v.
$$

q_+ 与 q_- 的支持不交，故

$$
D_q^2
=(\|q_+\|_2+\|q_-\|_2)^2
\le2(\|q_+\|_2^2+\|q_-\|_2^2)
=2\|q\|_2^2.
$$

于是 D_q/2<=||q||_2/sqrt(2)<=||q||_2。非负区间下界逐竞争类不低于原对称下界。若所有竞争类均以原对称规则通过，它们也均以非负区间规则通过，即

$$
\operatorname{Cert}_{\mathrm{sym}}(b)
\subseteq\operatorname{Cert}_{\mathrm{cone}}(b),\qquad
\operatorname{Cert}_{\mathrm{comb}}(b)
=\operatorname{Cert}_{\mathrm{cone}}(b).
$$

因此当前公式下的“组合”并集不是另一项额外的理论收紧。只具有一般欧氏单位球前提时，非负区间证书不可用，不能删掉一般域上的对称方法。

## 6. 非负域直径感知的更强对称基线

对 h,h' 属于 C，非负性给出 h^T h'>=0，因此

$$
\|h-h'\|_2^2
=\|h\|_2^2+\|h'\|_2^2-2h^\top h'
\le2.
$$

故 C 的直径至多为 sqrt(2)；隐藏维数至少为 2 时可由两个不同坐标单位向量达到。维数为 1 时实际直径为 1，sqrt(2) 仍安全但不紧。

用该直径替换定理 1 中一般欧氏单位球的直径 2，令

$$
s^{(0)}=0,\qquad
s^{(k)}=MPs^{(k-1)}+\sqrt2(\mathbf1-m).
$$

说明上界计数语义时，实际保留质量为 m_v(c_v)>=m_v。若前一阶已满足 s<=sqrt(2)，则 [Ps]_v<=sqrt(2)，所以

$$
m_v(c_v)[Ps]_v+\sqrt2(1-m_v(c_v))
\le m_v[Ps]_v+\sqrt2(1-m_v).
$$

因此该递推对每个允许的实际整数计数有效。与第 3 节相同的代数证明给出

$$
s^{(k)}=\sqrt2(\mathbf1-S^k\mathbf1)
=\frac{u^{(k)}}{\sqrt2}.
$$

该式已在 [0,sqrt(2)] 中，故对应截断同样冗余。多项式表示半径成为 r_v/sqrt(2)，从而得到新的有效对称下界

$$
B^{\mathrm{diam}}_{v,a,c}
=\Delta_{v,a,c}
-\alpha\frac{\|q\|_2}{\sqrt2}r_v.
$$

方向跨度下界为

$$
B^{\mathrm{span}}_{v,a,c}
=\Delta_{v,a,c}-\alpha\frac{D_q}{2}r_v.
$$

因此

$$
B^{\mathrm{sym}}_{v,a,c}
\le B^{\mathrm{diam}}_{v,a,c}
\le B^{\mathrm{span}}_{v,a,c}
\le B^{\mathrm{interval}}_{v,a,c}
\le\gamma_{v,a,c}.
$$

该对照可将“表示域直径由 2 缩小到 sqrt(2)”与“类别差方向及已知本地传播信息”分开。既有表格原对称数值仍来自直径 2 的旧规则；本次不把那些数值改标为直径感知结果，不推算未经实际回放的新覆盖率。

## 7. 与加性递推和逐类别界的统一层级

定义加性递推

$$
u_{\rm add}^{(0)}=0,\qquad
u_{\rm add}^{(k)}
=\min\{Pu_{\rm add}^{(k-1)}+2(\mathbf1-m),\,2\mathbf1\},
$$

以及 r_add=sum_k |theta_k|u_add^{(k)}。由于 M 的对角元位于 [0,1]，递推单调性归纳给出 u_mass<=u_add，因而 r<=r_add。

如果分别界定两个类别的 logit 变化，再相减得到类别差下界，则其惩罚为 alpha*r*(||w_{s,a}||_2+||w_{s,c}||_2)。三角不等式保证直接类别差方向的范数不大于该和。结合前述关系，在相同非负补全集上有

$$
\begin{aligned}
&\Delta_{v,a,c}
-\alpha r_{{\rm add},v}
 (\|w_{s,a}\|_2+\|w_{s,c}\|_2)\\
&\quad\le
\Delta_{v,a,c}
-\alpha r_v(\|w_{s,a}\|_2+\|w_{s,c}\|_2)\\
&\quad\le
\Delta_{v,a,c}-\alpha r_v\|q\|_2\\
&\quad\le B^{\rm diam}_{v,a,c}
\le B^{\rm span}_{v,a,c}
\le B^{\rm interval}_{v,a,c}
\le\gamma_{v,a,c}.
\end{aligned}
$$

分别对竞争类取最小值仍保序，故相应严格正间隔接受集合也按此顺序包含。这些都是同一模型的充分证书，不声称真实类别稳定事件本身被改变。

对相同模型、本地图及表示域，若两个有效上界满足 b'<=b，则补全集缩小，真实最坏间隔不降低。半径递推单调性和非负区间包含关系同时保证上述解析证书不丢失先前接受的节点。该结论不覆盖同时更新参数、改变本地关系或更换表示域的情况。

一般而言，逐竞争类组合 min_c max_j B_{j,c} 不弱于先对整条证书取并集的 max_j min_c B_{j,c}。但在本证明的条件下，interval 已经逐竞争类支配 sym；这两种组合都等于 interval，不能把交换 min/max 当作现有公式下的新覆盖率提升来源。

## 8. 一个严格改进的可手算例子

本节是代数例子，不是新增数据集、训练或实测结果。

只有一个本地节点，P=[1]，d=b=1，故 m=1/2、u^{(1)}=1。取 K=2、theta_0=0、theta_1=1、alpha=1。本地隐藏行为

$$
H=\left(\frac12,\frac{\sqrt3}{2}\right)\in\mathcal C.
$$

设两个类别的图分类头权重为 w_{s,a}=(1,0)^T、w_{s,c}=0，分类头偏置为零，不变特征类别差为 -1/10。于是 q=(1,0)^T、||q||=D_q=1，名义类别差为 2/5>0，r=1。

三个半径型下界分别为

$$
B^{\rm sym}=-\frac35,\qquad
B^{\rm diam}=\frac25-\frac1{\sqrt2}<0,\qquad
B^{\rm span}=-\frac1{10}.
$$

非负区间使用 l_q=0 与 Hq=1/2，得到 L^{(1)}=1/4，因此

$$
B^{\rm interval}=\frac14-\frac1{10}
=\frac3{20}>0.
$$

允许的远端表示可取第二坐标单位向量，其 q 投影为零，因此这个一阶例子的最坏类别差确为 3/20。非负区间规则在该例上严格多接受一个节点；它利用了已知本地投影，不仅是表示域直径或方向跨度的改善。该例不证明多阶区间松弛全局精确，也不表示所有节点都会严格改善。

## 9. 数值实现与解释边界

以上结论均在精确算术下成立。浮点实现可能因稀疏求和顺序、非有限值处理、阈值及舍入误差而出现细小差异。固定 guard 不等于向外舍入误差证明；不能根据理论包含关系直接强制覆盖某个数值实现的接受掩码。

若同一实现、同一输入与阈值下发现原对称规则接受而非负区间规则拒绝的节点，应优先检查：

- 两条规则是否使用相同预测器、名义类别、alpha、theta、分类头及不变特征差。
- 是否确实使用同一个计数上界和保留质量，而非一条使用精确计数等式、另一条使用上界语义。
- 是否对同一 q 传播各阶区间；是否正确处理负 theta 和零阶项。
- 本地与远端域前提、行归一化、自环和节点顺序是否一致。
- 数值差异是否来自非有限值、相等阈值、舍入或实现近似。

上述核查不能通过修改历史输出、重命名旧指标或忽略异常节点来完成。本次文稿修订不报告此类额外数值核查已经运行。

最后，支配关系不是最优性定理：保守拓扑闭包、实际编码器值域到 C 的放松，以及不同阶投影区间分别取最坏端点，仍可能造成松弛。它也不证明预测正确、真实图必须使用图传播、实际查询成本下降，或可以跳过联邦客户端的训练与上传。
