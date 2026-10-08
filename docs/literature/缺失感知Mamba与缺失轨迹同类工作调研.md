# 文献调研：缺失感知 Mamba 与缺失历史轨迹预测同类工作

日期：2026-10-08 | 前置文档：`docs/literature/不插补轨迹预测调研.md`（2026-09-14，免插补顶会核查）
本轮两个问题：①与选题（缺失/部分观测历史下的 evidence-aware 多模态轨迹预测，TrajImpute Mixed 口径）类似的研究；②能处理缺失时间序列输入（missing-aware / irregular-sampling-aware）的 Mamba/SSM 模型，作为缝合候选机制来源。

覆盖方式：web_search 扇出（8+ 组查询）+ arXiv API/S2 citations（TrajImpute 全部 16 篇施引逐条核对）+ 关键候选 abs 页验证。未查 Zotero 本地库（本轮主题为 2024-2026 新方向，本地库预计覆盖有限）。

## 一、TL;DR

1. **direct（免插补）行人轨迹预测在 2026 年仍是空白**：TrajImpute 全部 16 篇施引逐条核对，无一做"缺失坐标直接预测"——新增施引全部是插补系（dual-view diffusion 插补、TCN 插补、协同观测插补）或相邻基准（CrowdTraj/EgoTraj-Bench）。上次调研的空白结论继续成立，但**插补赛道竞争者持续增加**。
2. **最高撞车警报升级：EgoTraj-Bench/BiFlow（arXiv 2510.00405，v2 2026-03）**——双流 flow matching（dual-stream flow matching）同时去噪历史观测+预测未来，"流匹配+不完整历史"叙事已出现。划界点：它是 ego-view 噪声（遮挡/ID switch/跟踪漂移），显式做历史恢复（非 direct），数据集 TBD 非 ETH/UCY 系。
3. **缺失感知 Mamba 已是成型赛道**，按机制分五族（token/mask 输入流、Δ 离散化条件化、时间感知结构改造、mask 嵌入扫描、扩散插补骨干）。其中 **MAS-Mamba 式 Δ 条件化在 M0 上已判负关闭**（C-MAS-delta，2026-10-08），不可作为缝合方向重复投入。
4. 轨迹领域内最近的邻居是 **Sports-Traj 的 BTS 模块**（mask 序列嵌入双向 Mamba 扫描）——项目 skill 已将其列为可审计借鉴机制。

## 二、问题①同类研究（缺失历史下的轨迹预测，按缺失处理范式分类）

| # | 工作 | 出处 | 缺失处理范式 | 与本题关系 |
|---|---|---|---|---|
| 1 | TrajImpute (Chib & Singh) | NeurIPS 2024 D&B | 两步：SAITS 插补→预测；Easy/Hard 协议 | 本题数据基座；其"需要 direct 处理缺失坐标的模型"是 open problem 原文 |
| 2 | MS-TIP (Chib et al.) | ICML 2024 | 联合插补+预测（对角 mask self-attention） | 联合系代表；TrajImpute 期刊版（2025）扩展 |
| 3 | GC-VRNN (Xu et al.) | CVPR 2023 | 联合插补+预测（MS-GNN+VRNN） | 联合系早期代表 |
| 4 | **EgoTraj-Bench / BiFlow** (Liu et al., HKUST-GZ) | arXiv 2510.00405 v2 (2026-03) | **双流 flow matching：历史去噪流+未来预测流+EgoAnchor 特征调制** | **最近邻/最高撞车**：同用流匹配处理不完整历史；异：ego-view 噪声口径、TBD 数据集、显式历史恢复（非 direct）、单作者基准 |
| 5 | Dual-view diffusion (2026) | 期刊（S2 施引，无 arXiv） | 双视角扩散插补 | 插补系新增竞争者 |
| 6 | Collaborative Observation Imputation (2026) | IEEE TM | 并行插补+预测（一致性评估） | 插补系新增竞争者 |
| 7 | Inferring Missing Trajectory with TCN (2026) | arXiv 2607.25147 | TCN 插补 | 纯插补，弱相关 |
| 8 | RealTraj / Det2TrajFormer (Fujii et al.) | arXiv 2411.17376 | 检测噪声直入（detection 作为输入，对跟踪噪声不变） | 噪声（非缺失）口径的 direct 邻居；在 TrajImpute 上用插值填缺失后训练 |
| 9 | Hazy PTP (PhyFusion+Graph-Mamba) | arXiv 2509.24020 | 物理先验去雾+多粒度图+Mamba | 退化视觉（非缺失坐标）；证明 Mamba+PTP+退化观测组合已被占（期刊格式，venue 待核实） |
| 10 | Scene Informer (Kheterpal et al.) | ICRA 2024 | 遮挡 agent 概率查询 | 缺失 agent（非缺失帧）口径 |

**结论**：direct + 多模态 + ETH/UCY 系缺失坐标的组合仍然无人占坑；但"不完整历史 × 生成式模型（flow/diffusion）"整体升温，BiFlow 是第一个流匹配实例，ProFITi/MOSES（见前置文档）是流模型+缺失时序的团队性威胁。

## 三、问题②缺失感知 Mamba/SSM 模型清单（按机制注入位点分族）

| 族 | 工作 | 出处 | 机制（注入位点） | 对 M0 缝合的可借鉴性 |
|---|---|---|---|---|
| A. mask/表征双输入流 | **S4M** (Jing et al.) | ICLR 2025 | S4 双流：表征流+mask 流，GRU-D 式显式建模 mask（MDS-S4） | mask 作为独立流而非拼接特征；S4 非选择性，移植到 selective scan 需重设计 |
| A. mask/表征双输入流 | BiTGraph (Chen et al.) | ICLR 2024 | 偏置 GCN，missing pattern 逐层注入（非 Mamba，图版范式） | 范式参照 |
| B. Δ 离散化条件化 | **MAS-Mamba** (Huang et al.) | IET ITS (2026-07) | mask + elapsed-time 条件化 selective scan 的 Δ；全缺失步 exp(−λΔ) 收缩→状态保持；split-conformal 校准 | **M0 已实测判负（C-MAS-delta）**：UNIV-M minFDE +2.55%、MR +17%、all-valid identity 不成立；结论：只调 Δ 不阻止缺失帧写入状态 |
| C. 时间感知结构改造 | **TIDES** (Barreira et al.) | arXiv 2605.09742 (2026-05) | 把 input-dependence 从 Δ 移到对角状态矩阵 A，Δ 恢复物理时间间隔含义→不规则时间戳原生处理 | 与 B 族本质不同（不是给 Δ 加条件项而是重定位选择性）；轨迹上未见过；有 Fading Flash 诊断基准+代码 |
| C. 时间感知结构改造 | ReTAMamba (Kim & Park) | arXiv 2605.16380 (2026-05) | 不规则临床时序→time-variable token 序列，观测可靠性（缺失度+耗时）估计+多分辨率区间统计描述符 | token 级可靠性聚合思想可与 evidence 特征共用 |
| C. 时间感知（非 Mamba） | VISTA-SSM (Brindle et al.) | arXiv 2410.21527 | LGSSM 参数化混合，变采样率聚类 | 统计建模路线，弱相关 |
| D. mask 嵌入扫描 | **Sports-Traj BTS** (Xu et al.) | ICLR 2025 | BTS 模块：原 mask+翻转 mask 嵌入 Mamba 块学习缺失模式；双向时间 Mamba（BTM）+GSM 空间 | **轨迹领域内最近邻居**；skill 已审计（BTS=mask 学习嵌入，非特征衰减、非 Δ 条件化）；处理的是生成（补全+预测），非 direct 预测 |
| E. 双向+扩散插补骨干 | DiffImp/SSD-TS (Wang et al.) | 2024 (openreview) | BAM 双向注意力 Mamba+CMB 通道 Mamba 作 DDPM 去噪骨干 | 插补任务（非预测）；双向设计参考 |
| E. 双向+扩散插补骨干 | TIMBA (Solís-García et al.) | arXiv 2410.05916（ICLR'25 withdrawn） | 双向 Mamba 块替换 CSDI 的 transformer；消融显示双向>单向 | 插补任务；withdrawn 状态注意引用 |
| E. 双向+扩散插补骨干 | SSSD (Alcaraz & Strodthoff) | 2022 | S4 层作扩散去噪骨干，只对缺失区加噪 | 插补经典 |
| F. 轨迹 Mamba（非缺失） | MambaPTP (Zhang et al.) | TCSVT 36(3):3795-3807, 2026-03 | 纯 Mamba PTP：BGM 门控双向+BTA 对齐 | 无缺失处理；说明"纯 Mamba PTP"已被期刊占位 |
| F. 轨迹 Mamba（非缺失） | **Social-Mamba** (Luan et al., EPFL) | ECCV 2026 | Cycle Mamba 连续双向流；社会三元分解（时间/自我/目标中心扫描）；已嵌入 MoFlow 框架验证 | 无缺失处理；**其 MoFlow 嵌入证明 Mamba 编码器可替换进流匹配轨迹框架**——与本模型架构同构性最强的邻居 |
| F. 轨迹 Mamba（非缺失） | Multi-scale Mamba-LSTM (CVAE) | Neurocomputing 2025 | GAT+Mamba 混合专家 | 无缺失处理 |

## 四、撞车风险与定位

1. **密集竞争区**：插补+预测联合框架（GC-VRNN/MS-TIP/dual-view diffusion/协同插补）；纯 Mamba PTP（MambaPTP）；社会交互 Mamba（Social-Mamba）。这些是 Related Work 必须划界但不阻塞机制的邻居。
2. **空白区（本题立足点）**：direct/no-imputation + 多模态预测 + 缺失坐标（TrajImpute Mixed）在行人轨迹上连续两轮核查（9-14 与 10-08）无人占坑；缺失感知机制进入 **selective scan 内部**（而非输入端拼接）且保持 all-valid 恒等性的设计在轨迹上无先例。
3. **风险点**（按紧迫度）：
   - **BiFlow（EgoTraj-Bench）**：流匹配+不完整历史已出现。划界三件套：direct（不做历史去噪流）、BEV 缺失坐标口径（非 ego-view 感知噪声）、ETH/UCY+TrajImpute 基准。其 v2 是 2026-03，若其后续版本转向 BEV 缺失口径需立即重评。
   - **ProFITi/MOSES 团队**（前置文档已记）：流+缺失时序的持续产出者。
   - **MAS-Mamba 方向的后续**：Δ 条件化虽在 M0 判负，但该团队的 mask-aware 状态转移思想（含 skip-update 语义："missing 时 h_t≈h_{t-1}"）若被搬到行人轨迹并配齐协议，会压缩"缺失感知 selective scan"的叙事空间——本题的差异化必须落在**严格 all-valid 恒等 + 轨迹多模态协议**上。

## 五、对模型缝合的启示（机制菜单，不构成排期）

结合 M0 现状（4 层历史 Mamba + Scene Transformer + decoder state Mamba）与项目内已判负证据（C-MAS-delta Δ 条件化、C8 ET/anchor、evidence prompt、balanced sampling），按注入位点给出候选与警示：

| 缝合候选 | 来源 | 注入位点 | 状态 |
|---|---|---|---|
| mask 翻转嵌入扫描（BTS 式） | Sports-Traj | Mamba 块内嵌 mask 学习 | 未测；skill 已有 code-audit 路径（`bidirectional-mamba-variants.md`） |
| Δ 物理化（TIDES 式：选择性移到 A，Δ=真实时间间隔） | TIDES | selective scan 结构改造 | 未测；与已判负的"Δ 加条件项"机制不同族，判负证据不迁移，但工程量大（需 fork 慢路径） |
| 观测门控写入 / skip-update（缺失帧不写状态） | GRU-D/CRU 语义；C-MAS-delta 判负理由②显式点名未测 | 状态转移门控 | 未测；skill 定位为"最小可证伪路径"的另一半；须配 all-valid 恒等测试（判负理由①教训：禁全局 bias） |
| token 级可靠性重组（把有效观测 patch 化/重排，跳过缺失步） | t-PatchGNN + ReTAMamba | 输入序列构造 | 未测；优点：完全不动 Mamba 内核，风险最低；缺点：叙事接近特征工程 |
| mask 双输入流（S4M 式） | S4M | 双流 SSM | 未测；S4→Mamba 移植需自行设计 |
| ~~Δ 条件化~~ | MAS-Mamba | dt 注入 | **已判负关闭（2026-10-08），勿重复** |

纪律提醒（沿用项目裁定）：主线为 M0 + natural Mixed 唯一滚动基线；任何缝合臂须 default-off 开关、零初始化恒等、单变量、borrowed control 标注、五指标晋级门（含 MR veto）；是否排期以主线裁定为准，本文档仅提供文献依据。

## Sources

[1] https://arxiv.org/abs/2411.00174 — TrajImpute, NeurIPS 2024 D&B（施引清单经 S2 citations API 全量核对，16 篇）
[2] https://arxiv.org/abs/2510.00405 — EgoTraj-Bench/BiFlow v2, 2026-03
[3] https://arxiv.org/abs/2503.00900 — S4M, ICLR 2025
[4] https://doi.org/10.1049/itr2.70304 — MAS-Mamba, IET Intelligent Transport Systems 2026
[5] https://arxiv.org/abs/2605.09742 — TIDES, 2026-05
[6] https://arxiv.org/abs/2605.16380 — ReTAMamba, 2026-05
[7] https://arxiv.org/abs/2405.17680 — Sports-Traj/UniTraj (BTS), ICLR 2025
[8] https://arxiv.org/abs/2410.13338 — DiffImp/SSD-TS, 2024
[9] https://arxiv.org/abs/2410.05916 — TIMBA（ICLR 2025 withdrawn，引用注意状态）
[10] https://arxiv.org/abs/2208.09399 — SSSD
[11] https://ieeexplore.ieee.org/abstract/document/11192499/ — MambaPTP, IEEE TCSVT 36(3):3795-3807, 2026
[12] https://arxiv.org/abs/2605.15424 — Social-Mamba, ECCV 2026
[13] https://arxiv.org/abs/2411.17376 — RealTraj
[14] https://arxiv.org/abs/2509.24020 — Hazy PTP（venue 待核实，暂按 arXiv 引用）
[15] https://arxiv.org/abs/2410.21527 — VISTA-SSM
[16] https://arxiv.org/abs/2309.13893 — Scene Informer, ICRA 2024
[17] S2 施引新增（无 arXiv id）：Dual-view diffusion for pedestrian trajectory imputation (2026)；Collaborative Observation Imputation and Trajectory Prediction via Consistency Evaluation (IEEE TM 2026)；Inferring Missing Trajectory Data with TCN (arXiv 2607.25147)；CrowdTraj (arXiv 2609.07685)
