# 实验结果索引

结果文档按用途组织，原始运行产物仍在 `outputs/`。历史 v1/v2/v3 预实验文档（K=6）已删除，其结论数字保留在《[选题](../research/选题.md)》§八.1。

## 文档清单

| 文档 | 内容 | 关键结论 |
|---|---|---|
| [实验总汇总](实验总汇总.md) | 2026-08-27 起全部实验的唯一权威总汇总 | 当前基线 M0 三环境；E1 gap-scaling 待验证 |
| [完整数据基线对比](完整数据基线对比.md) | 无缺失完整数据上 DeMo vs MoFlow 及公开 SOTA，ETH/UCY + SDD | DeMo 距 MoFlow +14.7%/+20.7%（ETH/UCY）、+10.8%/+14.2%（SDD）；Clean 基线 = M0 0.232/0.389 |
| [缺失数据基线对比](缺失数据基线对比.md) | TrajImpute Easy/Hard 直接预测：DeMo vs MoFlow-direct 重训 vs 论文 impute 基线 | Easy 0.245/0.399（direct 双第一）、Hard 0.432/0.640（ADE 全场最佳） |
| [主线实验记录](主线实验记录.md) | K=20 主链实验 | M0 三环境基线收口（Clean 0.232/0.389 ｜ Easy 0.245/0.399 ｜ Hard 0.432/0.640） |
| [P0A：M0基线产物协议审计](P0A_M0基线产物协议审计_K20.md) | TrajImpute Easy/Hard direct 的 10 条 M0 产物审计 | 10/10 覆盖、配置一致、checkpoint 完整；结果 revision 为 9fd1d25，当前 HEAD 未重现 |
| [D0/P0：M0缺失分层失败诊断](D0_M0缺失分层失败诊断_K20.md) | 按缺失数、anchor lag、forecast gap、有效观测数和 MR 分层 | 高缺失尾部和 top-1/MR 可靠性退化是下一阶段主问题 |
| [P2A：Easy到Hard zero-shot](P2A_Easy到Hard_zero-shot_K20.md) | Easy train checkpoint 直接评估 Hard test | minFDE 宏平均相对 Hard train-adapt 恶化约 101%，MR 恶化约 146% |

## 口径约束

- direct（缺失直接输入）与 impute（先插补后预测）口径不并表；两种输入信息量不同。
- train_adapt（同条件训练测试）与 zero-shot（完整数据训练→缺失测试）分开报告。
- 阶段定义与验收门槛见《[选题](../research/选题.md)》§八。
