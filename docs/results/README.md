# 实验结果索引

结果文档按用途组织，原始运行产物仍在 `outputs/`。历史 v1/v2/v3 预实验文档（K=6）已删除，其结论数字保留在《[选题](../research/选题.md)》§八.1。

## 文档清单

| 文档 | 内容 | 关键结论 |
|---|---|---|
| [实验总汇总](实验总汇总.md) | 2026-08-27 起全部实验的唯一权威总汇总 | M0 基线、P2 交叉泛化矩阵已收口；采样策略 screening 已裁定不采用为主线 |
| [完整数据基线对比](完整数据基线对比.md) | 无缺失完整数据上的本文模型与公开 SOTA 参考，ETH/UCY + SDD | Clean 基线 = 0.232/0.389 |
| [缺失数据基线对比](缺失数据基线对比.md) | TrajImpute Easy/Hard 直接预测与公开插补基线 | Easy 0.245/0.399；Hard 0.432/0.640 |
| [主线实验记录](主线实验记录.md) | K=20 主链实验 | M0 三环境基线收口；P2 交叉泛化矩阵已完成 |
| [缺失分布交叉泛化结果](缺失分布交叉泛化结果.md) | Clean/Easy/Hard/Mixed/Hard 的交叉测试与宏平均 | Mixed 与 Hard-specific 接近；Clean zero-shot 显著退化；高缺失尾部仍未解决 |
| [采样策略逐场景对照](采样策略逐场景对照.md) | severity-balanced 与 evidence-balanced 的五场景 Easy/Hard 对照 | 已裁定不采用为主线；保留为 Benchmark 候选 screening |
| [P0A：M0基线产物协议审计](P0A_M0基线产物协议审计_K20.md) | TrajImpute Easy/Hard direct 的 10 条 M0 产物审计 | 10/10 覆盖、配置一致、checkpoint 完整；结果 revision 为 9fd1d25，当前 HEAD 未重现 |
| [D0/P0：M0缺失分层失败诊断](D0_M0缺失分层失败诊断_K20.md) | 按缺失数、anchor lag、forecast gap、有效观测数和 MR 分层 | 高缺失尾部和 top-1/MR 可靠性退化是下一阶段主问题 |

## 口径约束

- direct（缺失直接输入）与 impute（先插补后预测）口径不并表；两种输入信息量不同。
- train_adapt（同条件训练测试）与 zero-shot（完整数据训练→缺失测试）分开报告。
- 阶段定义与验收门槛见《[选题](../research/选题.md)》§八。
