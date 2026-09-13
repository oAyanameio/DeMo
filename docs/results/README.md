# 实验结果索引

结果文档按用途分为三份，原始运行产物仍在 `outputs/`。历史 v1/v2/v3 预实验文档（K=6）已删除，其结论数字保留在《[选题](../research/选题.md)》§八.1。

## 文档清单

| 文档 | 内容 | 关键结论 |
|---|---|---|
| [完整数据基线对比](完整数据基线对比.md) | 无缺失完整数据上 DeMo（单/双向）vs MoFlow 及公开 SOTA，ETH/UCY + SDD | 单向稳定优于双向；DeMo 距 MoFlow +14.7%/+20.7%（ETH/UCY）、+10.8%/+14.2%（SDD） |
| [缺失数据基线对比](缺失数据基线对比.md) | TrajImpute Easy/Hard 直接预测：DeMo-uni（主链）vs DeMo-bi vs MoFlow-direct 重训 vs 论文 impute 基线 | Easy 0.245/0.399（direct 双第一）、Hard 0.432/0.640（ADE 全场最佳）；uni 主链 2026-09-13 裁定 |
| [主线实验记录](主线实验记录.md) | K=20 主链阶段实验 P0 → P5 | P0 自建 Clean 臂作废，Clean=原始 ETH/UCY（本文 M0-uni 0.232/0.389，8/28 轮）；P0.5-a 编码器裁定固定单向（Easy uni −4.2%）；P1 证据时钟未通过；B1-uni 复验进行中 |

## 口径约束

- direct（缺失直接输入）与 impute（先插补后预测）口径不并表；两种输入信息量不同。
- train_adapt（同条件训练测试）与 zero-shot（完整数据训练→缺失测试）分开报告。
- 阶段定义与验收门槛见《[选题](../research/选题.md)》§八。
