# 实验结果索引

结果文档按实验协议和用途分类。原始运行产物仍保留在 `outputs/`，这里存放可阅读的汇总、对照和分析记录。

## 基线对照

用于完整数据上的 DeMo、MoFlow 及单向/双向编码器对照，不属于缺失感知方法主结果。

- [ETH/UCY 结果对照表](baselines/ETHUCY结果对照表.md)
- [SDD 结果对照表](baselines/SDD结果对照表.md)
- [MoFlow 与 DeMo 留出测试对比](baselines/MoFlow与DeMo留出测试对比.md)
- [留出测试结构化数据](baselines/MoFlow与DeMo留出测试数据.csv)

## 历史缺失实验

对应 v1、v2、v3_noguard 自定义协议，统一归档为历史预实验，不与 TrajImpute direct 主结果混排。

- [历史缺失实验总汇总](historical/缺失历史实验总汇总.md)
- [M0/M1/M2 第一组筛选结果](historical/M0M1M2第一组筛选实验结果.md)
- [MoFlow 缺失历史 K 扫描](historical/MoFlow缺失历史K扫描评估.md)
- [历史原始摘要](historical/raw/)

## TrajImpute 直接预测

当前正式 direct 协议的结果统一放在这里：`Clean-direct` 使用完整 ETH/UCY，`Easy-direct` 与 `Hard-direct` 使用 TrajImpute release，以及后续 P0–P5 实验结果。

- [P0：M0-current 完整历史基础能力（Clean-direct，K=20）](P0_Clean-direct_M0-current_K20.md)——2026-09-09，五折 LOSO 全 ok，minFDE20 均值 0.409
- [P0.5：基础增强双环境交叉筛选（K=20）](P0.5_双环境筛选_M0-strong_K20.md)——2026-09-10，Clean 持平(+0.17%)/Easy 改善(−2.49%)，裁定 **M0-strong = B1**
- [DeMo-direct 主协议首轮：M0-current × Easy/Hard（K=20）](DeMo-direct主协议首轮_M0-current_EasyHard_K20.md)——2026-09-10，五场景 train_adapt，Easy 0.253/0.417、Hard 0.433/0.640
- 后续结果按阶段（P1 起）追加。

## 口径约束

- `Clean/Easy/Hard-impute` 是外部插补后预测参考，不能与 direct 结果合并。
- 历史 v1/v2/v3 结果只能用于路线演化、工程诊断和失败边界分析。
- 训练适应、零样本和混合条件泛化必须分开报告。
