# 公开基线数字来源（用于 MoFlow vs DeMo held-out 对比）

时间：2026-08-27。检索方式：web_search（结果已逐条记录于会话 20260827_191641_f62631）。

## 1. MoFlow（CVPR 2025）教师模型 — 对比主基准
- Project page: https://moflow-imle.github.io/ （表 1，教师模型 / Teacher，单位：米，min20ADE / min20FDE，LOO 8→12 帧，K=20）
- 数字：eth 0.40/0.57, hotel 0.11/0.17, univ 0.23/0.39, zara1 0.15/0.26, zara2 0.12/0.22，平均 0.202/0.322（精确平均 0.20/0.32）
- 本地官方代码核对：/home/lbh/MoFlow（trainer/denoising_model_trainers.py:compute_ADE_FDE；data/dataloader_eth_ucy.py，指标用 *_original_scale，米制）

## 2. SOTA 区间参考（报告第 2 节表）
来源为各方法原论文 ETH/UCY LOO 平均（多经由 MoFlow/SingularTraj 等论文的对比表交叉核对，注意各论文 K/坐标归一化略有差异，仅作区间参考）：
- MID (WACV22) 0.21/0.38
- TUTR (AAAI23) 0.21/0.36
- EqMotion (CVPR23) 0.21/0.35
- EigenTraj (CVPR23) 0.21/0.34
- LED (CVPR23) 0.21/0.33
- SingularTraj (CVPR24) 0.21/0.32

## 3. 经典基线（警示：数字多源差异大）
- Social-GAN (CVPR18)：论文 avg ADE 0.39/FDE 5.88 为 K=1/20 混排，各处引用常不一致，勿直接引用，仅定性参考。
- Social-STGCNN (CVPR20)：avg 0.30/0.47（K=1，非 min20）。
- Trajectron++ (ECCV20)：官方 bug 修正后（GitHub issue #53，作者方复跑 eccv2020 分支 + master 的 derivative_of）：eth 0.67/1.18, hotel 0.18/0.28, univ 0.30/0.54, zara1 0.25/0.41, zara2 0.18/0.32。注意早期论文数字（eth 0.39/0.83 等）因数据处理 bug 偏乐观，社区已弃用。
  https://github.com/StanfordASL/Trajectron-plus-plus/issues/53

## 4. 检索到的辅助页面
- https://sota2.com/research/task/trajectory-prediction （聚合榜，口径混杂，仅参考）
- https://sota2.com/research/task/trajectory-forecasting

## DeMo 本地取证
- held-out 重跑日志：docs/audits/留出测试复核日志.log（10 折全部 rc=0，N 与 MoFlow 官方 test pkl 一致：eth=181, hotel=1053, univ=24334, zara1=2253, zara2=5833）
- 重跑脚本：docs/tools/数据审计复核脚本.py；批量评估：scripts/训练与评估/run_heldout_test.py
