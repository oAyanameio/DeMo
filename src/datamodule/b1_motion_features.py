"""B1（P0.5 运动证据增强）共享纯函数：跨缺口运动特征。

2026-09-08 统一：ethucy_benchmark_dataset 与 trajimpute_dataset 共用本实现，
消除此前"连续帧版 vs 跨缺口版"两套语义的分叉。

语义（完整历史下的中性约定：与旧相邻帧实现数值一致，s=t-1 时退化）：
    x_velocity      有效帧 t 与前一有效帧 s 的归一化速度 |p[t]-p[s]|/(t-s)
    x_velocity_diff 相邻有效速度对的差（跨缺口速度的导数代理）
    x_turn_rate     相邻有效观测（可不相邻）之间的航向角差，
                    wrap 到 [-pi,pi] 后除以步数 (t-s)；零位移保持前朝向

全部仅由 hist_pos（局部系有限坐标，缺失处占位 0）与 hist_valid 派生；
缺失帧对应位置为 0；无前序有效帧（首个有效帧/单帧可见）为 0。
"""

import torch


def build_b1_motion_features(hist_pos: torch.Tensor, hist_valid: torch.Tensor):
    """跨缺口 B1 运动特征（纯函数，无未来泄漏）。

    Args:
        hist_pos:  [A, T, 2] 有限局部坐标（缺失处为占位 0，不参与计算）
        hist_valid: [A, T] bool，True=该帧有效
    Returns:
        (x_velocity [A, T], x_velocity_diff [A, T], x_turn_rate [A, T])，全 finite。
    """
    A, T, _ = hist_pos.shape
    x_velocity = torch.zeros(A, T)
    x_velocity_diff = torch.zeros(A, T)
    x_turn_rate = torch.zeros(A, T)
    for i in range(A):
        vidx = torch.nonzero(hist_valid[i]).flatten().tolist()
        if len(vidx) < 2:
            continue
        prev_angle = None
        for k in range(1, len(vidx)):
            t, s = vidx[k], vidx[k - 1]
            gap = t - s
            d = hist_pos[i, t] - hist_pos[i, s]
            x_velocity[i, t] = torch.norm(d) / gap
            # x_velocity_diff：与 trajimpute build_gap_aware_motion 同语义——
            # 相邻有效速度对差 vel[t] - vel[s]（s 为前一有效帧，其速度已定义；
            # k=1 时前一有效帧速度为 0/未定义，故 k>=2 才计算）。完整历史下
            # 退化为 vel[t]-vel[t-1]，与旧相邻帧实现数值一致。
            if k >= 2:
                x_velocity_diff[i, t] = x_velocity[i, t] - x_velocity[i, s]
            angle = torch.atan2(d[1], d[0])
            if torch.norm(d) < 1e-4:
                angle = prev_angle if prev_angle is not None else angle
            if prev_angle is not None:
                delta = angle - prev_angle
                delta = torch.atan2(torch.sin(delta), torch.cos(delta))
                x_turn_rate[i, t] = delta / gap
            prev_angle = angle
    return x_velocity, x_velocity_diff, x_turn_rate
