"""划定不同的流量过程"""

import torch


# 2024年数据划分
PROCESS_SEGMENTS = {
    "P1": (1, 50),      # 初期扰动
    "P2": (51, 220),   # 中低流量波动
    "P3": (221, 370),  # 低流量平稳
    "P4": (371, 560),  # 上涨 + 中高流量波动
    "P5": (561, 680),  # 高流量平台
    "P6": (681, 744),  # 快速退水
}

def process_time_masks(time_count, test_segments, device):
    """
    返回 train/test 时间掩码。
    time_count: 时间点数量，例如 744
    test_segments: 例如 ("P4", "P6")
    """

    test_mask = torch.zeros(time_count, dtype=torch.bool, device=device)

    for name in test_segments:
        start, end = PROCESS_SEGMENTS[name]
        test_mask[start - 1:end] = True

    train_mask = ~test_mask
    return train_mask, test_mask