import torch


def random_sampling(data, count):
    duration = data.scales.duration_s
    x = data.scales.x_min_m + data.scales.length_m * torch.rand(count, dtype=data.dtype, device=data.device)
    t = data.scales.t_min_s + duration * torch.rand(count, dtype=data.dtype, device=data.device)
    return x, t


def causal_sampling(data, count, time_fraction):
    """在从初始时刻逐步扩展的时间窗内随机采样物理点。"""
    duration = data.scales.duration_s
    fraction = min(max(float(time_fraction), 1.0e-6), 1.0)
    x = data.scales.x_min_m + data.scales.length_m * torch.rand(
        count, dtype=data.dtype, device=data.device)
    t = data.scales.t_min_s + duration * fraction * torch.rand(
        count, dtype=data.dtype, device=data.device)
    return x, t


def physics_sampling(data, config, epoch):
    """按配置采样物理点。"""
    if config.sampling_strategy == "random":
        return random_sampling(data, config.num_physics_points)

    if config.sampling_strategy == "causal":
        progress = min(1.0, epoch / max(1, config.causal_warmup_epochs))
        time_fraction = config.causal_start_fraction + (
            1.0 - config.causal_start_fraction
        ) * progress
        return causal_sampling(data, config.num_physics_points, time_fraction)
