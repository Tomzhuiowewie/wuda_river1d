"""Framework-independent utilities used by the river1d experiment pipeline."""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def create_output_dir(base_dir: Path) -> Path:
    """按当前时间创建结果目录：xxx_YYYYMMDD_HHMMSS。"""
    base_dir = Path(base_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir.parent / f"{base_dir.name}_{timestamp}"
    index = 1
    while True:
        try:
            output_dir.mkdir(parents=True)
            return output_dir
        except FileExistsError:
            output_dir = base_dir.parent / f"{base_dir.name}_{timestamp}_{index:02d}"
            index += 1


def choose_device(requested: str) -> torch.device:
    """Choose an explicit device, otherwise prefer CUDA, then Apple MPS."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable experiment initialisation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





def build_scheduler(optimizer, config):
    if config.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.epochs), eta_min=config.learning_rate * 0.05
        )

    if config.scheduler_type == "warmup_exp":
        def lr_factor(epoch):
            warmup = max(1, config.warmup_epochs)
            if epoch < warmup:
                return max((epoch + 1) / warmup, 1.0e-6)
            decay_steps = max(1, config.epochs - warmup)
            progress = (epoch - warmup) / decay_steps
            return config.lr_decay ** progress

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    if config.scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

    raise ValueError("scheduler_type must be 'cosine', 'warmup_exp' or 'constant'")


def gradient_norm_squared(loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """用 loss 对参数的梯度范数平方近似 NTK trace。"""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    value = loss.new_tensor(0.0)
    for gradient in gradients:
        if gradient is not None:
            value = value + gradient.detach().square().sum()
    return value


def update_ntk_weights(model, data_term, physics_term, config,
                       data_weight, physics_weight):
    """根据 NTK trace 近似值更新 data/physics 权重。"""
    data_trace = gradient_norm_squared(data_term, model).clamp_min(1.0e-30)
    physics_trace = gradient_norm_squared(physics_term, model).clamp_min(1.0e-30)
    trace_sum = data_trace + physics_trace

    target_data_weight = (trace_sum / (2.0 * data_trace)).clamp(
        config.ntk_min_weight, config.ntk_max_weight
    )
    target_physics_weight = (trace_sum / (2.0 * physics_trace)).clamp(
        config.ntk_min_weight, config.ntk_max_weight
    )

    base_total = (config.data_weight * data_term.detach()
                  + config.physics_weight * physics_term.detach()).clamp_min(1.0e-30)
    target_total = (target_data_weight * data_term.detach()
                    + target_physics_weight * physics_term.detach()).clamp_min(1.0e-30)
    scale = base_total / target_total
    target_data_weight = (target_data_weight * scale).clamp(
        config.ntk_min_weight, config.ntk_max_weight
    )
    target_physics_weight = (target_physics_weight * scale).clamp(
        config.ntk_min_weight, config.ntk_max_weight
    )

    momentum = min(max(float(config.ntk_momentum), 0.0), 1.0)
    data_weight = momentum * data_weight + (1.0 - momentum) * target_data_weight
    physics_weight = momentum * physics_weight + (1.0 - momentum) * target_physics_weight
    return data_weight.detach(), physics_weight.detach(), data_trace.detach(), physics_trace.detach()


@torch.no_grad()
def grid_metrics(model, data, start_time_index=0, end_time_index=None):
    x, t = data.full_grid_coordinates(start_time_index, end_time_index)
    z_pred, q_pred = model(x, t)
    area, width, _ = data.hydraulic_geometry(x, z_pred, t)
    h_pred = area / width.clamp_min(1.0e-6)
    z_true = data.z_grid[start_time_index:end_time_index].reshape(-1)
    h_true = data.h_grid[start_time_index:end_time_index].reshape(-1)
    q_true = data.q_grid[start_time_index:end_time_index].reshape(-1)

    h_rel = torch.linalg.vector_norm(h_pred - h_true) / torch.linalg.vector_norm(h_true)
    q_rel = torch.linalg.vector_norm(q_pred - q_true) / torch.linalg.vector_norm(q_true)

    def nse(pred, true):
        return 1.0 - ((pred - true).square().sum() / (true - true.mean()).square().sum()).item()

    return (
        100.0 * h_rel.item(),
        100.0 * q_rel.item(),
        nse(z_pred, z_true),
        nse(h_pred, h_true),
        nse(q_pred, q_true),
    )
