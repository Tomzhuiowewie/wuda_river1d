"""
多个独立流量过程共同训练一个条件 PINN。

原来的 train.py、data.py、network.py 和 loss.py 保持不变。
"""

from dataclasses import dataclass
import csv
import json
import time

import torch
from torch import nn

from config import CONFIG
from data import RiverData
from loss import PINNResidual, data_loss
from network import FlowNetwork, activations, initialize_weights
from _util import build_scheduler, choose_device, create_output_dir, set_seed


# 1-based、首尾均包含。修改测试过程时只改 TEST_PROCESS_NAMES。
PROCESS_RANGES = {
    "P1": (1, 50),
    "P2": (51, 220),
    "P3": (221, 370),
    "P4": (371, 560),
    "P5": (561, 680),
    "P6": (681, 744),
}
TEST_PROCESS_NAMES = ("P4", "P6")
BOUNDARY_WINDOW_HOURS = 24.0


@dataclass(frozen=True)
class FlowProcess:
    name: str
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float

    @property
    def duration_s(self):
        return max(self.end_time_s - self.start_time_s, 1.0)


def split_processes(data):
    """生成相互独立的训练过程和测试过程。"""
    processes = []
    for name, (start, end) in PROCESS_RANGES.items():
        start_index = start - 1
        end_index = min(end, data.t_s.numel())
        if start_index >= end_index:
            continue
        processes.append(
            FlowProcess(
                name=name,
                start_index=start_index,
                end_index=end_index,
                start_time_s=float(data.t_s[start_index]),
                end_time_s=float(data.t_s[end_index - 1]),
            )
        )

    test_names = set(TEST_PROCESS_NAMES)
    train_processes = [p for p in processes if p.name not in test_names]
    test_processes = [p for p in processes if p.name in test_names]
    return train_processes, test_processes


def process_initial_boundary_data(data, process):
    """当前过程自己的初始条件和上下游边界，时间从 0 开始。"""
    time_slice = slice(process.start_index, process.end_index)
    local_t = data.t_s[time_slice] - process.start_time_s

    x = torch.cat((
        data.x_m,
        torch.full_like(local_t, data.x_m[0]),
        torch.full_like(local_t, data.x_m[-1]),
    ))
    t = torch.cat((torch.zeros_like(data.x_m), local_t, local_t))

    def values(grid):
        return torch.cat((
            grid[process.start_index],
            grid[time_slice, 0],
            grid[time_slice, -1],
        ))

    return {
        "x": x,
        "t": t,
        "z": values(data.z_grid),
        "q": values(data.q_grid),
    }


def process_observation_data(data, process, section_ids):
    """当前过程内指定断面的观测值；空编号表示不使用内部观测。"""
    if not section_ids:
        empty = data.x_m[:0]
        return {"x": empty, "t": empty, "z": empty, "q": empty}

    section_indices = torch.as_tensor(
        [int(section_id) - 1 for section_id in section_ids],
        dtype=torch.long,
        device=data.device,
    )
    section_indices = section_indices[
        (section_indices > 0) & (section_indices < data.x_m.numel() - 1)
    ].unique()
    if section_indices.numel() == 0:
        empty = data.x_m[:0]
        return {"x": empty, "t": empty, "z": empty, "q": empty}

    time_slice = slice(process.start_index, process.end_index)
    local_t = data.t_s[time_slice] - process.start_time_s
    section_count = section_indices.numel()
    return {
        "x": data.x_m[section_indices].repeat(local_t.numel()),
        "t": local_t.repeat_interleave(section_count),
        "z": data.z_grid[time_slice, section_indices].reshape(-1),
        "q": data.q_grid[time_slice, section_indices].reshape(-1),
    }


def sample_physics_points(data, process, count, time_fraction=1.0):
    """在一个过程内部按连续的局部时间采样 PDE 点。"""
    x = data.x_m[0] + (data.x_m[-1] - data.x_m[0]) * torch.rand(
        count, dtype=data.dtype, device=data.device
    )
    t = process.duration_s * time_fraction * torch.rand(
        count, dtype=data.dtype, device=data.device
    )
    return x, t


class ProcessGeometryAdapter:
    """只把局部时间还原成绝对时间，其余数据接口仍使用 RiverData。"""

    def __init__(self, data):
        self.data = data
        self.current_process = None

    def __getattr__(self, name):
        return getattr(self.data, name)

    def hydraulic_geometry(self, x, water_level, local_t):
        absolute_t = local_t + self.current_process.start_time_s
        return self.data.hydraulic_geometry(x, water_level, absolute_t)


class ProcessFlowNetwork(FlowNetwork):
    """共享 MLP：输入 x、局部时间、初始状态和当前过程边界。"""

    def __init__(self, data, config, train_processes):
        super().__init__(
            data,
            config.hidden_layers,
            config.activation,
            initial_boundary_data=None,
            dropout=config.dropout,
        )
        self.current_process = None
        self.window_seconds = BOUNDARY_WINDOW_HOURS * 3600.0

        # x, tau, Z0(x), Q0(x), 8 个边界特征。
        widths = list(config.hidden_layers)
        layers = []
        in_features = 12
        for out_features in widths:
            layers.extend((
                nn.Linear(in_features, out_features),
                activations[config.activation](),
            ))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, 2))
        self.network = nn.Sequential(*layers)
        initialize_weights(self)

        starts = torch.as_tensor(
            [p.start_index for p in train_processes],
            dtype=torch.long,
            device=data.device,
        )
        initial_z = data.z_grid[starts]
        initial_q = data.q_grid[starts]
        z_margin = torch.full_like(initial_z[0], data.scales.depth_m)
        q_margin = torch.full_like(initial_q[0], data.scales.discharge_m3_s)

        # 输出范围只使用训练过程的初始状态，不读取测试过程内部真值。
        self.z_min_curve.copy_(initial_z.min(dim=0).values - z_margin)
        self.z_max_curve.copy_(initial_z.max(dim=0).values + z_margin)
        self.q_min_curve.copy_(
            (initial_q.min(dim=0).values - q_margin).clamp_min(1.0e-6)
        )
        self.q_max_curve.copy_(initial_q.max(dim=0).values + q_margin)

        q_up = torch.cat([
            data.q_grid[p.start_index:p.end_index, 0] for p in train_processes
        ])
        z_down = torch.cat([
            data.z_grid[p.start_index:p.end_index, -1] for p in train_processes
        ])
        self.register_buffer("q_up_min", q_up.min())
        self.register_buffer("q_up_range", (q_up.max() - q_up.min()).clamp_min(1.0e-6))
        self.register_buffer("z_down_min", z_down.min())
        self.register_buffer(
            "z_down_range", (z_down.max() - z_down.min()).clamp_min(1.0e-6)
        )

        dt = (data.t_s[1] - data.t_s[0]).clamp_min(1.0)
        dq_values, dz_values = [], []
        for process in train_processes:
            q = data.q_grid[process.start_index:process.end_index, 0]
            z = data.z_grid[process.start_index:process.end_index, -1]
            dq_values.append((q[1:] - q[:-1]) / dt)
            dz_values.append((z[1:] - z[:-1]) / dt)
        self.register_buffer(
            "dq_scale", torch.cat(dq_values).abs().max().clamp_min(1.0e-6)
        )
        self.register_buffer(
            "dz_scale", torch.cat(dz_values).abs().max().clamp_min(1.0e-6)
        )
        self.register_buffer("dt_seconds", dt.clone())

    def set_process(self, process):
        self.current_process = process

    def _coordinates(self, x_m, local_t_s):
        if self.current_process is None:
            raise RuntimeError("Call model.set_process(process) before model(x, t)")
        shape = torch.broadcast_shapes(x_m.shape, local_t_s.shape)
        x = x_m.expand(shape)
        t = local_t_s.expand(shape)
        x_hat = 2.0 * (x - self.x_min_m) / self.length_m - 1.0
        t_hat = 2.0 * t / t.new_tensor(self.current_process.duration_s) - 1.0
        return shape, x.reshape(-1), t.reshape(-1), x_hat.reshape(-1), t_hat.reshape(-1)

    def _state_at_x(self, x):
        index = torch.searchsorted(
            self.x_grid.contiguous(), x.contiguous(), right=True
        ) - 1
        index = index.clamp(0, self.x_grid.numel() - 2)
        x0, x1 = self.x_grid[index], self.x_grid[index + 1]
        weight = (x - x0) / (x1 - x0)

        z_profile = self.data.z_grid[self.current_process.start_index]
        q_profile = self.data.q_grid[self.current_process.start_index]
        z0 = (1.0 - weight) * z_profile[index] + weight * z_profile[index + 1]
        q0 = (1.0 - weight) * q_profile[index] + weight * q_profile[index + 1]
        return z0, q0

    def _normalise(self, value, low, value_range):
        return 2.0 * (value - low) / value_range - 1.0

    def _boundary_features(self, local_t):
        process = self.current_process
        start = local_t.new_tensor(process.start_time_s)
        end = local_t.new_tensor(process.end_time_s)
        absolute_t = (start + local_t).clamp(start, end)

        _, z_now, q_now, _ = self.data.boundary_values(absolute_t)
        previous_t = torch.maximum(absolute_t - self.dt_seconds, start)
        _, z_previous, q_previous, _ = self.data.boundary_values(previous_t)

        window_t = torch.maximum(
            absolute_t - local_t.new_tensor(self.window_seconds),
            start,
        )
        _, z_window, q_window, _ = self.data.boundary_values(window_t)

        return torch.stack((
            self._normalise(q_now, self.q_up_min, self.q_up_range),
            self._normalise(z_now, self.z_down_min, self.z_down_range),
            (q_now - q_previous) / self.dt_seconds / self.dq_scale,
            (z_now - z_previous) / self.dt_seconds / self.dz_scale,
            self._normalise(
                0.5 * (q_now + q_window), self.q_up_min, self.q_up_range
            ),
            self._normalise(
                0.5 * (z_now + z_window), self.z_down_min, self.z_down_range
            ),
            (q_now - q_window) / self.q_up_range,
            (z_now - z_window) / self.z_down_range,
        ), dim=-1)

    def _features(self, x, t, x_hat, t_hat):
        z0, q0 = self._state_at_x(x)
        initial_features = torch.stack((
            (z0 - self.data.scales.water_level_m) / self.data.scales.depth_m,
            q0 / self.data.scales.discharge_m3_s,
        ), dim=-1)
        return torch.cat((
            torch.stack((x_hat, t_hat), dim=-1),
            initial_features,
            self._boundary_features(t),
        ), dim=-1)


@torch.no_grad()
def process_metrics(model, data, processes):
    """汇总若干独立过程的 L2 和 NSE。"""
    z_predictions, h_predictions, q_predictions = [], [], []
    z_targets, h_targets, q_targets = [], [], []

    for process in processes:
        model.set_process(process)
        indices = torch.arange(
            process.start_index, process.end_index, device=data.device
        )
        absolute_t = data.t_s[indices]
        local_t = absolute_t - process.start_time_s
        tt, xx = torch.meshgrid(local_t, data.x_m, indexing="ij")
        x, t = xx.reshape(-1), tt.reshape(-1)

        z_pred, q_pred = model(x, t)
        area, width, _ = data.hydraulic_geometry(
            x, z_pred, t + process.start_time_s
        )
        h_pred = area / width.clamp_min(1.0e-6)

        z_predictions.append(z_pred)
        h_predictions.append(h_pred)
        q_predictions.append(q_pred)
        z_targets.append(data.z_grid[indices].reshape(-1))
        h_targets.append(data.h_grid[indices].reshape(-1))
        q_targets.append(data.q_grid[indices].reshape(-1))

    z_pred, h_pred, q_pred = map(
        torch.cat, (z_predictions, h_predictions, q_predictions)
    )
    z_true, h_true, q_true = map(torch.cat, (z_targets, h_targets, q_targets))

    def relative_l2(prediction, target):
        return 100.0 * (
            torch.linalg.vector_norm(prediction - target)
            / torch.linalg.vector_norm(target).clamp_min(1.0e-12)
        ).item()

    def nse(prediction, target):
        denominator = (target - target.mean()).square().sum().clamp_min(1.0e-12)
        return 1.0 - ((prediction - target).square().sum() / denominator).item()

    return (
        relative_l2(h_pred, h_true),
        relative_l2(q_pred, q_true),
        nse(z_pred, z_true),
        nse(h_pred, h_true),
        nse(q_pred, q_true),
    )


def save_model(path, model, optimizer, scheduler, epoch, config, best_metric):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.to_dict(),
        "process_ranges": PROCESS_RANGES,
        "test_processes": TEST_PROCESS_NAMES,
        "boundary_window_hours": BOUNDARY_WINDOW_HOURS,
        "best_metric": best_metric,
    }, path)


def train_flow_generalization(config=CONFIG):
    if config.network_type != "mlp":
        raise ValueError("flow_generalization.py 当前只复用 MLP 网络结构")

    set_seed(config.seed)
    device = choose_device(config.device)
    dtype = getattr(torch, config.dtype)
    output_dir = create_output_dir(
        config.output_dir.parent / "flow_generalization"
    )

    data = RiverData(
        config.data_path,
        config.cross_section_path,
        device=device,
        dtype=dtype,
    )
    train_processes, test_processes = split_processes(data)
    boundary_sets = [
        process_initial_boundary_data(data, process)
        for process in train_processes
    ]
    observation_sets = [
        process_observation_data(data, process, config.interior_section_ids)
        for process in train_processes
    ]

    model = ProcessFlowNetwork(
        data, config, train_processes
    ).to(device=device, dtype=dtype)
    process_data = ProcessGeometryAdapter(data)
    residual = PINNResidual(process_data).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)

    experiment_config = config.to_dict()
    experiment_config.update({
        "train_processes": [p.name for p in train_processes],
        "test_processes": [p.name for p in test_processes],
        "process_ranges": PROCESS_RANGES,
        "boundary_window_hours": BOUNDARY_WINDOW_HOURS,
        "network_inputs": ["x", "tau", "z0_x", "q0_x", "boundary_features"],
    })
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(experiment_config, file, ensure_ascii=False, indent=2)

    history_file = (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    )
    fields = [
        "epoch", "total_loss", "data_loss", "physics_loss",
        "train_h_l2", "train_q_l2", "train_mean_l2",
        "train_z_nse", "train_h_nse", "train_q_nse",
        "test_h_l2", "test_q_l2", "test_mean_l2",
        "test_z_nse", "test_h_nse", "test_q_nse",
        "learning_rate",
    ]
    writer = csv.DictWriter(history_file, fieldnames=fields)
    writer.writeheader()

    print(
        f"device={device} train={[p.name for p in train_processes]} "
        f"test={[p.name for p in test_processes]} "
        f"inputs=x,tau,Z0(x),Q0(x),boundary "
        f"physics_points/epoch={config.num_physics_points}"
    )

    best_metric = float("inf")
    best_epoch = 0
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z_losses, q_losses = [], []
        mass_losses, momentum_losses = [], []
        base_count, remainder = divmod(
            config.num_physics_points, len(train_processes)
        )

        if config.sampling_strategy == "causal":
            progress = min(1.0, epoch / max(1, config.causal_warmup_epochs))
            time_fraction = config.causal_start_fraction + (
                1.0 - config.causal_start_fraction
            ) * progress
        else:
            time_fraction = 1.0

        for index, (process, boundary_data, observation_data) in enumerate(
            zip(train_processes, boundary_sets, observation_sets)
        ):
            model.set_process(process)
            process_data.current_process = process

            z_loss, q_loss = data_loss(model, data, boundary_data)
            if len(observation_data["x"]) > 0:
                observation_z_loss, observation_q_loss = data_loss(
                    model, data, observation_data
                )
                z_loss = (
                    z_loss
                    + config.interior_data_weight * observation_z_loss
                )
                q_loss = (
                    q_loss
                    + config.interior_data_weight * observation_q_loss
                )

            point_count = max(1, base_count + int(index < remainder))
            x_phys, t_phys = sample_physics_points(
                data, process, point_count, time_fraction
            )
            mass_loss, momentum_loss = residual.loss(
                model, x_phys, t_phys
            )
            z_losses.append(z_loss)
            q_losses.append(q_loss)
            mass_losses.append(mass_loss)
            momentum_losses.append(momentum_loss)

        z_loss = torch.stack(z_losses).mean()
        q_loss = torch.stack(q_losses).mean()
        mass_loss = torch.stack(mass_losses).mean()
        momentum_loss = torch.stack(momentum_losses).mean()

        data_term = config.z_weight * z_loss + config.q_weight * q_loss
        physics_term = (
            config.mass_weight * mass_loss
            + config.momentum_weight * momentum_loss
        )
        total = (
            config.data_weight * data_term
            + config.physics_weight * physics_term
        )
        total.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
        optimizer.step()
        scheduler.step()

        should_report = (
            epoch == 1
            or epoch % config.print_every == 0
            or epoch == config.epochs
        )
        if should_report:
            model.eval()
            train_metrics = process_metrics(model, data, train_processes)
            test_metrics = process_metrics(model, data, test_processes)
            train_h, train_q, train_z_nse, train_h_nse, train_q_nse = train_metrics
            test_h, test_q, test_z_nse, test_h_nse, test_q_nse = test_metrics
            train_mean = 0.5 * (train_h + train_q)
            test_mean = 0.5 * (test_h + test_q)
            elapsed = time.time() - start_time
            display_best = min(best_metric, test_mean)
            display_best_epoch = epoch if test_mean < best_metric else best_epoch

            writer.writerow({
                "epoch": epoch,
                "total_loss": total.item(),
                "data_loss": data_term.item(),
                "physics_loss": physics_term.item(),
                "train_h_l2": train_h,
                "train_q_l2": train_q,
                "train_mean_l2": train_mean,
                "train_z_nse": train_z_nse,
                "train_h_nse": train_h_nse,
                "train_q_nse": train_q_nse,
                "test_h_l2": test_h,
                "test_q_l2": test_q,
                "test_mean_l2": test_mean,
                "test_z_nse": test_z_nse,
                "test_h_nse": test_h_nse,
                "test_q_nse": test_q_nse,
                "learning_rate": scheduler.get_last_lr()[0],
            })
            history_file.flush()
            print(
                f"epoch {epoch:5d} | "
                f"L {total.item():8.2e} "
                f"D {data_term.item():8.2e} "
                f"P {physics_term.item():8.2e} | "
                f"train L2 {train_h:4.1f}/{train_q:4.1f}/{train_mean:4.1f}% "
                f"NSE {train_z_nse:5.3f}/{train_h_nse:5.3f}/{train_q_nse:5.3f} | "
                f"test L2 {test_h:4.1f}/{test_q:4.1f}/{test_mean:4.1f}% "
                f"NSE {test_z_nse:5.3f}/{test_h_nse:5.3f}/{test_q_nse:5.3f} | "
                f"best {display_best:4.1f}@{display_best_epoch} | "
                f"{elapsed:5.1f}s"
            )

            if test_mean < best_metric:
                best_metric = test_mean
                best_epoch = epoch
                save_model(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    best_metric,
                )

        if epoch % config.save_every == 0 or epoch == config.epochs:
            save_model(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                best_metric,
            )

    history_file.close()

    checkpoint = torch.load(
        output_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Best checkpoint test processes:")
    for process in test_processes:
        h_l2, q_l2, z_nse, h_nse, q_nse = process_metrics(
            model, data, [process]
        )
        mean_l2 = 0.5 * (h_l2 + q_l2)
        print(
            f"  {process.name}: L2 H/Q/mean="
            f"{h_l2:.2f}/{q_l2:.2f}/{mean_l2:.2f}% "
            f"NSE Z/H/Q={z_nse:.4f}/{h_nse:.4f}/{q_nse:.4f}"
        )

    return output_dir / "best.pt"


if __name__ == "__main__":
    train_flow_generalization()
