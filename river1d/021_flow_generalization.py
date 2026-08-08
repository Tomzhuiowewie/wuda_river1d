import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from config import CONFIG
except ImportError:
    from .config import CONFIG

try:
    from _geometry import CrossSectionGeometry
except ImportError:
    from ._geometry import CrossSectionGeometry

# ============================ 配置 ============================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "training_dataset"
CACHE_DIR = REPO_ROOT / "training_dataset_cache"
CROSS_SECTION_PATH = CONFIG.cross_section_path
MANNING_N = 0.016
SEED = 2032
READ_GEOMETRY_GRIDS = False
CONDITION_TIME_POINTS = 32
CONDITION_SPACE_POINTS = 16
CONDITION_DIM = 2 * CONDITION_TIME_POINTS + 2 * CONDITION_SPACE_POINTS
CODE_DIM = 32
HIDDEN_DIM = 64


@dataclass
class TrainOptions:
    """训练参数；修改实验时优先只改这里。"""

    # 训练过程
    epochs: int = 200
    physics_batch_size: int = 256
    scenarios_per_batch: int = 8
    eval_every: int = 500
    save_every: int = 500

    # 损失权重
    initial_weight: float = 1.0
    boundary_weight: float = 1.0
    data_weight: float = 1.0
    physics_weight: float = 1.0e-3
    physics_warmup_epochs: int = 1000
    mass_weight: float = 1.0
    momentum_weight: float = 1.0

    # 优化器
    grad_clip: float = 1.0
    learning_rate: float = 1.0e-3

    # 内部断面监督
    data_section_indices: object = None


# ============================ 基础工具 ============================
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def fit_normalizer_and_scales(
    dataset,
    files,
    time_points=CONDITION_TIME_POINTS,
    space_points=CONDITION_SPACE_POINTS,
):
    """
    一次扫描训练文件，同时完成：
    1. condition mean/std
    2. Z/Q 输出 mean/std
    这样避免 ConditionNormalizer 和 DataScales 各自再扫一遍全部 CSV。
    """
    first = dataset.load(files[0])
    condition_values = []
    z_sum = 0.0
    z_square_sum = 0.0
    z_count = 0
    q_sum = 0.0
    q_square_sum = 0.0
    q_count = 0
    for path in files:
        scenario = dataset.load(path)
        condition_values.append(
            dataset.condition_vector(path, time_points=time_points, space_points=space_points)
        )
        z = scenario.z_grid.astype(np.float64, copy=False)
        q = scenario.q_grid.astype(np.float64, copy=False)
        z_sum += z.sum()
        z_square_sum += np.square(z).sum()
        z_count += z.size
        q_sum += q.sum()
        q_square_sum += np.square(q).sum()
        q_count += q.size
    condition_values = np.stack(condition_values, axis=0)
    condition_mean = condition_values.mean(axis=0).astype(np.float32)
    condition_std = condition_values.std(axis=0).astype(np.float32)
    # condition_std = np.maximum(condition_std, 1.0e-6)
    scales = DataScales.from_statistics(
        first, z_sum, z_square_sum, z_count,
        q_sum, q_square_sum, q_count,
    )
    normalizer = ConditionNormalizer(condition_mean, condition_std)
    return normalizer, scales


def to_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32)


# ============================ 数据读取与条件编码 ============================
def read_one_scenario(path, warmup_days=3.0):
    """文件读取与整理，其中 warmup 取为模型热启动/调整期"""
    cache_path = CACHE_DIR / f"{Path(path).stem}_warmup{warmup_days:g}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return (
            cached["times"], cached["stations"], cached["z_grid"], cached["q_grid"],
            None, None, None,
        )
    usecols = ["time_days", "river_station", "water_surface_m", "flow_m3s"]
    dtype = {
        "time_days": "float32",
        "river_station": "float32",
        "water_surface_m": "float32",
        "flow_m3s": "float32",
    }
    if READ_GEOMETRY_GRIDS:
        usecols += [
            "velocity_total_ms",
            "area_flow_total_m2",
            "top_width_total_m",
        ]
        dtype.update({
            "velocity_total_ms": "float32",
            "area_flow_total_m2": "float32",
            "top_width_total_m": "float32",
        })
    frame = pd.read_csv(path, usecols=usecols, dtype=dtype)
    frame = frame[frame["time_days"] >= warmup_days].copy()
    frame["time_days"] = frame["time_days"] - warmup_days
    times = np.sort(frame["time_days"].unique())
    stations = np.sort(frame["river_station"].unique())[::-1]
    def pivot(column):
        table = (
            frame.pivot(
                index="time_days",
                columns="river_station",
                values=column,
            ).reindex(index=times, columns=stations)
        )
        return table.to_numpy(dtype=np.float32)
    z_grid = pivot("water_surface_m")       # 水位
    q_grid = pivot("flow_m3s")              # 流量
    if READ_GEOMETRY_GRIDS:
        u_grid = pivot("velocity_total_ms")     # 流速
        area_grid = pivot("area_flow_total_m2") # 过流面积
        width_grid = pivot("top_width_total_m") # 河宽
    else:
        u_grid = None
        area_grid = None
        width_grid = None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        times=times.astype(np.float32),
        stations=stations.astype(np.float32),
        z_grid=z_grid,
        q_grid=q_grid,
    )
    return times, stations, z_grid, q_grid, u_grid, area_grid, width_grid

class ScenarioData:
    def __init__(self, path, warmup_days=3.0):
        self.path = path
        self.name = path.name.split("_", 1)[0]
        (
            times, stations, z_grid, q_grid, 
            u_grid, area_grid, width_grid, 
        ) = read_one_scenario(path, warmup_days=warmup_days)
        t_s = (times * 24.0 * 3600.0).astype(np.float32)
        x_m = ((stations[0] - stations) * 1000.0).astype(np.float32)
        self.times = times
        self.stations = stations
        self.x_m = x_m
        self.t_s = t_s
        self.z_grid = z_grid
        self.q_grid = q_grid
        self.area_grid = area_grid
        self.width_grid = width_grid
        self.u_grid = u_grid
        if area_grid is None or width_grid is None:
            self.h_grid = None
        else:
            self.h_grid = area_grid / np.maximum(width_grid, 1.0e-6)
        self.nt = len(t_s)
        self.nx = len(x_m)
        self.duration_s = t_s[-1] - t_s[0]
        self.length_m = x_m[-1] - x_m[0]
        self.supervised_cache = {}

class ScenarioDataset:
    def __init__(self, dataset_dir, warmup_days=3.0, test_every=10, cache_size=64):
        self.dataset_dir = Path(dataset_dir)
        self.warmup_days = warmup_days
        self.test_every = test_every
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.condition_cache = {}
        self.files = sorted(
            self.dataset_dir.glob("S*_hydrodynamics_all_sections_15min.csv")
        )
        self.names = [path.name.split("_", 1)[0] for path in self.files]
        self.name_to_path = {
            path.name.split("_", 1)[0]: path for path in self.files
        }
        self.train_files = []
        self.test_files = []
        for i, path in enumerate(self.files, start=1):
            if i % self.test_every == 0:
                self.test_files.append(path)
            else:
                self.train_files.append(path)

    def load(self, path):
        path = Path(path)
        key = path.name
        if key in self.cache:
            scenario = self.cache.pop(key)
            self.cache[key] = scenario
            return scenario
        scenario = ScenarioData(path, warmup_days=self.warmup_days)
        self.cache[key] = scenario
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return scenario

    def load_by_name(self, name):
        if name not in self.name_to_path:
            raise ValueError(f"没有找到 scenario: {name}")
        return self.load(self.name_to_path[name])

    def condition_vector(
        self,
        path,
        time_points=CONDITION_TIME_POINTS,
        space_points=CONDITION_SPACE_POINTS,
    ):
        key = (Path(path).name, time_points, space_points)
        if key not in self.condition_cache:
            scenario = self.load(path)
            self.condition_cache[key] = scenario_condition_vector(
                scenario, time_points=time_points, space_points=space_points
            )
        return self.condition_cache[key]

def scenario_condition_vector(
    scenario,
    time_points=CONDITION_TIME_POINTS,
    space_points=CONDITION_SPACE_POINTS,
):
    """
    构造一个 scenario 的算子条件编码输入。
    不使用人工统计特征，只使用原始函数的固定采样点。
    """
    q_up = scenario.q_grid[:, 0]    # 上游流量边界过程
    z_down = scenario.z_grid[:, -1] # 下游水位边界过程
    z0 = scenario.z_grid[0, :]  # 初始水位场
    q0 = scenario.q_grid[0, :]  # 初始流量场
    time_index = np.linspace(0, len(q_up) - 1, time_points).round().astype(int)
    space_index = np.linspace(0, len(z0) - 1, space_points).round().astype(int)
    q_up_sample = q_up[time_index]
    z_down_sample = z_down[time_index]
    z0_sample = z0[space_index]
    q0_sample = q0[space_index]
    condition = np.concatenate([
        q_up_sample,
        z_down_sample,
        z0_sample,
        q0_sample,
    ])
    return condition

class ConditionNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, condition):
        return (condition - self.mean) / self.std


# ============================ 算子网络 ============================
class OperatorPINN(nn.Module):
    def __init__(
        self,
        condition_dim,
        scales, 
        code_dim=CODE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=2,
        geometry=None,
        depth_floor=0.05,
        depth_scale=5.0,
    ):
        super().__init__()
        self.scales = scales
        self.geometry = geometry
        self.depth_floor = depth_floor  # 最小水深，防止水深为 0
        self.depth_scale = depth_scale  # 控制网络输出水深的变化范围
        self.branch = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(code_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, t, condition):
        """
        x: [n]
        t: [n]
        condition: [condition_dim] 或 [n, condition_dim]
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).expand(x.shape[0], -1)
        x_hat = self.scales.normalize_x(x)
        t_hat = self.scales.normalize_t(t)
        xt = torch.stack([x_hat, t_hat], dim=1)
        branch_code = self.branch(condition)
        trunk_code = self.trunk(xt)
        features = torch.cat([trunk_code, branch_code], dim=1)
        output = self.head(features)
        z_raw = output[:, 0]
        q_hat = output[:, 1]
        if self.geometry is None:
            raise RuntimeError("OperatorPINN 需要断面 geometry 才能计算有效水位")
        bed_level, _ = self.geometry.stage_bounds(x)
        # 以米为单位生成正水深；不再用全局水位均值初始化，避免局部断面干涸。
        depth = self.depth_floor + self.depth_scale * torch.nn.functional.softplus(z_raw)
        z = bed_level + depth
        q = self.scales.denormalize_q(q_hat)
        return z, q

# ============================ 物理量尺度 ============================
class DataScales:
    @classmethod
    def from_statistics(
        cls,
        scenario,
        z_sum,
        z_square_sum,
        z_count,
        q_sum,
        q_square_sum,
        q_count,
    ):
        obj = cls.__new__(cls)
        obj.x_min = scenario.x_m[0]
        obj.x_max = scenario.x_m[-1]
        obj.t_min = scenario.t_s[0]
        obj.t_max = scenario.t_s[-1]
        obj.z_mean = z_sum / z_count
        z_var = z_square_sum / z_count - obj.z_mean ** 2
        obj.z_std = max(float(np.sqrt(max(z_var, 1.0e-12))), 1.0e-6)
        obj.q_mean = q_sum / q_count
        q_var = q_square_sum / q_count - obj.q_mean ** 2
        obj.q_std = max(float(np.sqrt(max(q_var, 1.0e-12))), 1.0e-6)
        obj.length_m = float(scenario.length_m)
        obj.depth_m = float(obj.z_std)
        obj.discharge_m3_s = float(max(abs(obj.q_mean), obj.q_std, 1.0e-6))
        return obj
    def normalize_x(self, x):
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
    def normalize_t(self, t):
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
    def denormalize_z(self, z_hat):
        return z_hat * self.z_std + self.z_mean
    def denormalize_q(self, q_hat):
        return q_hat * self.q_std + self.q_mean

def scales_to_dict(scales):
    return {
        "x_min": float(scales.x_min),
        "x_max": float(scales.x_max),
        "t_min": float(scales.t_min),
        "t_max": float(scales.t_max),
        "z_mean": float(scales.z_mean),
        "z_std": float(scales.z_std),
        "q_mean": float(scales.q_mean),
        "q_std": float(scales.q_std),
        "length_m": float(scales.length_m),
        "depth_m": float(scales.depth_m),
        "discharge_m3_s": float(scales.discharge_m3_s),
    }

# ============================ 监督数据与 PDE ============================
def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_metric,
    normalizer,
    scales,
    train_options,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_test_q_nse": best_metric,
            "condition_mean": normalizer.mean,
            "condition_std": normalizer.std,
            "scales": scales_to_dict(scales),
            "train_options": train_options,
        },
        path,
    )

def supervised_data(scenario, kind, section_indices=None):
    """返回初始、边界或内部断面监督数据。"""
    if kind == "initial":
        key = "initial"
        if key not in scenario.supervised_cache:
            scenario.supervised_cache[key] = (
                scenario.x_m,
                np.zeros_like(scenario.x_m),
                scenario.z_grid[0],
                scenario.q_grid[0],
            )
        return scenario.supervised_cache[key]
    if kind == "boundary":
        key = "boundary"
        if key not in scenario.supervised_cache:
            t = scenario.t_s
            scenario.supervised_cache[key] = (
                np.concatenate([np.full_like(t, scenario.x_m[0]), np.full_like(t, scenario.x_m[-1])]),
                np.tile(t, 2),
                np.concatenate([scenario.z_grid[:, 0], scenario.z_grid[:, -1]]),
                np.concatenate([scenario.q_grid[:, 0], scenario.q_grid[:, -1]]),
            )
        return scenario.supervised_cache[key]
    if section_indices is None:
        fractions = np.linspace(0.25, 0.75, 3)
        section_indices = np.round(fractions * (scenario.nx - 1)).astype(int)
    section_indices = np.asarray(section_indices, dtype=int)
    if np.any(section_indices <= 0) or np.any(section_indices >= scenario.nx - 1):
        raise ValueError("内部监督断面不能包含上游/下游边界断面")
    key = ("sections", tuple(section_indices.tolist()))
    if key not in scenario.supervised_cache:
        x_grid, t_grid = np.meshgrid(
            scenario.x_m[section_indices], scenario.t_s, indexing="xy"
        )
        scenario.supervised_cache[key] = (
            x_grid.reshape(-1), t_grid.reshape(-1),
            scenario.z_grid[:, section_indices].reshape(-1),
            scenario.q_grid[:, section_indices].reshape(-1),
        )
    return scenario.supervised_cache[key]

def supervised_loss(model, scales, condition, batch):
    """计算一组监督点的 Z/Q 无量纲 MSE。"""
    x_np, t_np, z_np, q_np = batch
    z_pred, q_pred = model(to_tensor(x_np), to_tensor(t_np), condition)
    z_true = to_tensor(z_np)
    q_true = to_tensor(q_np)
    z_loss = ((z_pred - z_true)).square().mean()
    q_loss = ((q_pred - q_true)).square().mean()
    return z_loss + q_loss, z_loss, q_loss

def sample_physics_batch(scenario, batch_size):
    x = scenario.x_m[0] + (scenario.x_m[-1] - scenario.x_m[0]) * np.random.rand(
        batch_size
    ).astype(np.float32)
    t = scenario.t_s[0] + (scenario.t_s[-1] - scenario.t_s[0]) * np.random.rand(
        batch_size
    ).astype(np.float32)
    return x, t

class OperatorPDEResidual(nn.Module):
    """
    对条件算子网络计算 1D Saint-Venant PDE residual。
    输入仍是 x,t,condition。
    输出是无量纲 mass/momentum loss。
    """
    def __init__(
        self,
        geometry,
        scales,
        manning_n=MANNING_N,
        gravity=9.81,
        epsilon=1.0e-6,
    ):
        super().__init__()
        self.geometry = geometry
        self.scales = scales
        self.manning_n = manning_n
        self.gravity = gravity
        self.epsilon = epsilon
        (
            self.area_m2,
            self.velocity_m_s,
            self.friction_slope,
            self.momentum_scale,
        ) = self._estimate_physics_scales()
    def _estimate_physics_scales(self):
        x = torch.linspace(
            float(self.scales.x_min),
            float(self.scales.x_max),
            steps=256,
            dtype=torch.float32,
        )
        z = torch.full_like(x, float(self.scales.z_mean))
        area, _, perimeter = self.geometry(x, z, None)
        area0 = float(area.mean().clamp_min(self.epsilon))
        radius0 = float((area / perimeter.clamp_min(self.epsilon)).mean().clamp_min(self.epsilon))
        q0 = float(self.scales.discharge_m3_s)
        velocity0 = q0 / max(area0, self.epsilon)
        friction_slope = (
            self.manning_n**2
            * q0**2
            / (area0**2 * radius0 ** (4.0 / 3.0))
        )
        inertial_scale = q0 * abs(velocity0) / max(self.scales.length_m, self.epsilon)
        pressure_scale = (
            self.gravity
            * area0
            * max(float(self.scales.depth_m), self.epsilon)
            / max(self.scales.length_m, self.epsilon)
        )
        friction_scale = self.gravity * area0 * max(float(friction_slope), self.epsilon)
        momentum_scale = max(inertial_scale, pressure_scale, friction_scale, self.epsilon)
        return area0, abs(float(velocity0)), max(float(friction_slope), self.epsilon), momentum_scale

    def residual(self, model, x, t, condition):
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)
        water_level, discharge = model(x, t, condition)
        area, _, perimeter = self.geometry(x, water_level, t)
        area_t = grad(area, t)
        discharge_t = grad(discharge, t)
        discharge_x = grad(discharge, x)
        flux_adv_x = grad(discharge.square() / area.clamp_min(self.epsilon), x)
        water_level_x = grad(water_level, x)
        radius = area / perimeter.clamp_min(self.epsilon)
        friction = (
            self.manning_n**2
            * discharge
            * discharge.abs() / (
                area.square().clamp_min(self.epsilon)
                * radius.clamp_min(self.epsilon).pow(4.0 / 3.0)
            )
        )
        mass = area_t + discharge_x
        momentum = (
            discharge_t
            + flux_adv_x
            + self.gravity * area * water_level_x
            + self.gravity * area * friction
        )
        # mass_scale = self.scales.discharge_m3_s / max(self.scales.length_m, self.epsilon)
        # momentum_scale = self.momentum_scale
        # mass = mass / max(mass_scale, self.epsilon)
        # momentum = momentum / max(momentum_scale, self.epsilon)
        return mass, momentum

    def loss(self, model, x, t, condition):
        mass, momentum = self.residual(model, x, t, condition)
        return mass.square().mean(), momentum.square().mean()
    
# ============================ 评估与训练 ============================
@torch.no_grad()
def evaluate(
    model,
    dataset,
    normalizer,
    files,
    scenario_count=32,
    points_per_scenario=4096,
):
    """
    对若干 scenario 随机采样评估。
    不扫全场，避免评估拖慢训练。
    """
    model.eval()
    if len(files) > scenario_count:
        selected_files = np.random.choice(files, size=scenario_count, replace=False)
    else:
        selected_files = files
    z_predictions = []
    q_predictions = []
    z_targets = []
    q_targets = []
    for path in selected_files:
        scenario = dataset.load(path)
        condition = dataset.condition_vector(path)
        condition_norm = normalizer.transform(condition)
        time_index = np.random.randint(0, scenario.nt, size=points_per_scenario)
        space_index = np.random.randint(0, scenario.nx, size=points_per_scenario)
        x_np = scenario.x_m[space_index]
        t_np = scenario.t_s[time_index]
        z_np = scenario.z_grid[time_index, space_index]
        q_np = scenario.q_grid[time_index, space_index]
        z_pred, q_pred = model(
            to_tensor(x_np), to_tensor(t_np), to_tensor(condition_norm)
        )
        z_predictions.append(z_pred.cpu().numpy())
        q_predictions.append(q_pred.cpu().numpy())
        z_targets.append(z_np)
        q_targets.append(q_np)
    z_pred = np.concatenate(z_predictions)
    q_pred = np.concatenate(q_predictions)
    z_true = np.concatenate(z_targets)
    q_true = np.concatenate(q_targets)
    z_den = max(np.linalg.norm(z_true), 1.0e-12)
    q_den = max(np.linalg.norm(q_true), 1.0e-12)
    z_var = np.sum((z_true - z_true.mean()) ** 2)
    q_var = np.sum((q_true - q_true.mean()) ** 2)
    return {
        "z_l2": 100.0 * np.linalg.norm(z_pred - z_true) / z_den,
        "q_l2": 100.0 * np.linalg.norm(q_pred - q_true) / q_den,
        "z_nse": np.nan if z_var < 1.0e-12 else 1.0 - np.sum((z_pred - z_true) ** 2) / z_var,
        "q_nse": np.nan if q_var < 1.0e-12 else 1.0 - np.sum((q_pred - q_true) ** 2) / q_var,
        "scenario_count": len(selected_files),
    }

# 计算单个流量场景的损失
def scenario_loss(
        model,
        pde_residual,
        scenario,
        condition,
        scales,
        options,
        epoch,
    ):
    """计算一个流量场景的 IC、BC、内部数据和 PDE 损失。"""
    # 初始损失
    initial_loss, initial_z_loss, initial_q_loss = supervised_loss(
        model, scales, condition, supervised_data(scenario, "initial")
    )
    # 边界损失
    boundary_loss, boundary_z_loss, boundary_q_loss = supervised_loss(
        model, scales, condition, supervised_data(scenario, "boundary")
    )
    # 内部数据损失
    section_data = supervised_data(
        scenario, "sections", section_indices=options.data_section_indices
    )
    data_loss, data_z_loss, data_q_loss = supervised_loss(
        model, scales, condition, section_data
    )
    # PDE 物理损失
    x_phys_np, t_phys_np = sample_physics_batch(
        scenario,
        batch_size=options.physics_batch_size,
    )
    mass_loss, momentum_loss = pde_residual.loss(
        model, to_tensor(x_phys_np), to_tensor(t_phys_np), condition
    )
    physics_loss = (
        options.mass_weight * torch.log1p(mass_loss)
        + options.momentum_weight * torch.log1p(momentum_loss)
    )

    physics_factor = 1 # min(1.0, epoch / max(1, options.physics_warmup_epochs))
    effective_physics_weight = options.physics_weight * physics_factor

    total_loss = (
        options.initial_weight * initial_loss
        + options.boundary_weight * boundary_loss
        + options.data_weight * data_loss
        + effective_physics_weight * physics_loss
    )

    return total_loss, {
        "initial": initial_loss,
        "boundary": boundary_loss,
        "data": data_loss,
        "initial_z": initial_z_loss,
        "initial_q": initial_q_loss,
        "boundary_z": boundary_z_loss,
        "boundary_q": boundary_q_loss,
        "data_z": data_z_loss,
        "data_q": data_q_loss,
        "mass": mass_loss,
        "momentum": momentum_loss,
        "effective_physics_weight": effective_physics_weight,
    }

def train_one_epoch(model,pde_residual,dataset,normalizer,scales,optimizer,options,epoch):
    """遍历全部训练场景一次，并完成本 epoch 的参数更新。"""
    model.train()
    names = (
        "loss", "initial", "boundary", "data", "initial_z", "initial_q",
        "boundary_z", "boundary_q", "data_z", "data_q", "mass", "momentum",
    ) 
    epoch_values = {name: [] for name in names}
    shuffled_files = np.random.permutation(dataset.train_files) # 
    batch_count = 0
    effective_physics_weight = 0.0
    for start in range(0, len(shuffled_files), options.scenarios_per_batch):
        selected_files = shuffled_files[start:start + options.scenarios_per_batch]
        batch_count += 1
        optimizer.zero_grad()
        batch_losses = []
        for path in selected_files:
            scenario = dataset.load(path)
            condition = to_tensor(normalizer.transform(dataset.condition_vector(path)))

            # 计算损失
            total_loss, losses = scenario_loss(model, pde_residual, scenario, condition, scales, options, epoch)

            batch_losses.append(total_loss)

            for name in names[1:]:
                epoch_values[name].append(losses[name].detach())

            effective_physics_weight = losses["effective_physics_weight"]

        batch_loss = torch.stack(batch_losses).mean()
        batch_loss.backward()
        if options.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        epoch_values["loss"].append(batch_loss.detach())
    return epoch_values, batch_count, effective_physics_weight

def train_operator_pinn(
        model,
        pde_residual,
        dataset,
        normalizer,
        scales,
        output_dir,
        options=None,
    ):
    options = options or TrainOptions()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    train_files = dataset.train_files
    output_dir = Path(output_dir)
    history_path = output_dir / "history.csv"
    best_metric = -float("inf")
    best_epoch = 0
    train_options = {
        "epochs": options.epochs,
        "data_section_indices": (
            None
            if options.data_section_indices is None
            else list(options.data_section_indices)
        ),
        "physics_batch_size": options.physics_batch_size,
        "scenarios_per_batch": options.scenarios_per_batch,
        "batches_per_epoch": int(np.ceil(len(train_files) / options.scenarios_per_batch)),
        "eval_every": options.eval_every,
        "save_every": options.save_every,
        "initial_weight": options.initial_weight,
        "boundary_weight": options.boundary_weight,
        "data_weight": options.data_weight,
        "physics_weight": options.physics_weight,
        "physics_warmup_epochs": options.physics_warmup_epochs,
        "mass_weight": options.mass_weight,
        "momentum_weight": options.momentum_weight,
        "grad_clip": options.grad_clip,
        "learning_rate": options.learning_rate,
    }
    history_file = history_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "epoch", "loss", "scenario_count", "batch_count",
        "initial_loss", "boundary_loss", "data_loss",
        "initial_z_loss", "initial_q_loss", "boundary_z_loss",
        "boundary_q_loss", "data_z_loss", "data_q_loss",
        "mass_loss", "momentum_loss", "effective_physics_weight",
        "train_z_l2", "train_q_l2", "train_z_nse", "train_q_nse",
        "test_z_l2", "test_q_l2", "test_z_nse", "test_q_nse",
        "best_test_q_nse", "best_epoch",
    ]

    writer = csv.DictWriter(history_file, fieldnames=fieldnames)
    writer.writeheader()
    for epoch in range(1, options.epochs + 1):
        epoch_values, batch_count, effective_physics_weight = train_one_epoch(
            model, pde_residual, dataset, normalizer, scales, optimizer, options, epoch
        )
        should_print_loss = epoch == 1 or epoch % 100 == 0
        should_evaluate = (
            epoch == 1
            or epoch % options.eval_every == 0
            or epoch == options.epochs
        )
        def mean_epoch_value(name):
            return torch.stack(epoch_values[name]).mean().item()
        loss_value = mean_epoch_value("loss")
        initial_loss_value = mean_epoch_value("initial")
        boundary_loss_value = mean_epoch_value("boundary")
        data_loss_value = mean_epoch_value("data")
        initial_z_loss_value = mean_epoch_value("initial_z")
        initial_q_loss_value = mean_epoch_value("initial_q")
        boundary_z_loss_value = mean_epoch_value("boundary_z")
        boundary_q_loss_value = mean_epoch_value("boundary_q")
        data_z_loss_value = mean_epoch_value("data_z")
        data_q_loss_value = mean_epoch_value("data_q")
        mass_loss_value = mean_epoch_value("mass")
        momentum_loss_value = mean_epoch_value("momentum")
        if should_print_loss:
            print(
                f"epoch {epoch:4d} | "  # 当前训练轮数
                f"loss {loss_value:.4e} | " # 当前总损失
                f"IC {initial_loss_value:.2e} | "   # 初始条件损失
                f"BC {boundary_loss_value:.2e} | "  # 边界条件损失
                f"Data {data_loss_value:.2e} | "    # 内部数据损失
                f"mass {mass_loss_value:.2e} | "    # PDE 质量守恒损失
                f"mom {momentum_loss_value:.2e} | " # PDE 动量守恒损失
                f"pw {effective_physics_weight:.1e} | " # 当前有效 PDE 权重
                f"scenarios {len(train_files)} | "  # 本 epoch 使用的训练场景数量
                f"batches {batch_count}",   # 本 epoch 分成的批次数
                flush=True,
            )
        train_metrics = None
        test_metrics = None
        if should_evaluate:
            train_metrics = evaluate(
                model, dataset, normalizer, dataset.train_files,
                scenario_count=32, points_per_scenario=4096,
            )
            test_metrics = evaluate(
                model, dataset, normalizer, dataset.test_files,
                scenario_count=len(dataset.test_files),
                points_per_scenario=4096,
            )
            print(
                f"eval  {epoch:5d} | "
                f"train L2 Z/Q {train_metrics['z_l2']:5.2f}/"
                f"{train_metrics['q_l2']:5.2f}% "
                f"NSE Z/Q {train_metrics['z_nse']:6.3f}/"
                f"{train_metrics['q_nse']:6.3f} | "
                f"test L2 Z/Q {test_metrics['z_l2']:5.2f}/"
                f"{test_metrics['q_l2']:5.2f}% "
                f"NSE Z/Q {test_metrics['z_nse']:6.3f}/"
                f"{test_metrics['q_nse']:6.3f}",
                flush=True,
            )
            if test_metrics["q_nse"] > best_metric:
                best_metric = test_metrics["q_nse"]
                best_epoch = epoch
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, epoch,
                    best_metric, normalizer, scales, train_options,
                )
        if should_print_loss or should_evaluate:
            row = {
                "epoch": epoch,
                "loss": loss_value,
                "scenario_count": len(train_files),
                "batch_count": batch_count,
                "initial_loss": initial_loss_value,
                "boundary_loss": boundary_loss_value,
                "data_loss": data_loss_value,
                "initial_z_loss": initial_z_loss_value,
                "initial_q_loss": initial_q_loss_value,
                "boundary_z_loss": boundary_z_loss_value,
                "boundary_q_loss": boundary_q_loss_value,
                "data_z_loss": data_z_loss_value,
                "data_q_loss": data_q_loss_value,
                "mass_loss": mass_loss_value,
                "momentum_loss": momentum_loss_value,
                "effective_physics_weight": effective_physics_weight,
                "best_test_q_nse": best_metric,
                "best_epoch": best_epoch,
            }
            if train_metrics is not None and test_metrics is not None:
                row.update({
                    "train_z_l2": train_metrics["z_l2"],
                    "train_q_l2": train_metrics["q_l2"],
                    "train_z_nse": train_metrics["z_nse"],
                    "train_q_nse": train_metrics["q_nse"],
                    "test_z_l2": test_metrics["z_l2"],
                    "test_q_l2": test_metrics["q_l2"],
                    "test_z_nse": test_metrics["z_nse"],
                    "test_q_nse": test_metrics["q_nse"],
                })
            writer.writerow(row)
            history_file.flush()
        if epoch % options.save_every == 0 or epoch == options.epochs:
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, epoch,
                best_metric, normalizer, scales, train_options,
            )
    history_file.close()
    print(
        f"finished | best test Q-NSE {best_metric:.4f} @ epoch {best_epoch} | "
        f"output {output_dir}",
        flush=True,
    )

# ============================ 实验入口 ============================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dataset = ScenarioDataset(DATASET_DIR, warmup_days=3.0, test_every=10, cache_size=512)

    # 测试
    # 小批量测试
    dataset.train_files = dataset.train_files[:8]
    dataset.test_files = dataset.test_files[:2]


    sample_scenario = dataset.load(dataset.train_files[0])
    # 选择 3 个内部断面作为监督数据
    data_section_indices = np.round(
        np.linspace(0.25, 0.75, 3) * (sample_scenario.nx - 1)
    ).astype(int)

    normalizer, scales = fit_normalizer_and_scales(
        dataset,
        dataset.train_files,
        time_points=CONDITION_TIME_POINTS,
        space_points=CONDITION_SPACE_POINTS,
    )

    geometry = CrossSectionGeometry(
        CROSS_SECTION_PATH, to_tensor(sample_scenario.x_m)
    )

    model = OperatorPINN(
        CONDITION_DIM,
        scales,
        code_dim=CODE_DIM,
        hidden_dim=HIDDEN_DIM,
        geometry=geometry,
    )

    pde_residual = OperatorPDEResidual(geometry=geometry, scales=scales)

    output_root = REPO_ROOT / "outputs" / "flow_generalization_operator"
    output_dir = output_root / ("operator_pinn_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)

    experiment_config = {
        "seed": SEED,
        "dataset_dir": str(DATASET_DIR),
        "cache_dir": str(CACHE_DIR),
        "cross_section_path": str(CROSS_SECTION_PATH),
        "warmup_days": dataset.warmup_days,
        "test_every": dataset.test_every,
        "scenario_count": len(dataset.files),
        "train_count": len(dataset.train_files),
        "test_count": len(dataset.test_files),
        "condition_dim": CONDITION_DIM,
        "condition_time_points": CONDITION_TIME_POINTS,
        "condition_space_points": CONDITION_SPACE_POINTS,
        "data_section_indices": data_section_indices.tolist(),
        "data_section_x_m": sample_scenario.x_m[data_section_indices].tolist(),
        "code_dim": 32,
        "hidden_dim": 64,
        "depth_floor": model.depth_floor,
        "depth_scale": model.depth_scale,
        "manning_n": MANNING_N,
        "read_geometry_grids": READ_GEOMETRY_GRIDS,
        "scales": scales_to_dict(scales),
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(experiment_config, file, ensure_ascii=False, indent=2)

    # 测试
    # options = TrainOptions(data_section_indices=data_section_indices)
    options = TrainOptions(
            epochs=5,
            physics_batch_size=32,
            scenarios_per_batch=2,
            eval_every=1,
            save_every=5,
            data_section_indices=data_section_indices,
        )

    train_operator_pinn(model, pde_residual, dataset, normalizer, scales, output_dir,options=options)

if __name__ == "__main__":
    main()
