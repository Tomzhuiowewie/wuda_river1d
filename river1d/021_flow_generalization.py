from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import csv
import json
import pandas as pd
import numpy as np
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "training_dataset"
CACHE_DIR = REPO_ROOT / "training_dataset_cache"
CROSS_SECTION_PATH = CONFIG.cross_section_path
MANNING_N = 0.016
SEED = 2032

# 当前 data-only 算子网络只训练 Z/Q，默认不读取 U/A/B。
# 后续加入 PDE 或 H 指标时，再改成 True。
READ_GEOMETRY_GRIDS = False


def create_run_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(base_dir) / f"operator_pinn_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

def build_coordinates(times, stations):
    """
    times: time_days
    stations: HEC-RAS river_station，已按上游到下游降序排列
    """
    t_s = (times * 24.0 * 3600.0).astype(np.float32)
    x_m = ((stations[0] - stations) * 1000.0).astype(np.float32)
    return x_m, t_s

def fit_normalizer_and_scales(dataset, files, time_points=32, space_points=16):
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
            dataset.condition_vector(
                path,
                time_points=time_points,
                space_points=space_points,
            )
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
    condition_std = np.maximum(condition_std, 1.0e-6)

    scales = DataScales.from_statistics(
        first,
        z_sum,
        z_square_sum,
        z_count,
        q_sum,
        q_square_sum,
        q_count,
    )

    normalizer = ConditionNormalizer(condition_mean, condition_std)
    return normalizer, scales

def to_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32)


def relative_l2(pred, true):
    return 100.0 * np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1.0e-12)


def nse(pred, true):
    denominator = np.sum((true - true.mean()) ** 2)
    if denominator < 1.0e-12:
        return np.nan
    return 1.0 - np.sum((pred - true) ** 2) / denominator


def read_one_scenario(path, warmup_days=3.0):
    """文件读取与整理，其中 warmup 取为模型热启动/调整期"""
    cache_path = CACHE_DIR / f"{Path(path).stem}_warmup{warmup_days:g}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return (
            cached["times"],
            cached["stations"],
            cached["z_grid"],
            cached["q_grid"],
            None,
            None,
            None,
        )

    usecols = [
        "time_days",
        "river_station",
        "water_surface_m",
        "flow_m3s",
    ]
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

    frame = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
    )

    frame = frame[frame["time_days"] >= warmup_days].copy()
    frame["time_days"] = frame["time_days"] - warmup_days

    times = np.sort(frame["time_days"].unique())

    # HEC-RAS river_station 通常是上游大、下游小
    # 所以这里降序排列：第 0 个断面 = 上游，第 -1 个断面 = 下游
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

        x_m, t_s = build_coordinates(times, stations)

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

    def condition_vector(self, path, time_points=32, space_points=16):
        key = (Path(path).name, time_points, space_points)
        if key not in self.condition_cache:
            scenario = self.load(path)
            self.condition_cache[key] = scenario_condition_vector(
                scenario,
                time_points=time_points,
                space_points=space_points,
            )
        return self.condition_cache[key]


# 构造每个 scenario 的条件向量
def sample_1d(values, count):
    """
    从一维序列中均匀采样 count 个点。
    values: shape = [n]
    """
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return values[indices]


def scenario_condition_vector(
    scenario,
    time_points=32,
    space_points=16,
):
    """
    构造一个 scenario 的算子条件编码输入。
    不使用人工统计特征，只使用原始函数的固定采样点。
    """

    q_up = scenario.q_grid[:, 0]    # 上游流量边界过程
    z_down = scenario.z_grid[:, -1] # 下游水位边界过程

    z0 = scenario.z_grid[0, :]  # 初始水位场
    q0 = scenario.q_grid[0, :]  # 初始流量场

    q_up_sample = sample_1d(q_up, time_points)
    z_down_sample = sample_1d(z_down, time_points)

    z0_sample = sample_1d(z0, space_points)
    q0_sample = sample_1d(q0, space_points)

    condition = np.concatenate([
        q_up_sample,
        z_down_sample,
        z0_sample,
        q0_sample,
    ])

    return condition


# 归一化
class ConditionNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, dataset, files, time_points=32, space_points=16):
        conditions = []

        for path in files:
            c = dataset.condition_vector(
                path,
                time_points=time_points,
                space_points=space_points,
            )
            conditions.append(c)

        conditions = np.stack(conditions, axis=0)

        mean = conditions.mean(axis=0)
        std = conditions.std(axis=0)

        std = np.maximum(std, 1.0e-6)

        return cls(mean, std)

    def transform(self, condition):
        return (condition - self.mean) / self.std


class OperatorPINN(nn.Module):
    def __init__(
        self,
        condition_dim,
        scales, 
        code_dim=32,
        hidden_dim=64,
        output_dim=2,
    ):
        super().__init__()

        self.scales = scales

        # Branch：编码整个 scenario 条件
        self.branch = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )

        # Trunk：编码坐标 x,t
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )

        # Head：融合坐标特征和条件特征，输出 Z,Q
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

        z_hat = output[:, 0]
        q_hat = output[:, 1]

        z = self.scales.denormalize_z(z_hat)
        q = self.scales.denormalize_q(q_hat)

        return z, q


class DataScales:
    def __init__(self, dataset):
        # 用第一个 scenario 获取坐标尺度
        scenario = dataset.load(dataset.train_files[0])

        self.x_min = scenario.x_m[0]
        self.x_max = scenario.x_m[-1]
        self.t_min = scenario.t_s[0]
        self.t_max = scenario.t_s[-1]

        z_values = []
        q_values = []

        # 用训练集估计 Z/Q 输出尺度
        for path in dataset.train_files:
            s = dataset.load(path)
            z_values.append(s.z_grid.reshape(-1))
            q_values.append(s.q_grid.reshape(-1))

        z_values = np.concatenate(z_values)
        q_values = np.concatenate(q_values)

        self.z_mean = z_values.mean()
        self.z_std = max(z_values.std(), 1.0e-6)

        self.q_mean = q_values.mean()
        self.q_std = max(q_values.std(), 1.0e-6)

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


def sample_data_batch(scenario, batch_size):
    """
    评估用：从完整 HEC-RAS 结果中随机采样点。
    训练损失不再用这个函数。
    """

    time_indices = np.random.randint(0, scenario.nt, size=batch_size)
    space_indices = np.random.randint(0, scenario.nx, size=batch_size)

    x = scenario.x_m[space_indices]
    t = scenario.t_s[time_indices]

    z = scenario.z_grid[time_indices, space_indices]
    q = scenario.q_grid[time_indices, space_indices]

    return x, t, z, q


def initial_condition_data(scenario):
    """初始条件：t=0，所有断面。"""
    if "initial" in scenario.supervised_cache:
        return scenario.supervised_cache["initial"]

    x = scenario.x_m
    t = np.zeros_like(scenario.x_m)
    z = scenario.z_grid[0, :]
    q = scenario.q_grid[0, :]
    batch = (x, t, z, q)
    scenario.supervised_cache["initial"] = batch
    return batch


def boundary_condition_data(scenario):
    """边界条件：所有时间的上游断面和下游断面。"""
    if "boundary" in scenario.supervised_cache:
        return scenario.supervised_cache["boundary"]

    t = scenario.t_s
    x_upstream = np.full_like(t, scenario.x_m[0])
    x_downstream = np.full_like(t, scenario.x_m[-1])

    x = np.concatenate([x_upstream, x_downstream])
    t = np.concatenate([t, t])
    z = np.concatenate([scenario.z_grid[:, 0], scenario.z_grid[:, -1]])
    q = np.concatenate([scenario.q_grid[:, 0], scenario.q_grid[:, -1]])
    batch = (x, t, z, q)
    scenario.supervised_cache["boundary"] = batch
    return batch


def default_data_section_indices(scenario, section_count=3):
    """
    默认选 3 个内部监督断面：约 1/4、1/2、3/4 河长位置。
    不包含上下游边界。
    """
    fractions = np.linspace(
        1.0 / (section_count + 1),
        section_count / (section_count + 1),
        section_count,
    )
    indices = np.round(fractions * (scenario.nx - 1)).astype(int)
    return np.clip(indices, 1, scenario.nx - 2)


def section_observation_data(scenario, section_indices=None):
    """
    数据监督点：固定若干内部断面，取这些断面的完整时间序列。
    """
    if section_indices is None:
        section_indices = default_data_section_indices(scenario)
    section_indices = np.asarray(section_indices, dtype=int)
    cache_key = ("sections", tuple(section_indices.tolist()))
    if cache_key in scenario.supervised_cache:
        return scenario.supervised_cache[cache_key]

    if np.any(section_indices <= 0) or np.any(section_indices >= scenario.nx - 1):
        raise ValueError("内部监督断面不能包含上游/下游边界断面")

    x_grid, t_grid = np.meshgrid(
        scenario.x_m[section_indices],
        scenario.t_s,
        indexing="xy",
    )
    z = scenario.z_grid[:, section_indices]
    q = scenario.q_grid[:, section_indices]

    batch = (
        x_grid.reshape(-1),
        t_grid.reshape(-1),
        z.reshape(-1),
        q.reshape(-1),
    )
    scenario.supervised_cache[cache_key] = batch
    return batch


def supervised_loss(model, scales, condition, batch):
    """计算一组监督点的 Z/Q 无量纲 MSE。"""
    x_np, t_np, z_np, q_np = batch

    z_pred, q_pred = model(
        to_tensor(x_np),
        to_tensor(t_np),
        condition,
    )
    z_true = to_tensor(z_np)
    q_true = to_tensor(q_np)

    z_loss = ((z_pred - z_true) / scales.z_std).square().mean()
    q_loss = ((q_pred - q_true) / scales.q_std).square().mean()
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
            * discharge.abs()
            / (
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

        mass_scale = self.scales.discharge_m3_s / max(self.scales.length_m, self.epsilon)
        momentum_scale = self.momentum_scale

        mass = mass / max(mass_scale, self.epsilon)
        momentum = momentum / max(momentum_scale, self.epsilon)
        return mass, momentum

    def loss(self, model, x, t, condition):
        mass, momentum = self.residual(model, x, t, condition)
        return mass.square().mean(), momentum.square().mean()


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

        x_np, t_np, z_np, q_np = sample_data_batch(
            scenario,
            batch_size=points_per_scenario,
        )

        z_pred, q_pred = model(
            to_tensor(x_np),
            to_tensor(t_np),
            to_tensor(condition_norm),
        )

        z_predictions.append(z_pred.cpu().numpy())
        q_predictions.append(q_pred.cpu().numpy())
        z_targets.append(z_np)
        q_targets.append(q_np)

    z_pred = np.concatenate(z_predictions)
    q_pred = np.concatenate(q_predictions)
    z_true = np.concatenate(z_targets)
    q_true = np.concatenate(q_targets)

    return {
        "z_l2": relative_l2(z_pred, z_true),
        "q_l2": relative_l2(q_pred, q_true),
        "z_nse": nse(z_pred, z_true),
        "q_nse": nse(q_pred, q_true),
        "scenario_count": len(selected_files),
    }


def train_operator_pinn(
    model,
    pde_residual,
    dataset,
    normalizer,
    scales,
    output_dir,
    epochs=1000,
    data_section_indices=None,
    physics_batch_size=512,
    scenarios_per_epoch=8,
    eval_every=500,
    save_every=500,
    initial_weight=1.0,
    boundary_weight=1.0,
    data_weight=1.0,
    physics_weight=1.0e-3,
    physics_warmup_epochs=1000,
    mass_weight=1.0,
    momentum_weight=1.0,
    grad_clip=1.0,
    lr=1.0e-3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_files = dataset.train_files
    output_dir = Path(output_dir)
    history_path = output_dir / "history.csv"
    best_metric = -float("inf")
    best_epoch = 0

    train_options = {
        "epochs": epochs,
        "data_section_indices": (
            None if data_section_indices is None else list(data_section_indices)
        ),
        "physics_batch_size": physics_batch_size,
        "scenarios_per_epoch": scenarios_per_epoch,
        "eval_every": eval_every,
        "save_every": save_every,
        "initial_weight": initial_weight,
        "boundary_weight": boundary_weight,
        "data_weight": data_weight,
        "physics_weight": physics_weight,
        "physics_warmup_epochs": physics_warmup_epochs,
        "mass_weight": mass_weight,
        "momentum_weight": momentum_weight,
        "grad_clip": grad_clip,
        "learning_rate": lr,
    }

    history_file = history_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "epoch",
        "loss",
        "initial_loss",
        "boundary_loss",
        "data_loss",
        "initial_z_loss",
        "initial_q_loss",
        "boundary_z_loss",
        "boundary_q_loss",
        "data_z_loss",
        "data_q_loss",
        "mass_loss",
        "momentum_loss",
        "effective_physics_weight",
        "train_z_l2",
        "train_q_l2",
        "train_z_nse",
        "train_q_nse",
        "test_z_l2",
        "test_q_l2",
        "test_z_nse",
        "test_q_nse",
        "best_test_q_nse",
        "best_epoch",
    ]
    writer = csv.DictWriter(history_file, fieldnames=fieldnames)
    writer.writeheader()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        epoch_losses = []
        epoch_initial_losses = []
        epoch_boundary_losses = []
        epoch_data_losses = []
        epoch_initial_z_losses = []
        epoch_initial_q_losses = []
        epoch_boundary_z_losses = []
        epoch_boundary_q_losses = []
        epoch_data_z_losses = []
        epoch_data_q_losses = []
        epoch_mass_losses = []
        epoch_momentum_losses = []

        # 每个 epoch 随机抽多个独立流量过程，取平均 loss。
        selected_files = np.random.choice(
            train_files,
            size=min(scenarios_per_epoch, len(train_files)),
            replace=False,
        )

        for path in selected_files:
            scenario = dataset.load(path)

            # 当前 scenario 的条件向量
            condition = dataset.condition_vector(path)
            condition_norm = normalizer.transform(condition)
            c = to_tensor(condition_norm)

            initial_loss, initial_z_loss, initial_q_loss = supervised_loss(
                model,
                scales,
                c,
                initial_condition_data(scenario),
            )
            boundary_loss, boundary_z_loss, boundary_q_loss = supervised_loss(
                model,
                scales,
                c,
                boundary_condition_data(scenario),
            )
            data_loss, data_z_loss, data_q_loss = supervised_loss(
                model,
                scales,
                c,
                section_observation_data(
                    scenario,
                    section_indices=data_section_indices,
                ),
            )

            x_phys_np, t_phys_np = sample_physics_batch(
                scenario,
                batch_size=physics_batch_size,
            )
            mass_loss, momentum_loss = pde_residual.loss(
                model,
                to_tensor(x_phys_np),
                to_tensor(t_phys_np),
                c,
            )

            physics_loss = (
                mass_weight * torch.log1p(mass_loss)
                + momentum_weight * torch.log1p(momentum_loss)
            )
            physics_factor = min(1.0, epoch / max(1, physics_warmup_epochs))
            effective_physics_weight = physics_weight * physics_factor
            loss = (
                initial_weight * initial_loss
                + boundary_weight * boundary_loss
                + data_weight * data_loss
                + effective_physics_weight * physics_loss
            )

            epoch_losses.append(loss)
            epoch_initial_losses.append(initial_loss.detach())
            epoch_boundary_losses.append(boundary_loss.detach())
            epoch_data_losses.append(data_loss.detach())
            epoch_initial_z_losses.append(initial_z_loss.detach())
            epoch_initial_q_losses.append(initial_q_loss.detach())
            epoch_boundary_z_losses.append(boundary_z_loss.detach())
            epoch_boundary_q_losses.append(boundary_q_loss.detach())
            epoch_data_z_losses.append(data_z_loss.detach())
            epoch_data_q_losses.append(data_q_loss.detach())
            epoch_mass_losses.append(mass_loss.detach())
            epoch_momentum_losses.append(momentum_loss.detach())

        loss = torch.stack(epoch_losses).mean()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        should_print_loss = epoch == 1 or epoch % 100 == 0
        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == epochs

        initial_loss_value = torch.stack(epoch_initial_losses).mean().item()
        boundary_loss_value = torch.stack(epoch_boundary_losses).mean().item()
        data_loss_value = torch.stack(epoch_data_losses).mean().item()
        initial_z_loss_value = torch.stack(epoch_initial_z_losses).mean().item()
        initial_q_loss_value = torch.stack(epoch_initial_q_losses).mean().item()
        boundary_z_loss_value = torch.stack(epoch_boundary_z_losses).mean().item()
        boundary_q_loss_value = torch.stack(epoch_boundary_q_losses).mean().item()
        data_z_loss_value = torch.stack(epoch_data_z_losses).mean().item()
        data_q_loss_value = torch.stack(epoch_data_q_losses).mean().item()
        mass_loss_value = torch.stack(epoch_mass_losses).mean().item()
        momentum_loss_value = torch.stack(epoch_momentum_losses).mean().item()

        if should_print_loss:
            print(
                f"epoch {epoch:5d} | "
                f"loss {loss.item():.4e} | "
                f"IC {initial_loss_value:.2e} | "
                f"BC {boundary_loss_value:.2e} | "
                f"Data {data_loss_value:.2e} | "
                f"mass {mass_loss_value:.2e} | "
                f"mom {momentum_loss_value:.2e} | "
                f"pw {effective_physics_weight:.1e} | "
                f"scenarios {len(selected_files)}"
                ,
                flush=True,
            )

        train_metrics = None
        test_metrics = None
        if should_evaluate:
            train_metrics = evaluate(
                model,
                dataset,
                normalizer,
                dataset.train_files,
                scenario_count=32,
                points_per_scenario=4096,
            )
            test_metrics = evaluate(
                model,
                dataset,
                normalizer,
                dataset.test_files,
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
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    best_metric,
                    normalizer,
                    scales,
                    train_options,
                )

        if should_print_loss or should_evaluate:
            row = {
                "epoch": epoch,
                "loss": loss.item(),
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

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                best_metric,
                normalizer,
                scales,
                train_options,
            )

    history_file.close()
    print(
        f"finished | best test Q-NSE {best_metric:.4f} @ epoch {best_epoch} | "
        f"output {output_dir}",
        flush=True,
    )


def main():
    set_seed(SEED)

    dataset = ScenarioDataset(
        dataset_dir=DATASET_DIR, warmup_days=3.0, test_every=10, cache_size=512
    )

    sample_scenario = dataset.load(dataset.train_files[0])
    data_section_indices = default_data_section_indices(
        sample_scenario,
        section_count=3,
    )

    normalizer, scales = fit_normalizer_and_scales(
        dataset, dataset.train_files, time_points=32, space_points=16
    )

    model = OperatorPINN(condition_dim=96, scales=scales, code_dim=32, hidden_dim=64)

    geometry = CrossSectionGeometry(
        CROSS_SECTION_PATH,
        to_tensor(sample_scenario.x_m),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    pde_residual = OperatorPDEResidual(geometry=geometry, scales=scales)

    output_dir = create_run_dir(REPO_ROOT / "outputs" / "flow_generalization_operator")

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
        "condition_dim": 96,
        "condition_time_points": 32,
        "condition_space_points": 16,
        "data_section_indices": data_section_indices.tolist(),
        "data_section_x_m": sample_scenario.x_m[data_section_indices].tolist(),
        "code_dim": 32,
        "hidden_dim": 64,
        "manning_n": MANNING_N,
        "read_geometry_grids": READ_GEOMETRY_GRIDS,
        "scales": scales_to_dict(scales),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(experiment_config, file, ensure_ascii=False, indent=2)

    train_operator_pinn(
        model, pde_residual, dataset, normalizer, scales,
        output_dir,
        epochs=2000,
        data_section_indices=data_section_indices,
        physics_batch_size=256,
        scenarios_per_epoch=8,
        eval_every=500,
        save_every=500,

        initial_weight=1.0,
        boundary_weight=1.0,
        data_weight=1.0,
        physics_weight=1.0e-3,
        physics_warmup_epochs=1000,
        mass_weight=1.0,
        momentum_weight=1.0,
        grad_clip=1.0,
        lr=1.0e-3,
    )


if __name__ == "__main__":
    main()