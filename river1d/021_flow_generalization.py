"""501 个 HEC-RAS 流量过程的条件算子 PINN

主流程只有：读取数据 -> 条件编码 -> 网络/PDE -> 训练评估
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from config import CONFIG
    from _geometry import CrossSectionGeometry
except ImportError:
    from .config import CONFIG
    from ._geometry import CrossSectionGeometry


# ============================== 配置 ==============================
project_root = Path(__file__).resolve().parent.parent
cache_directory = project_root / "training_dataset_cache"
cross_section_profile_path = CONFIG.cross_section_path
manning_roughness = 0.016
condition_time_points = 32
condition_space_points = 16
condition_input_dim = 2 * (condition_time_points + condition_space_points)


@dataclass
class TrainingOptions:
    epochs: int = 300
    scenarios_per_batch: int = 4
    physics_points: int = 512
    learning_rate: float = 1.0e-3
    grad_clip: float = 1.0
    # 监督点断面索引
    supervised_section_indices: tuple = (23, 46, 69)
    # 初始权重
    initial_weight: float = 1.0
    boundary_weight: float = 1.0
    data_weight: float = 1.0
    physics_weight: float = 1.0
    mass_weight: float = 1.0
    momentum_weight: float = 1.0
    # 适应性权重更新
    adaptive_weighting: bool = True
    # 第 1 个 epoch 结束后根据各项初始损失自动标定权重；此后先固定预热
    # 若干 epoch，再从固定阶段结束后进行 GradNorm 自适应更新。
    auto_initialize_weights: bool = True
    fixed_weight_epochs: int = 100
    weight_update_start: int = 101
    weight_update_every: int = 30
    weight_update_rate: float = 0.2
    min_loss_weight: float = 0.01
    max_loss_weight: float = 100.0

def to_tensor(value):
    """将输入数据转换为当前计算设备上的单精度张量"""
    return torch.as_tensor(value, dtype=torch.float32, device=compute_device)

def select_device():
    """选择可用的计算设备；当前 MPS 因几何插值兼容性问题回退到 CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("检测到 MPS，但当前 PDE 几何插值存在兼容性问题，回退 CPU")
        return torch.device("cpu")

    return torch.device("cpu")

compute_device = select_device()

# ============================== 数据 ==============================
class ScenarioDataset:
    """场景文件列表、训练/测试划分和内存缓存"""

    def __init__(self, cache_directory, test_every=10, cache_limit=16):
        """初始化场景文件列表、训练测试划分及场景缓存"""
        self.cache_limit = cache_limit
        self.scenario_files = sorted(Path(cache_directory).glob("S*_hydrodynamics_all_sections_15min_warmup3.npz"))
        self.train_files = [path for i, path in enumerate(self.scenario_files, 1) if i % test_every]
        self.test_files = [path for i, path in enumerate(self.scenario_files, 1) if not i % test_every]
        self.scenario_cache = {}
        self.condition_cache = {}

    def load_scenario(self, scenario_path):
        """加载指定场景，并在缓存容量超限时移除最早缓存项"""
        scenario_path = Path(scenario_path)
        if scenario_path not in self.scenario_cache:
            raw = np.load(scenario_path)
            stations = raw["stations"]
            self.scenario_cache[scenario_path] = {
                "t": raw["times"] * 24.0 * 3600.0,
                "x": (stations[0] - stations) * 1000.0,
                "z": raw["z_grid"],
                "q": raw["q_grid"],
            }
            if len(self.scenario_cache) > self.cache_limit:
                self.scenario_cache.pop(next(iter(self.scenario_cache)))

        return self.scenario_cache[scenario_path]

    def get_condition_vector(self, scenario_path):
        """提取指定场景的边界和初始条件并组成条件向量："""
        scenario_path = Path(scenario_path)
        if scenario_path not in self.condition_cache:
            scenario = self.load_scenario(scenario_path)
            time_indices = np.linspace(0, len(scenario["t"]) - 1, condition_time_points).round().astype(int)
            space_indices = np.linspace(0, len(scenario["x"]) - 1, condition_space_points).round().astype(int)
            # 上游边界流量过程、下游边界水位条件、初始时刻的水位空间分布、初始时刻的流量空间分布
            self.condition_cache[scenario_path] = np.concatenate((
                scenario["q"][:, 0][time_indices], scenario["z"][:, -1][time_indices],
                scenario["z"][0, :][space_indices], scenario["q"][0, :][space_indices],
            )).astype("float32")
        return self.condition_cache[scenario_path]


class ConditionStandardizer:
    def __init__(self, condition_vectors):
        """根据《条件样本》计算逐维标准化所需的均值和标准差"""
        self.mean = condition_vectors.mean(axis=0).astype("float32")
        self.std = np.maximum(condition_vectors.std(axis=0), 1.0e-6).astype("float32")

    def transform(self, condition_vector):
        """使用已计算的统计量标准化一个条件向量"""
        return (condition_vector - self.mean) / self.std

# ============================== 物理尺度 ==============================
class PhysicalScales:
    def __init__(self, reference_scenario, water_level_values, discharge_values):
        """根据参考场景及样本数据初始化空间、时间和物理量尺度"""
        self.x_min, self.x_max = float(reference_scenario["x"][0]), float(reference_scenario["x"][-1])
        self.t_min, self.t_max = float(reference_scenario["t"][0]), float(reference_scenario["t"][-1])
        self.length = self.x_max - self.x_min
        self.z_mean, self.z_std = float(water_level_values.mean()), float(water_level_values.std())
        self.q_mean, self.q_std = float(discharge_values.mean()), float(discharge_values.std())

    def x_normalize(self, x):
        """将空间坐标线性归一化到 [-1, 1]"""
        return 2.0 * (x - self.x_min) / self.length - 1.0

    def t_normalize(self, t):
        """将时间坐标线性归一化到 [-1, 1]"""
        return 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

    def q_denormalize(self, q):
        """将标准化流量还原为物理单位下的流量"""
        return q * self.q_std + self.q_mean

def fit_condition_normalizer_and_scales(dataset):
    """根据训练集计算：条件向量的均值和标准差，以及水位和流量的物理尺度"""
    condition_vectors, water_level_values, discharge_values = [], [], []
    for scenario_path in dataset.train_files:
        scenario = dataset.load_scenario(scenario_path)
        condition_vectors.append(dataset.get_condition_vector(scenario_path))   # 所有条件向量
        water_level_values.append(scenario["z"].reshape(-1))    # 所有水位
        discharge_values.append(scenario["q"].reshape(-1))      # 所有流量
    reference_scenario = dataset.load_scenario(dataset.train_files[0])

    return ConditionStandardizer(np.stack(condition_vectors)), PhysicalScales(
        reference_scenario,
        np.concatenate(water_level_values),
        np.concatenate(discharge_values),
    )


# ============================== 网络架构 ==============================
class OperatorPINN(nn.Module):
    def __init__(self, scales, geometry, code_dim=32, hidden=64):
        """初始化条件分支、时空分支及水位流量预测头"""
        super().__init__()
        self.scales, self.geometry = scales, geometry
        self.branch = nn.Sequential(
            nn.Linear(condition_input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, code_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, code_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * code_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )

    def forward(self, x, t, condition):
        """根据时空坐标和场景条件预测水位与流量"""
        if condition.ndim == 1:
            condition = condition.expand(x.numel(), -1)
        xt = torch.stack((self.scales.x_normalize(x), self.scales.t_normalize(t)), 1)
        code = torch.cat((self.trunk(xt), self.branch(condition)), 1)
        raw = self.head(code)
        bed, _ = self.geometry.stage_bounds(x)
        depth = 0.0005 + 5.0 * torch.nn.functional.softplus(raw[:, 0])
        return bed + depth, self.scales.q_denormalize(raw[:, 1])


# ============================== PDE ==============================
def derivative(output, variable):
    """计算输出相对于输入变量的一阶自动微分导数"""
    return torch.autograd.grad(
        output, variable, torch.ones_like(output), create_graph=True, retain_graph=True
    )[0]


class PDE:
    def __init__(self, geometry, manning_n=manning_roughness, gravity=9.81):
        """初始化基于断面几何的浅水方程物理参数"""
        self.geometry = geometry
        self.manning_n, self.gravity = manning_n, gravity

    def loss(self, model, x, t, condition):
        """计算质量守恒和动量守恒方程的均方残差"""
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)
        z, q = model(x, t, condition)
        area, _, perimeter = self.geometry(x, z)
        area_t = derivative(area, t)
        q_t = derivative(q, t)
        q_x = derivative(q, x)
        flux_x = derivative(q.square() / area.clamp_min(1.0e-6), x)
        z_x = derivative(z, x)
        radius = area / perimeter.clamp_min(1.0e-6)
        friction = self.manning_n ** 2 * q * q.abs() / (
            area.square().clamp_min(1.0e-6) * radius.clamp_min(1.0e-6).pow(4.0 / 3.0)
        )
        mass = area_t + q_x
        momentum = q_t + flux_x + self.gravity * area * (z_x + friction)
        return mass.square().mean(), momentum.square().mean()


# ============================== 训练 ==============================
def supervised_points(scenario, kind, section_indices=(23, 46, 69)):
    """返回初始或内部断面监督点。"""
    x, t, z, q = scenario["x"], scenario["t"], scenario["z"], scenario["q"]
    if kind == "initial":
        return x, np.full_like(x, t[0]), z[0], q[0]
    selected_x, selected_t = np.meshgrid(x[list(section_indices)], t, indexing="xy")
    return selected_x.ravel(), selected_t.ravel(), z[:, list(section_indices)].ravel(), q[:, list(section_indices)].ravel()


def loss_for_scenario(model, pde, scenario, condition, options):
    """计算单个场景的初始、边界、数据及物理约束损失"""
    z_scale = max(model.scales.z_std, 1.0e-6)
    q_scale = max(model.scales.q_std, 1.0e-6)

    losses = []
    for kind in ("initial", "sections"):
        x, t, z, q = supervised_points(scenario, kind, options.supervised_section_indices)
        zp, qp = model(to_tensor(x), to_tensor(t), condition)
        losses.append(
            (((zp - to_tensor(z)) / z_scale) ** 2).mean()
            + (((qp - to_tensor(q)) / q_scale) ** 2).mean()
        )

    initial_loss, data_loss = losses

    # 一维浅水方程只使用已知的两个外边界：上游流量 Q、下游水位 Z。
    t_boundary = to_tensor(scenario["t"])
    upstream_x = to_tensor(np.full_like(scenario["t"], scenario["x"][0]))
    downstream_x = to_tensor(np.full_like(scenario["t"], scenario["x"][-1]))
    _, upstream_q = model(upstream_x, t_boundary, condition)
    downstream_z, _ = model(downstream_x, t_boundary, condition)
    boundary_loss = (
        (((upstream_q - to_tensor(scenario["q"][:, 0])) / q_scale) ** 2).mean()
        + (((downstream_z - to_tensor(scenario["z"][:, -1])) / z_scale) ** 2).mean()
    )

    # 随机采样时空点，计算质量守恒和动量守恒损失
    x = scenario["x"][0] + (scenario["x"][-1] - scenario["x"][0]) * np.random.rand(options.physics_points)
    t = scenario["t"][0] + (scenario["t"][-1] - scenario["t"][0]) * np.random.rand(options.physics_points)
    mass_loss, momentum_loss = pde.loss(model, to_tensor(x), to_tensor(t), condition)

    return torch.stack((initial_loss, boundary_loss, data_loss, mass_loss, momentum_loss))

# 权重
def update_gradnorm_weights(component_losses, loss_weights, model, learning_rate, min_weight, max_weight):
    """根据五项损失的梯度范数更新权重"""
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradient_norms = []

    for index in range(component_losses.numel()):
        gradients = torch.autograd.grad(
            loss_weights[index] * component_losses[index],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = torch.zeros(
            (), dtype=component_losses.dtype, device=component_losses.device
        )
        for gradient in gradients:
            if gradient is not None:
                squared_norm = squared_norm + gradient.square().sum()
        gradient_norms.append(torch.sqrt(squared_norm + 1.0e-12))

    gradient_norms = torch.stack(gradient_norms)
    target_norm = gradient_norms.mean()

    with torch.no_grad():
        ratio = target_norm / gradient_norms.clamp_min(1.0e-12)
        loss_weights.mul_(ratio.pow(learning_rate))
        loss_weights.clamp_(min=min_weight, max=max_weight)
        loss_weights.mul_(loss_weights.numel() / loss_weights.sum().clamp_min(1.0e-12))

    return gradient_norms.detach()


@torch.no_grad()
def evaluate(model, dataset, condition_standardizer, scenario_paths, count=32, points_per_scenario=1024, seed=0):
    """在固定采样的场景和时空点上评估水位、流量预测误差及 NSE。"""
    model.eval()
    rng = np.random.default_rng(seed)
    selected_paths = scenario_paths if len(scenario_paths) <= count else rng.choice(scenario_paths, count, replace=False)
    predicted_water_levels, predicted_discharges = [], []
    true_water_levels, true_discharges = [], []
    for scenario_path in selected_paths:
        scenario = dataset.load_scenario(scenario_path)
        point_count = min(points_per_scenario, scenario["x"].size * scenario["t"].size)
        time_indices = rng.integers(0, len(scenario["t"]), point_count)
        space_indices = rng.integers(0, len(scenario["x"]), point_count)
        condition = condition_standardizer.transform(dataset.get_condition_vector(scenario_path))
        predicted_z, predicted_q = model(
            to_tensor(scenario["x"][space_indices]),
            to_tensor(scenario["t"][time_indices]),
            to_tensor(condition),
        )
        predicted_water_levels.append(predicted_z.detach().cpu().numpy())
        predicted_discharges.append(predicted_q.detach().cpu().numpy())
        true_water_levels.append(scenario["z"][time_indices, space_indices])
        true_discharges.append(scenario["q"][time_indices, space_indices])
    zp, qp, zt, qt = map(np.concatenate, (predicted_water_levels, predicted_discharges, true_water_levels, true_discharges))
    return {
        "z_l2": 100 * np.linalg.norm(zp - zt) / np.linalg.norm(zt),
        "q_l2": 100 * np.linalg.norm(qp - qt) / np.linalg.norm(qt),
        "z_nse": 1 - np.sum((zp - zt) ** 2) / np.sum((zt - zt.mean()) ** 2),
        "q_nse": 1 - np.sum((qp - qt) ** 2) / np.sum((qt - qt.mean()) ** 2),
    }


def train(model, pde, dataset, condition_standardizer, options, output_directory):
    """执行模型训练、定期评估，并保存训练历史和最佳模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)  # 优化器
    loss_weights = torch.tensor(
        (
            options.initial_weight,
            options.boundary_weight,
            options.data_weight,
            options.physics_weight * options.mass_weight,
            options.physics_weight * options.momentum_weight,
        ),
        dtype=torch.float32,
        device=compute_device,
    )

    history = (output_directory / "history.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    writer.writerow(
        (
            "epoch", "loss", "initial", "boundary", "data", "mass", "momentum",
            "weighted_initial", "weighted_boundary", "weighted_data",
            "weighted_mass", "weighted_momentum",
            "w_initial", "w_boundary", "w_data", "w_mass", "w_momentum", "test_q_nse",
        )
    )

    best_q_nse = -float("inf")
    loss_scales = None if options.auto_initialize_weights else torch.ones(
        5, dtype=torch.float32, device=compute_device
    )
    adaptive_start_epoch = max(options.fixed_weight_epochs + 1, options.weight_update_start)
    # ============================== 训练循环 ==============================
    for epoch in range(1, options.epochs + 1):
        model.train()
        # 固定本 epoch 使用的基准，避免第 1 个 epoch 结束时更新基准后影响当期日志。
        epoch_loss_scales = loss_scales
        shuffled_scenario_paths = np.random.permutation(dataset.train_files)    # 随机打乱训练场景顺序，按批次训练

        scenario_loss_terms, weighted_loss_terms = [], []   # 场景损失项、加权损失项
        batch_total_losses, batch_count = [], 0 # 批次总损失、批次数

        # 按批次训练
        for start in range(0, len(shuffled_scenario_paths), options.scenarios_per_batch):
            optimizer.zero_grad()

            batch_scenario_paths = shuffled_scenario_paths[start:start + options.scenarios_per_batch]   # 批次场景
            batch_component_losses = [] # 批次组件损失项
            # 计算批次中每个场景的损失项
            for scenario_path in batch_scenario_paths:
                # 计算场景条件向量并标准化
                condition = to_tensor(condition_standardizer.transform(dataset.get_condition_vector(scenario_path)))
                # 计算场景的初始、边界、数据及物理约束损失项(原始损失项,单值)
                terms = loss_for_scenario(model, pde, dataset.load_scenario(scenario_path), condition, options)
                batch_component_losses.append(terms)

            component_losses = torch.stack(batch_component_losses).mean(0)  # 批次中每个场景的损失项平均值
            objective_losses = (
                component_losses
                if epoch_loss_scales is None
                else component_losses / epoch_loss_scales
            )

            if (
                options.adaptive_weighting
                and epoch >= adaptive_start_epoch
                # 第一个自适应 epoch 更新一次，之后按周期更新。
                and (epoch - adaptive_start_epoch) % options.weight_update_every == 0
                and start == 0
            ):
                update_gradnorm_weights(
                    objective_losses, loss_weights, model,
                    options.weight_update_rate,
                    options.min_loss_weight,
                    options.max_loss_weight,
                )

            batch_total_loss = torch.dot(loss_weights, objective_losses)    # 归一化后的加权损失
            batch_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)   # 梯度裁剪
            optimizer.step()

            # 记录批次损失项和总损失
            scenario_loss_terms.extend([terms.detach() for terms in batch_component_losses])
            weighted_loss_terms.extend([
                loss_weights.detach() * (
                    terms.detach()
                    if epoch_loss_scales is None
                    else terms.detach() / epoch_loss_scales
                )
                for terms in batch_component_losses
            ])
            batch_total_losses.append(batch_total_loss.detach())    # 加权损失

            batch_count += 1

        mean_tensor = torch.stack(scenario_loss_terms).mean(0)
        if loss_scales is None and options.auto_initialize_weights:
            loss_scales = mean_tensor.detach().clamp_min(1.0e-12)
        mean = mean_tensor.cpu().numpy()   # 计算每个损失项的平均值
        weighted_mean = (torch.stack(weighted_loss_terms).mean(0).cpu().numpy())   # 计算每个加权损失项的平均值
        total_mean = torch.stack(batch_total_losses).mean().item()

        # 评估模型在测试集和训练集上的表现
        test = evaluate(model, dataset, condition_standardizer, dataset.test_files, seed=2033)
        train_eval = evaluate(model, dataset, condition_standardizer, dataset.train_files, seed=2034)

        # 保存最佳模型
        if test and test["q_nse"] > best_q_nse:
            best_q_nse = test["q_nse"]
            torch.save(model.state_dict(), output_directory / "best.pt")

        # 按固定列宽输出 epoch 日志块，便于横向比较
        labels = ("IC", "BC", "Data", "Mass", "Mom")
        header = "           " + "".join(f"{label:>14}" for label in labels)
        raw_text = "raw      " + "".join(f"{value:14.2e}" for value in mean)
        effective_loss_weights = (
            loss_weights.detach()
            if epoch_loss_scales is None
            else loss_weights.detach() / epoch_loss_scales
        )
        weight_text = "weight   " + "".join(
            f"{value:14.2e}" for value in effective_loss_weights.cpu().tolist()
        )
        weighted_text = "weighted " + "".join(f"{value:14.2e}" for value in weighted_mean)
        print(
            f"\n----- epoch {epoch:4d} | total={total_mean:.2e} -----\n"
            f"train: L2 Z/Q={train_eval['z_l2']:.2f}/{train_eval['q_l2']:.2f}% | "
            f"NSE Z/Q={train_eval['z_nse']:.3f}/{train_eval['q_nse']:.3f}\n"
            f"test : L2 Z/Q={test['z_l2']:.2f}/{test['q_l2']:.2f}% | "
            f"NSE Z/Q={test['z_nse']:.3f}/{test['q_nse']:.3f}\n"
            f"{header}\n{raw_text}\n{weight_text}\n{weighted_text}",
            flush=True,
        )

        # 保存训练历史
        writer.writerow(
            (
                epoch, total_mean, *mean, *weighted_mean,
                *effective_loss_weights.cpu().tolist(),
                "" if test is None else test["q_nse"],
            )
        )
        history.flush()

    history.close()
    print(f"finished | best test Q-NSE {best_q_nse:.4f} | output {output_directory}")


def main():
    # 训练配置
    random_seed = 2032; np.random.seed(random_seed); torch.manual_seed(random_seed)
    if compute_device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)
    print(f"device = {compute_device}", flush=True)

    # 初始化数据、几何、模型
    dataset = ScenarioDataset(cache_directory)
    condition_standardizer, scales = fit_condition_normalizer_and_scales(dataset)   # 计算条件向量标准化器和物理尺度
    geometry = CrossSectionGeometry(
        cross_section_profile_path,
        to_tensor(dataset.load_scenario(dataset.train_files[0])["x"]),
        device=compute_device,
    )
    model = OperatorPINN(scales, geometry).to(compute_device)
    pde = PDE(geometry)

    # 保存训练配置和物理尺度
    output_directory = project_root / "outputs" / "flow_generalization_operator" / "operator_pinn_simple"
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"options": TrainingOptions().__dict__, "scales": scales.__dict__.copy()}, f, ensure_ascii=False, indent=2)

    # 启动训练
    train(model, pde, dataset, condition_standardizer, TrainingOptions(), output_directory)


if __name__ == "__main__":
    main()
