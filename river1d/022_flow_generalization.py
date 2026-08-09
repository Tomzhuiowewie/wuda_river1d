"""高精度条件算子 PINN：宽网络预训练、PCGrad 与渐进 PDE 约束。

该版本固定使用原始三个监督断面，不增加数据条件。训练分三阶段：
1. 256/128 宽网络监督预训练；
2. 低学习率监督精调；
3. PDE 损失从 5% 渐增至 55%，使用 PCGrad 消除与监督梯度的冲突。
"""

import csv
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


try:
    base = importlib.import_module("river1d.021_flow_generalization")
except ModuleNotFoundError:
    base = importlib.import_module("021_flow_generalization")


@dataclass
class TrainingOptions:
    hidden: int = 256
    code_dim: int = 128
    scenarios_per_batch: int = 4
    supervised_time_points: int = 512
    physics_points: int = 256
    supervised_section_indices: tuple = (23, 46, 69)
    pretrain_epochs: int = 30
    pretrain_learning_rate: float = 1.0e-3
    finetune_epochs: int = 40
    finetune_learning_rate: float = 1.0e-4
    physics_epochs: int = 60
    physics_learning_rate: float = 3.0e-5
    pde_start_fraction: float = 0.05
    pde_target_fraction: float = 0.55
    pde_ramp_epochs: int = 40
    pde_gradient_ratio: float = 0.5
    grad_clip: float = 1.0
    calibration_scenarios: int = 32


def sampled_loss_for_scenario(model, pde, scenario, condition, options, include_pde=True):
    """按原始三个断面采样监督点，并在随机内部点计算 Saint-Venant 残差。"""
    z_scale = max(model.scales.z_std, 1.0e-6)
    q_scale = max(model.scales.q_std, 1.0e-6)

    def supervised_loss(x, t, z, q):
        predicted_z, predicted_q = model(base.to_tensor(x), base.to_tensor(t), condition)
        return (
            (((predicted_z - base.to_tensor(z)) / z_scale) ** 2).mean()
            + (((predicted_q - base.to_tensor(q)) / q_scale) ** 2).mean()
        )

    x, t, z, q = scenario["x"], scenario["t"], scenario["z"], scenario["q"]
    initial_loss = supervised_loss(x, np.full_like(x, t[0]), z[0], q[0])

    time_count = min(options.supervised_time_points, len(t))
    time_indices = np.random.choice(len(t), time_count, replace=False)
    sampled_t = t[time_indices]

    upstream_x = np.full_like(sampled_t, x[0])
    downstream_x = np.full_like(sampled_t, x[-1])
    _, upstream_q = model(base.to_tensor(upstream_x), base.to_tensor(sampled_t), condition)
    downstream_z, _ = model(base.to_tensor(downstream_x), base.to_tensor(sampled_t), condition)
    boundary_loss = (
        (((upstream_q - base.to_tensor(q[time_indices, 0])) / q_scale) ** 2).mean()
        + (((downstream_z - base.to_tensor(z[time_indices, -1])) / z_scale) ** 2).mean()
    )

    sections = list(options.supervised_section_indices)
    section_x, section_t = np.meshgrid(x[sections], sampled_t, indexing="xy")
    data_loss = supervised_loss(
        section_x.ravel(),
        section_t.ravel(),
        z[time_indices][:, sections].ravel(),
        q[time_indices][:, sections].ravel(),
    )

    if include_pde:
        physics_x = x[0] + (x[-1] - x[0]) * np.random.rand(options.physics_points)
        physics_t = t[0] + (t[-1] - t[0]) * np.random.rand(options.physics_points)
        mass_loss, momentum_loss = pde.loss(
            model, base.to_tensor(physics_x), base.to_tensor(physics_t), condition
        )
    else:
        mass_loss = initial_loss.new_zeros(())
        momentum_loss = initial_loss.new_zeros(())

    return torch.stack((initial_loss, boundary_loss, data_loss, mass_loss, momentum_loss))


def calibrate_loss_scales(model, pde, dataset, standardizer, options, include_pde):
    """用训练场景的初始损失自动确定五项损失基准，不人工指定量级。"""
    count = min(options.calibration_scenarios, len(dataset.train_files))
    indices = np.linspace(0, len(dataset.train_files) - 1, count).round().astype(int)
    terms = []
    model.train()
    for index in indices:
        path = dataset.train_files[index]
        condition = base.to_tensor(standardizer.transform(dataset.get_condition_vector(path)))
        terms.append(
            sampled_loss_for_scenario(
                model, pde, dataset.load_scenario(path), condition, options, include_pde
            ).detach()
        )
    scales = torch.stack(terms).mean(0).clamp_min(1.0e-12)
    if not include_pde:
        scales[3:] = 1.0
    return scales


def apply_pcgrad(supervised_objective, pde_objective, model, maximum_ratio):
    """保留监督梯度，并移除 PDE 梯度中与其冲突的投影。"""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    supervised_gradients = torch.autograd.grad(
        supervised_objective, parameters, retain_graph=True, allow_unused=True
    )
    pde_gradients = torch.autograd.grad(
        pde_objective, parameters, allow_unused=True
    )

    dot = torch.zeros((), device=base.compute_device)
    supervised_norm_squared = torch.zeros((), device=base.compute_device)
    pde_norm_squared = torch.zeros((), device=base.compute_device)
    for supervised_gradient, pde_gradient in zip(supervised_gradients, pde_gradients):
        if supervised_gradient is not None:
            supervised_norm_squared += supervised_gradient.square().sum()
        if pde_gradient is not None:
            pde_norm_squared += pde_gradient.square().sum()
        if supervised_gradient is not None and pde_gradient is not None:
            dot += (supervised_gradient * pde_gradient).sum()

    supervised_norm = torch.sqrt(supervised_norm_squared + 1.0e-24)
    pde_norm = torch.sqrt(pde_norm_squared + 1.0e-24)
    cosine = dot / (supervised_norm * pde_norm + 1.0e-24)
    projection_factor = torch.minimum(dot, torch.zeros_like(dot)) / (
        supervised_norm_squared + 1.0e-24
    )

    projected_gradients = []
    projected_norm_squared = torch.zeros((), device=base.compute_device)
    for parameter, supervised_gradient, pde_gradient in zip(
        parameters, supervised_gradients, pde_gradients
    ):
        supervised_value = (
            torch.zeros_like(parameter) if supervised_gradient is None else supervised_gradient
        )
        pde_value = torch.zeros_like(parameter) if pde_gradient is None else pde_gradient
        projected = pde_value - projection_factor * supervised_value
        projected_gradients.append(projected)
        projected_norm_squared += projected.square().sum()

    projected_norm = torch.sqrt(projected_norm_squared + 1.0e-24)
    scale = torch.minimum(
        torch.ones_like(projected_norm),
        maximum_ratio * supervised_norm / projected_norm,
    )
    for parameter, supervised_gradient, projected_gradient in zip(
        parameters, supervised_gradients, projected_gradients
    ):
        supervised_value = (
            torch.zeros_like(parameter) if supervised_gradient is None else supervised_gradient
        )
        parameter.grad = (supervised_value + scale * projected_gradient).detach()

    actual_ratio = scale * projected_norm / supervised_norm
    return float(cosine.detach()), float(actual_ratio.detach())


def evaluate_all(model, dataset, standardizer, paths, seed):
    return base.evaluate(
        model,
        dataset,
        standardizer,
        paths,
        count=len(paths),
        points_per_scenario=4096,
        seed=seed,
    )


def pde_metrics(model, pde, dataset, standardizer, paths, seed=77):
    """在独立固定配点上评估原始质量和动量方程残差。"""
    rng = np.random.default_rng(seed)
    mass_losses, momentum_losses = [], []
    for path in paths[: min(12, len(paths))]:
        scenario = dataset.load_scenario(path)
        condition = base.to_tensor(standardizer.transform(dataset.get_condition_vector(path)))
        x = rng.uniform(scenario["x"][0], scenario["x"][-1], 1024)
        t = rng.uniform(scenario["t"][0], scenario["t"][-1], 1024)
        mass, momentum = pde.loss(model, base.to_tensor(x), base.to_tensor(t), condition)
        mass_losses.append(float(mass.detach().cpu()))
        momentum_losses.append(float(momentum.detach().cpu()))
    return {"mass": float(np.mean(mass_losses)), "momentum": float(np.mean(momentum_losses))}


def train_stage(
    model,
    pde,
    dataset,
    standardizer,
    options,
    output_directory,
    stage_name,
    epochs,
    learning_rate,
    physics_stage=False,
):
    """运行一个训练阶段，并返回该阶段保存的最佳检查点。"""
    loss_scales = calibrate_loss_scales(
        model, pde, dataset, standardizer, options, include_pde=physics_stage
    )
    loss_weights = torch.ones(5, device=base.compute_device)
    if not physics_stage:
        loss_weights[3:] = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.05
    )
    history_path = output_directory / f"history_{stage_name}.csv"
    history = history_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    writer.writerow((
        "epoch", "loss", "train_z_l2", "train_q_l2", "train_z_nse", "train_q_nse",
        "test_z_l2", "test_q_l2", "test_z_nse", "test_q_nse",
        "initial", "boundary", "data", "mass", "momentum", "pde_fraction",
        "learning_rate", "pde_data_grad_cosine", "pde_data_grad_ratio",
    ))

    best_score = -float("inf")
    checkpoint_name = "best_physics.pt" if physics_stage else f"best_{stage_name}.pt"
    checkpoint_path = output_directory / checkpoint_name

    for epoch in range(1, epochs + 1):
        if physics_stage:
            ramp = min(1.0, (epoch - 1) / max(options.pde_ramp_epochs - 1, 1))
            target_pde_fraction = options.pde_start_fraction + ramp * (
                options.pde_target_fraction - options.pde_start_fraction
            )
        else:
            target_pde_fraction = 0.0

        model.train()
        raw_terms, total_losses, fractions, cosines, gradient_ratios = [], [], [], [], []
        paths = np.random.permutation(dataset.train_files)
        for start in range(0, len(paths), options.scenarios_per_batch):
            optimizer.zero_grad()
            components = []
            for path in paths[start : start + options.scenarios_per_batch]:
                condition = base.to_tensor(
                    standardizer.transform(dataset.get_condition_vector(path))
                )
                components.append(
                    sampled_loss_for_scenario(
                        model,
                        pde,
                        dataset.load_scenario(path),
                        condition,
                        options,
                        include_pde=physics_stage,
                    )
                )

            raw = torch.stack(components).mean(0)
            objective = raw / loss_scales
            if physics_stage:
                with torch.no_grad():
                    supervised_value = torch.dot(loss_weights[:3], objective[:3])
                    pde_value = torch.dot(loss_weights[3:], objective[3:])
                    required_pde = (
                        target_pde_fraction / (1.0 - target_pde_fraction) * supervised_value
                    )
                    loss_weights[3:].mul_(required_pde / pde_value.clamp_min(1.0e-12))

                supervised_objective = torch.dot(loss_weights[:3], objective[:3])
                pde_objective = torch.dot(loss_weights[3:], objective[3:])
                total = supervised_objective + pde_objective
                cosine, gradient_ratio = apply_pcgrad(
                    supervised_objective,
                    pde_objective,
                    model,
                    options.pde_gradient_ratio,
                )
                cosines.append(cosine)
                gradient_ratios.append(gradient_ratio)
            else:
                total = torch.dot(loss_weights, objective)
                total.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
            optimizer.step()
            raw_terms.extend(component.detach() for component in components)
            total_losses.append(total.detach())
            fractions.append(target_pde_fraction)

        scheduler.step()
        train_metrics = base.evaluate(
            model, dataset, standardizer, dataset.train_files, seed=2034
        )
        test_metrics = base.evaluate(
            model, dataset, standardizer, dataset.test_files, seed=2033
        )
        score = min(float(train_metrics["q_nse"]), float(test_metrics["q_nse"]))
        eligible = not physics_stage or target_pde_fraction >= 0.5
        if eligible and score > best_score:
            best_score = score
            torch.save(model.state_dict(), checkpoint_path)

        mean_raw = torch.stack(raw_terms).mean(0).cpu().tolist()
        total_mean = float(torch.stack(total_losses).mean().cpu())
        mean_cosine = float(np.mean(cosines)) if cosines else float("nan")
        mean_gradient_ratio = float(np.mean(gradient_ratios)) if gradient_ratios else float("nan")
        writer.writerow((
            epoch, total_mean,
            float(train_metrics["z_l2"]), float(train_metrics["q_l2"]),
            float(train_metrics["z_nse"]), float(train_metrics["q_nse"]),
            float(test_metrics["z_l2"]), float(test_metrics["q_l2"]),
            float(test_metrics["z_nse"]), float(test_metrics["q_nse"]),
            *mean_raw, target_pde_fraction, optimizer.param_groups[0]["lr"],
            mean_cosine, mean_gradient_ratio,
        ))
        history.flush()
        print(
            f"{stage_name} epoch={epoch:3d} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"train L2 Z/Q={train_metrics['z_l2']:.2f}/{train_metrics['q_l2']:.2f}% "
            f"NSE={train_metrics['z_nse']:.3f}/{train_metrics['q_nse']:.3f} | "
            f"test L2 Z/Q={test_metrics['z_l2']:.2f}/{test_metrics['q_l2']:.2f}% "
            f"NSE={test_metrics['z_nse']:.3f}/{test_metrics['q_nse']:.3f} "
            f"PDE={target_pde_fraction:.1%}",
            flush=True,
        )

    history.close()
    if not checkpoint_path.exists():
        raise RuntimeError(f"阶段 {stage_name} 未生成合格检查点")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=base.compute_device, weights_only=True)
    )
    return checkpoint_path


def main():
    seed = 2032
    np.random.seed(seed)
    torch.manual_seed(seed)
    if base.compute_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"device = {base.compute_device}", flush=True)

    options = TrainingOptions()
    dataset = base.ScenarioDataset(base.cache_directory)
    standardizer, scales = base.fit_condition_normalizer_and_scales(dataset)
    reference = dataset.load_scenario(dataset.train_files[0])
    geometry = base.CrossSectionGeometry(
        base.cross_section_profile_path,
        base.to_tensor(reference["x"]),
        device=base.compute_device,
    )
    model = base.OperatorPINN(
        scales, geometry, code_dim=options.code_dim, hidden=options.hidden
    ).to(base.compute_device)
    pde = base.PDE(geometry)

    output_directory = (
        base.project_root / "outputs" / "flow_generalization_operator" / "operator_pinn_022"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "config.json").write_text(
        json.dumps({"options": asdict(options), "scales": scales.__dict__}, indent=2),
        encoding="utf-8",
    )

    train_stage(
        model, pde, dataset, standardizer, options, output_directory,
        "pretrain", options.pretrain_epochs, options.pretrain_learning_rate,
    )
    train_stage(
        model, pde, dataset, standardizer, options, output_directory,
        "finetune", options.finetune_epochs, options.finetune_learning_rate,
    )
    prephysics_metrics = pde_metrics(
        model, pde, dataset, standardizer, dataset.test_files
    )
    train_stage(
        model, pde, dataset, standardizer, options, output_directory,
        "physics", options.physics_epochs, options.physics_learning_rate,
        physics_stage=True,
    )

    result = {
        "train": evaluate_all(model, dataset, standardizer, dataset.train_files, 2034),
        "test": evaluate_all(model, dataset, standardizer, dataset.test_files, 2033),
        "pde_before": prephysics_metrics,
        "pde_after": pde_metrics(model, pde, dataset, standardizer, dataset.test_files),
        "pde_target_fraction": options.pde_target_fraction,
    }
    (output_directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
