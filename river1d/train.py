from __future__ import annotations

import csv
import json
import time

import torch

from config import CONFIG, TrainConfig
from data import RiverData
from network import build_flow_network
from loss import PINNResidual, data_loss
from _save import save_checkpoint
from _sampling import physics_sampling
from _util import build_scheduler, choose_device, create_output_dir, grid_metrics, set_seed, update_ntk_weights


def stage_weights(epoch, total_epochs, dtype, device):
    """三阶段权重：前期数据主导，中期平衡，后期增强 PDE。"""
    progress = epoch / total_epochs
    data_weight = 1.0
    if progress < 0.3:
        physics_weight = 1.0e-3
    elif progress < 0.7:
        physics_weight = 1.0e-2
    else:
        physics_weight = 3.0e-2
    return (
        torch.tensor(data_weight, dtype=dtype, device=device),
        torch.tensor(physics_weight, dtype=dtype, device=device),
    )


def train(config: TrainConfig):
    set_seed(config.seed)
    device = choose_device(config.device)
    dtype = getattr(torch, config.dtype)
    output_dir = create_output_dir(config.output_dir)
    config.output_dir = output_dir

    data = RiverData(config.data_path, config.cross_section_path, device=device)

    train_time_count = data.t_s.numel()

    # 初始条件 + 边界条件
    initial_boundary_data = data.initial_boundary_data(None)

    if len(config.interior_section_ids) > 0:
        interior_data = data.selected_interior_section_data(None, config.interior_section_ids)
    else:
        interior_data = data.interior_section_data(train_time_count, config.interior_section_count)
    
    model = build_flow_network(
        data,
        config.hidden_layers,
        config.activation,
        config.network_type,
        fourier_features=config.fourier_features,
        sigma_x=config.fourier_sigma_x,
        sigma_t=config.fourier_sigma_t,
        head_width=config.fourier_head_width,
        dropout=config.dropout,
        initial_boundary_data=initial_boundary_data,
    ).to(device=device, dtype=dtype)

    residual = PINNResidual(data).to(device=device, dtype=dtype)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)

    data_weight = torch.tensor(config.data_weight, dtype=dtype, device=device)
    physics_weight = torch.tensor(config.physics_weight, dtype=dtype, device=device)

    ntk_data_trace = torch.tensor(float("nan"), dtype=dtype, device=device)
    ntk_physics_trace = torch.tensor(float("nan"), dtype=dtype, device=device)

    best_metric = float("inf")
    best_epoch = 0
    # 保存训练参数
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)

    history_path = output_dir / "history.csv"
    history_file = history_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["epoch", "total_loss", "data_loss", "z_data_loss", "q_data_loss", "physics_loss", "mass_loss", "momentum_loss",
                  "data_weight", "physics_weight", "ntk_data_trace", "ntk_physics_trace", 
                  "z_loss", "q_loss", "mass_loss_unweighted", "momentum_loss_unweighted", "train_h_l2_percent", 
                  "train_q_l2_percent", "train_mean_l2_percent", "train_max_l2_percent", "learning_rate"]
    
    writer = csv.DictWriter(history_file, fieldnames=fieldnames)
    writer.writeheader()

    print(f"device={device} dtype={config.dtype} network={config.network_type} "
        f"scheduler={config.scheduler_type} "
        f"mode=whole "
        f"ntk_weighting={config.ntk_weighting} "
        f"physics_points/epoch={config.num_physics_points} "
        f"train_times={train_time_count} "
        f"test_times={data.t_s.numel()} "
        f"initial_boundary_points={len(initial_boundary_data['x'])} "
        f"interior_points={len(interior_data['x'])}")
    
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # 采样点
        x_phys, t_phys = physics_sampling(data, config, epoch)

        z_loss, q_loss = data_loss(model, data, initial_boundary_data)

        if len(interior_data["x"]) > 0:
            interior_z_loss, interior_q_loss = data_loss(model, data, interior_data)
            z_loss = z_loss + config.interior_data_weight * interior_z_loss
            q_loss = q_loss + config.interior_data_weight * interior_q_loss
        
        mass_loss, momentum_loss = residual.loss(model, x_phys, t_phys)

        z_term = config.z_weight * z_loss
        q_term = config.q_weight * q_loss
        data_term = z_term + q_term

        mass_term = config.mass_weight * mass_loss
        momentum_term = config.momentum_weight * momentum_loss
        physics_term = mass_term + momentum_term


        # 权重配置
        if not config.ntk_weighting:
            data_weight, physics_weight = stage_weights(epoch, config.epochs, dtype, device)

        if config.ntk_weighting and (epoch == 1 or epoch % max(1, config.ntk_update_every) == 0):
            data_weight, physics_weight, ntk_data_trace, ntk_physics_trace = update_ntk_weights(
                model, data_term, physics_term, config, data_weight, physics_weight)

        total = data_weight * data_term + physics_weight * physics_term
        
        total.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        scheduler.step()

        total_value = total.item()
        
        should_report = epoch == 1 or epoch % config.print_every == 0 or epoch == config.epochs
        if should_report:
            model.eval()
            train_h_l2, train_q_l2 = grid_metrics(model, data)
            train_mean_l2 = 0.5 * (train_h_l2 + train_q_l2)
            train_max_l2 = max(train_h_l2, train_q_l2)
            row = {
                "epoch": epoch,
                "total_loss": total_value,
                "data_loss": data_term.item(),
                "z_data_loss": z_term.item(),
                "q_data_loss": q_term.item(),
                "physics_loss": physics_term.item(),
                "mass_loss": mass_term.item(),
                "momentum_loss": momentum_term.item(),
                "data_weight": data_weight.item(),
                "physics_weight": physics_weight.item(),
                "ntk_data_trace": ntk_data_trace.item(),
                "ntk_physics_trace": ntk_physics_trace.item(),
                "z_loss": z_loss.item(),
                "q_loss": q_loss.item(),
                "mass_loss_unweighted": mass_loss.item(),
                "momentum_loss_unweighted": momentum_loss.item(),
                "train_h_l2_percent": train_h_l2,
                "train_q_l2_percent": train_q_l2,
                "train_mean_l2_percent": train_mean_l2,
                "train_max_l2_percent": train_max_l2,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            writer.writerow(row)
            history_file.flush()
            elapsed = time.time() - start_time
            display_best_metric = min(best_metric, train_mean_l2)
            display_best_epoch = epoch if train_mean_l2 < best_metric else best_epoch
            print(
                f"epoch {epoch:5d} | "
                f"L {total_value:8.2e} "
                f"D {data_term.item():8.2e} "
                f"P {physics_term.item():8.2e} | "
                f"z/q {z_term.item():7.1e}/{q_term.item():7.1e} | "
                f"mass/mom {mass_loss.item():7.1e}/{momentum_loss.item():7.1e} | "
                f"L2 {train_h_l2:4.1f}/{train_q_l2:4.1f}/{train_mean_l2:4.1f}% | "
                f"best {display_best_metric:4.1f}@{display_best_epoch} | "
                f"{elapsed:4.1f}s"
            )
            current_metric = train_mean_l2
            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    best_metric,
                )

        if epoch % config.save_every == 0 or epoch == config.epochs:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                best_metric,
            )

    history_file.close()

    return output_dir / "best.pt"


if __name__ == "__main__":
    train(CONFIG)
