"""Checkpoint and prediction persistence for a single river1d run."""

from pathlib import Path

import pandas as pd
import torch

try:
    from .config import TrainConfig
    from .data import RiverData
    from .network import build_flow_network
    from ._util import choose_device
except ImportError:
    from config import TrainConfig
    from data import RiverData
    from network import build_flow_network
    from _util import choose_device


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, best_metric):
    """Persist all state required to reproduce or resume a training run."""
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config.to_dict(),
            "best_loss": best_metric,
            "best_metric": best_metric,
            "best_metric_name": "test_mean_h_q_l2_percent",
        },
        path,
    )


@torch.no_grad()
def export_predictions(checkpoint_path: Path, output_csv: Path, device: str = "auto"):
    """Export h, q, u, area and geometry on the full observed x-t grid."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    runtime_device = choose_device(device)
    dtype = getattr(torch, cfg.get("dtype", "float32"))
    data = RiverData(
        cfg["data_path"], cfg["cross_section_path"],
        device=runtime_device, dtype=dtype,
        geometry_levels=cfg.get("geometry_levels", 512),
    )
    config = TrainConfig(
        data_path=Path(cfg["data_path"]),
        cross_section_path=Path(cfg["cross_section_path"]),
        output_dir=Path(cfg.get("output_dir", "outputs/river1d")),
        network_type=cfg.get("network_type", "fourier_resnet"),
        hidden_layers=tuple(cfg.get("hidden_layers", [20] * 6)),
        activation=cfg.get("activation", "tanh"),
        fourier_features=cfg.get("fourier_features", 128),
        fourier_sigma_x=cfg.get("fourier_sigma_x", 10.0),
        fourier_sigma_t=cfg.get("fourier_sigma_t", 1.0),
        fourier_head_width=cfg.get("fourier_head_width", 16),
        dropout=cfg.get("dropout", 0.7),
        use_geometry_calibration=cfg.get("use_geometry_calibration", False),
        geometry_levels=cfg.get("geometry_levels", 512),
        dtype=cfg.get("dtype", "float32"),
        device=device,
    )
    if config.use_geometry_calibration:
        data.calibrate_geometry(data.t_s.numel())
    model = build_flow_network(
        data, config.hidden_layers, config.activation, config.network_type,
        fourier_features=config.fourier_features,
        sigma_x=config.fourier_sigma_x,
        sigma_t=config.fourier_sigma_t,
        head_width=config.fourier_head_width,
        dropout=config.dropout,
    ).to(device=runtime_device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x, t = data.full_grid_coordinates()
    z, q = model(x, t)
    area, width, perimeter = data.hydraulic_geometry(x, z, t)
    h = area / width.clamp_min(1.0e-6)
    u = q / area.clamp_min(1.0e-6)
    result = pd.DataFrame({
        "time_s": t.cpu().numpy(), "time_h": t.cpu().numpy() / 3600.0,
        "dist_m": x.cpu().numpy(), "dist_km": x.cpu().numpy() / 1000.0,
        "z_pred_m": z.cpu().numpy(), "h_pred_m": h.cpu().numpy(),
        "u_pred_m_s": u.cpu().numpy(), "width_pred_m": width.cpu().numpy(),
        "area_pred_m2": area.cpu().numpy(), "perimeter_pred_m": perimeter.cpu().numpy(),
        "q_pred_m3_s": q.cpu().numpy(),
        "z_true_m": data.z_grid.reshape(-1).cpu().numpy(),
        "h_true_m": data.h_grid.reshape(-1).cpu().numpy(),
        "u_true_m_s": data.u_grid.reshape(-1).cpu().numpy(),
        "q_true_m3_s": data.q_grid.reshape(-1).cpu().numpy(),
    })
    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
