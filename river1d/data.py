from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

try:
    from ._geometry import CrossSectionGeometry
except ImportError:
    from _geometry import CrossSectionGeometry


@dataclass(frozen=True)
class PhysicalScales:
    x_min_m: float
    x_max_m: float
    t_min_s: float
    t_max_s: float
    length_m: float
    duration_s: float
    water_level_m: float
    depth_m: float
    velocity_m_s: float
    width_m: float
    area_m2: float
    discharge_m3_s: float
    friction_slope: float


class RiverData:
    """读取水动力数据和预测开始前已知的初始断面"""

    def __init__(
        self,
        csv_path,
        cross_section_path,
        device,
        dtype=torch.float32,
        geometry_levels=512,
    ):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.cross_section_path = Path(cross_section_path).expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = dtype

        frame = pd.read_csv(self.csv_path)
        self.frame = frame.sort_values(["time_h", "dist_km"]).reset_index(drop=True)
        
        # 单位转换(km→m;h→s)
        self.x_m = self._tensor(np.sort(frame["dist_km"].unique()) * 1000.0)
        self.t_s = self._tensor(np.sort(frame["time_h"].unique()) * 3600.0)
        
        # 真实值读取
        self.z_grid = self._pivot("zw_m")       # 水位
        self.h_grid = self._pivot("hav_m")      # 水深
        self.u_grid = self._pivot("uav_m_s")    # 流速

        self.q_grid = self._pivot("qd_m3_s")    # 流量
        self.area_grid = self._pivot("area_m2") # 过流面积
        self.width_grid = self._pivot("bw_m")   # 水面宽度

        initial_rows = self.frame[self.frame["time_h"] == self.frame["time_h"].min()].sort_values("dist_km")
        self.initial_manning_n = self._tensor(initial_rows["rn"].to_numpy()) # 初始曼宁系数

        self.geometry = CrossSectionGeometry(
            self.cross_section_path,
            self.x_m,
            device=self.device,
            dtype=self.dtype,
            levels=geometry_levels,
        )
        self.area_calibration = None
        self.width_calibration = None

        # 缩放因子（目的：使得PDE无量纲化）
        z0 = float(self.z_grid[0].mean())
        h0 = float(self.h_grid[0].mean())
        u0 = float(self.u_grid[0].abs().mean())

        q0 = float(self.q_grid[0].abs().mean())
        a0 = float(self.area_grid[0].mean())
        b0 = float(self.width_grid[0].mean())
        
        r0 = float(a0 / (b0 + 2 * h0))   # 水力半径（A / P）
        n0 = float(self.initial_manning_n.median())  # 初始糙率
        friction_slope0 = n0**2 * q0**2 / (a0**2 * r0 ** (4.0 / 3.0))

        self.scales = PhysicalScales(
            x_min_m=float(self.x_m[0]),
            x_max_m=float(self.x_m[-1]),
            t_min_s=float(self.t_s[0]),
            t_max_s=float(self.t_s[-1]),
            length_m=float(self.x_m[-1] - self.x_m[0]),
            duration_s=float(self.t_s[-1] - self.t_s[0]),
            water_level_m=z0,
            depth_m=h0,
            velocity_m_s=u0,
            discharge_m3_s=q0,
            area_m2=a0,
            width_m=b0,
            friction_slope=friction_slope0)

    def _tensor(self, values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    def _pivot(self, value: str) -> torch.Tensor:
        table = (
            self.frame.pivot(index="time_h", columns="dist_km", values=value)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        if table.isna().any().any():
            raise ValueError(f"Variable {value!r} contains missing grid values")
        return self._tensor(table.to_numpy())

    def initial_boundary_data(self, end_time_index):
        x_initial = self.x_m
        t_initial = torch.full_like(x_initial, self.t_s[0])

        if end_time_index is None:
            t_boundary = self.t_s
        else:
            t_boundary = self.t_s[:end_time_index]
        
        x_left = torch.full_like(t_boundary, self.x_m[0])
        x_right = torch.full_like(t_boundary, self.x_m[-1])

        x = torch.cat((x_initial, x_left, x_right))
        t = torch.cat((t_initial, t_boundary, t_boundary))

        z = torch.cat((self.z_grid[0], self.z_grid[:len(t_boundary), 0], self.z_grid[:len(t_boundary), -1]))
        q = torch.cat((self.q_grid[0], self.q_grid[:len(t_boundary), 0], self.q_grid[:len(t_boundary), -1]))
        h = torch.cat((self.h_grid[0], self.h_grid[:len(t_boundary), 0], self.h_grid[:len(t_boundary), -1]))
        u = torch.cat((self.u_grid[0], self.u_grid[:len(t_boundary), 0], self.u_grid[:len(t_boundary), -1]))

        area = torch.cat((self.area_grid[0], self.area_grid[:len(t_boundary), 0], self.area_grid[:len(t_boundary), -1]))
        width = torch.cat((self.width_grid[0], self.width_grid[:len(t_boundary), 0], self.width_grid[:len(t_boundary), -1]))

        return {"x": x, "t": t, "z": z, "h": h, "u": u, "q": q, "area": area, "width": width}

    def interior_section_data(self, end_time_index, section_count):
        """训练期内部断面监督数据；不包含上下游边界，不包含测试期。"""
        count = int(section_count)
        if count <= 0:
            empty = self.x_m[:0]
            return {"x": empty, "t": empty, "z": empty, "h": empty, "u": empty, "q": empty}

        interior_indices = torch.arange(1, self.x_m.numel() - 1, device=self.device)
        if interior_indices.numel() == 0:
            empty = self.x_m[:0]
            return {"x": empty, "t": empty, "z": empty, "h": empty, "u": empty, "q": empty}

        pick = torch.linspace(
            0, interior_indices.numel() - 1,
            steps=min(count, interior_indices.numel()),
            device=self.device,
        ).round().long()
        section_indices = interior_indices[pick]

        t_values = self.t_s[:end_time_index]
        time_count = t_values.numel()
        section_count = section_indices.numel()
        x = self.x_m[section_indices].repeat(time_count)
        t = t_values.repeat_interleave(section_count)
        z = self.z_grid[:end_time_index, section_indices].reshape(-1)
        h = self.h_grid[:end_time_index, section_indices].reshape(-1)
        u = self.u_grid[:end_time_index, section_indices].reshape(-1)
        q = self.q_grid[:end_time_index, section_indices].reshape(-1)
        return {"x": x, "t": t, "z": z, "h": h, "u": u, "q": q}

    def selected_interior_section_data(self, end_time_index, section_ids):
        """训练期指定内部断面监督数据；section_ids 是 1-based 断面编号。"""
        if len(section_ids) == 0:
            empty = self.x_m[:0]
            return {"x": empty, "t": empty, "z": empty, "h": empty, "u": empty, "q": empty}

        section_indices = torch.as_tensor(
            [int(section_id) - 1 for section_id in section_ids],
            dtype=torch.long,
            device=self.device,
        )
        section_indices = section_indices[
            (section_indices > 0) & (section_indices < self.x_m.numel() - 1)
        ].unique()
        if section_indices.numel() == 0:
            empty = self.x_m[:0]
            return {"x": empty, "t": empty, "z": empty, "h": empty, "u": empty, "q": empty}

        t_values = self.t_s[:end_time_index]
        time_count = t_values.numel()
        section_count = section_indices.numel()
        x = self.x_m[section_indices].repeat(time_count)
        t = t_values.repeat_interleave(section_count)
        z = self.z_grid[:end_time_index, section_indices].reshape(-1)
        h = self.h_grid[:end_time_index, section_indices].reshape(-1)
        u = self.u_grid[:end_time_index, section_indices].reshape(-1)
        q = self.q_grid[:end_time_index, section_indices].reshape(-1)
        return {"x": x, "t": t, "z": z, "h": h, "u": u, "q": q}

    def boundary_values(self, t_s: torch.Tensor):
        """插值给定时刻的上下游水位和流量边界条件。"""
        shape = t_s.shape
        t = t_s.reshape(-1).clamp(self.t_s[0], self.t_s[-1])
        index = self._cell_indices(t, self.t_s)
        t0, t1 = self.t_s[index], self.t_s[index + 1]
        weight = (t - t0) / (t1 - t0)

        def interpolate(values):
            return ((1.0 - weight) * values[index] + weight * values[index + 1]).reshape(shape)

        return (
            interpolate(self.z_grid[:, 0]),
            interpolate(self.z_grid[:, -1]),
            interpolate(self.q_grid[:, 0]),
            interpolate(self.q_grid[:, -1]),
        )

    def initial_values(self, x_m: torch.Tensor):
        """插值初始时刻的水位和流量。"""
        shape = x_m.shape
        x = x_m.reshape(-1).clamp(self.x_m[0], self.x_m[-1])
        index = self._cell_indices(x, self.x_m)
        x0, x1 = self.x_m[index], self.x_m[index + 1]
        weight = (x - x0) / (x1 - x0)

        def interpolate(values):
            return ((1.0 - weight) * values[index] + weight * values[index + 1]).reshape(shape)

        return interpolate(self.z_grid[0]), interpolate(self.q_grid[0])

    def _cell_indices(self, q: torch.Tensor, grid: torch.Tensor):
        idx = torch.searchsorted(grid.contiguous(), q.contiguous(), right=True) - 1
        return idx.clamp(0, grid.numel() - 2)

    def hydraulic_geometry(self, x_m, water_level, t_s):
        """根据预测水位返回过流面积、水面宽和湿周"""
        area, width, perimeter = self.geometry(x_m, water_level, t_s)
        if self.area_calibration is None or self.width_calibration is None:
            return area, width, perimeter

        shape = torch.broadcast_shapes(x_m.shape, water_level.shape)
        x = x_m.expand(shape).reshape(-1).clamp(self.x_m[0], self.x_m[-1])
        ix = self._cell_indices(x, self.x_m)
        x0, x1 = self.x_m[ix], self.x_m[ix + 1]
        weight = (x - x0) / (x1 - x0)

        area_alpha, area_beta = self.area_calibration
        width_alpha, width_beta = self.width_calibration

        def interpolate(values):
            return ((1.0 - weight) * values[ix] + weight * values[ix + 1]).reshape(shape)

        raw_width = width.clamp_min(1.0e-6)
        area = (interpolate(area_alpha) * area + interpolate(area_beta)).clamp_min(1.0e-6)
        width = (interpolate(width_alpha) * width + interpolate(width_beta)).clamp_min(1.0e-6)
        perimeter = perimeter * (width / raw_width)
        return area, width, perimeter

    def calibrate_geometry(self, end_time_index):
        """用训练期数据校正 A(Z) 和 B(Z)，不使用测试期。"""
        x, t = self.full_grid_coordinates(0, end_time_index)
        z = self.z_grid[:end_time_index].reshape(-1)
        area_raw, width_raw, _ = self.geometry(x, z, t)
        nt = end_time_index
        nx = self.x_m.numel()
        area_raw = area_raw.reshape(nt, nx)
        width_raw = width_raw.reshape(nt, nx)
        area_true = self.area_grid[:end_time_index]
        width_true = self.width_grid[:end_time_index]

        def fit(raw, true):
            alpha = true.mean(dim=0) / raw.mean(dim=0).clamp_min(1.0e-6)
            beta = torch.zeros_like(alpha)
            return alpha.detach(), beta.detach()

        self.area_calibration = fit(area_raw, area_true)
        self.width_calibration = fit(width_raw, width_true)

    def manning(self, x_m: torch.Tensor):
        """预测期只使用起始时刻的空间糙率 n0(x)。"""
        shape = x_m.shape
        x = x_m.reshape(-1).clamp(self.x_m[0], self.x_m[-1])
        ix = self._cell_indices(x, self.x_m)
        x0, x1 = self.x_m[ix], self.x_m[ix + 1]
        weight = (x - x0) / (x1 - x0)
        value = (1.0 - weight) * self.initial_manning_n[ix] + weight * self.initial_manning_n[ix + 1]
        return value.reshape(shape)

    def full_grid_coordinates(self, start_time_index: int = 0, end_time_index: Optional[int] = None):
        t_values = self.t_s[start_time_index:end_time_index]
        tt, xx = torch.meshgrid(t_values, self.x_m, indexing="ij")
        return xx.reshape(-1), tt.reshape(-1)
