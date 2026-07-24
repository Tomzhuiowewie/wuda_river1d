# 假定河底高程不变
# 基于水深计算A、B、P

from pathlib import Path

import numpy as np
import torch


class CrossSectionGeometry:
    """由初始断面点生成可微的 A(Z)、B(Z)、P(Z)查询表。"""

    def __init__(self, profile_path, x_m, device="cpu", dtype=torch.float32, levels=1024):
        sections = self._read_profiles(profile_path)

        minimum_z = np.array([section["z"].min() for section in sections])
        maximum_z = np.array([section["z"].max() for section in sections])
        z_min = min(section["z"].min() for section in sections)
        z_max = max(section["z"].max() for section in sections)
        z_levels = np.linspace(z_min - 0.5, z_max + 1.0, levels)

        area = np.empty((len(sections), levels))
        width = np.empty_like(area)
        perimeter = np.empty_like(area)
        for i, section in enumerate(sections):
            area[i], width[i], perimeter[i] = self._rating_curve(section, z_levels)

        self.x_m = x_m
        self.minimum_z = torch.as_tensor(minimum_z, dtype=dtype, device=device)
        self.maximum_z = torch.as_tensor(maximum_z, dtype=dtype, device=device)
        self.z_levels = torch.as_tensor(z_levels, dtype=dtype, device=device)
        self.area_table = torch.as_tensor(area, dtype=dtype, device=device)
        self.width_table = torch.as_tensor(width, dtype=dtype, device=device)
        self.perimeter_table = torch.as_tensor(perimeter, dtype=dtype, device=device)

    @staticmethod
    def _read_profiles(profile_path):
        lines = Path(profile_path).expanduser().resolve().read_text().splitlines()
        sections = []
        i = 0
        while i < len(lines):
            words = lines[i].split()
            if len(words) != 1 or not words[0].startswith("CS"):
                i += 1
                continue

            section_id = int(words[0][2:])
            distance_m = float(lines[i + 1].split()[0]) * 1000.0
            point_count = int(lines[i + 2].split()[0])
            points = []
            for line in lines[i + 4:i + 4 + point_count]:
                point_id, xx, z, flag = line.split()
                points.append((int(point_id), float(xx), float(z), int(flag)))
            points.sort()
            sections.append({
                "section_id": section_id,
                "distance_m": distance_m,
                "x": np.array([point[1] for point in points]),
                "z": np.array([point[2] for point in points]),
                "flag": np.array([point[3] for point in points]),
            })
            i += 4 + point_count

        sections.sort(key=lambda section: section["section_id"])
        return sections

    @staticmethod
    def _rating_curve(section, water_levels):
        x0 = section["x"][:-1]
        x1 = section["x"][1:]
        z0 = section["z"][:-1]
        z1 = section["z"][1:]
        active = (section["flag"][:-1] == 0) & (section["flag"][1:] == 0)

        dx = x1 - x0    # 线段长度
        segment_length = np.hypot(dx, z1 - z0)  # 斜边长度
        d0 = water_levels[:, None] - z0[None, :]    # 
        d1 = water_levels[:, None] - z1[None, :]
        wet0 = np.maximum(d0, 0.0)
        wet1 = np.maximum(d1, 0.0)

        both_wet = (d0 >= 0.0) & (d1 >= 0.0)    # 判断线段的两个端点是否都在水下
        crossing = (d0 * d1 < 0.0)
        crossing_fraction = np.zeros_like(d0)
        np.divide(
            np.maximum(d0, d1), np.abs(d0 - d1),
            out=crossing_fraction, where=crossing,
        )
        fraction = np.where(both_wet, 1.0, crossing_fraction)   # 线段湿润比例
        fraction *= active[None, :]

        width = np.sum(dx[None, :] * fraction, axis=1)
        perimeter = np.sum(segment_length[None, :] * fraction, axis=1)
        area = np.sum(0.5 * (wet0 + wet1) * dx[None, :] * fraction, axis=1)
        return area, width, perimeter

    def _space_indices(self, x):
        index = torch.searchsorted(self.x_m.contiguous(), x.contiguous(), right=True) - 1
        return index.clamp(0, self.x_m.numel() - 2)

    def _level_indices(self, water_level):
        index = torch.searchsorted(self.z_levels.contiguous(), water_level.contiguous(), right=True) - 1
        return index.clamp(0, self.z_levels.numel() - 2)

    def minimum_stage(self, x_m):
        """断面点最低绝对高程"""
        shape = x_m.shape
        x = x_m.reshape(-1).clamp(self.x_m[0], self.x_m[-1])
        ix = self._space_indices(x)
        wx = (x - self.x_m[ix]) / (self.x_m[ix + 1] - self.x_m[ix])
        value = (1.0 - wx) * self.minimum_z[ix] + wx * self.minimum_z[ix + 1]
        return value.reshape(shape)

    def maximum_stage(self, x_m):
        """断面点最高绝对高程。"""
        shape = x_m.shape
        x = x_m.reshape(-1).clamp(self.x_m[0], self.x_m[-1])
        ix = self._space_indices(x)
        wx = (x - self.x_m[ix]) / (self.x_m[ix + 1] - self.x_m[ix])
        value = (1.0 - wx) * self.maximum_z[ix] + wx * self.maximum_z[ix + 1]
        return value.reshape(shape)

    def __call__(self, x_m, water_level, t_s=None):
        shape = torch.broadcast_shapes(x_m.shape, water_level.shape)
        x = x_m.expand(shape).reshape(-1).clamp(self.x_m[0], self.x_m[-1])
        z = water_level.expand(shape).reshape(-1)
        ix = self._space_indices(x)
        wx = (x - self.x_m[ix]) / (self.x_m[ix + 1] - self.x_m[ix])

        z = z.clamp(self.z_levels[0], self.z_levels[-1])
        iz = self._level_indices(z)
        wz = (z - self.z_levels[iz]) / (self.z_levels[iz + 1] - self.z_levels[iz])

        def interpolate(table):
            left = (1.0 - wz) * table[ix, iz] + wz * table[ix, iz + 1]
            right = (1.0 - wz) * table[ix + 1, iz] + wz * table[ix + 1, iz + 1]
            return ((1.0 - wx) * left + wx * right).reshape(shape)

        return interpolate(self.area_table), interpolate(self.width_table), interpolate(self.perimeter_table)

    def stage_from_area(self, x_m, area):
        """根据过流面积反算水位，使用与 A(Z) 查询表一致的线性插值。"""
        shape = torch.broadcast_shapes(x_m.shape, area.shape)
        x = x_m.expand(shape).reshape(-1).clamp(self.x_m[0], self.x_m[-1])
        a = area.expand(shape).reshape(-1)
        ix = self._space_indices(x)
        wx = (x - self.x_m[ix]) / (self.x_m[ix + 1] - self.x_m[ix])

        area_curve = (
            (1.0 - wx[:, None]) * self.area_table[ix]
            + wx[:, None] * self.area_table[ix + 1]
        )
        a = a.clamp(area_curve[:, 0], area_curve[:, -1])
        ia = torch.searchsorted(
            area_curve.contiguous(), a[:, None].contiguous(), right=True
        ).squeeze(1) - 1
        ia = ia.clamp(0, self.z_levels.numel() - 2)
        a0 = area_curve.gather(1, ia[:, None]).squeeze(1)
        a1 = area_curve.gather(1, (ia + 1)[:, None]).squeeze(1)
        wa = (a - a0) / (a1 - a0).clamp_min(1.0e-8)
        z = (1.0 - wa) * self.z_levels[ia] + wa * self.z_levels[ia + 1]
        return z.reshape(shape)
