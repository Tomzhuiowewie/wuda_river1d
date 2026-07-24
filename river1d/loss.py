from __future__ import annotations

import torch
from torch import nn


from data import RiverData


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


class PINNResidual(nn.Module):
    """Strong-form 1D Saint-Venant residual for a traditional PINN."""

    def __init__(
        self,
        data: RiverData,
        gravity: float = 9.81,
        epsilon: float = 1.0e-6,
    ):
        super().__init__()
        self.data = data
        self.gravity = gravity
        self.epsilon = epsilon

    def residual(self, model, x, t):
        """返回无量纲 mass/momentum PDE residual。"""
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        water_level, discharge = model(x, t)
        area, _, perimeter = self.data.hydraulic_geometry(x, water_level, t)

        area_t = _grad(area, t)
        discharge_t = _grad(discharge, t)
        discharge_x = _grad(discharge, x)
        flux_adv_x = _grad(discharge.square() / area, x)
        water_level_x = _grad(water_level, x)

        manning_n = self.data.manning(x)
        radius = area / perimeter.clamp_min(self.epsilon)
        friction = (
            manning_n.square()
            * discharge
            * discharge.abs()
            / (area.square() * radius.clamp_min(self.epsilon).pow(4.0 / 3.0))
        )

        mass = area_t + discharge_x
        momentum = (
            discharge_t
            + flux_adv_x
            + self.gravity * area * water_level_x
            + self.gravity * area * friction
        )

        mass_scale = self.data.scales.discharge_m3_s / self.data.scales.length_m
        momentum_scale = (
            self.gravity
            * self.data.scales.area_m2
            * self.data.scales.friction_slope
        )
        mass = mass / max(mass_scale, self.epsilon)
        momentum = momentum / max(momentum_scale, self.epsilon)

        return mass, momentum

    def forward(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor):
        return self.residual(model, x, t)

    def loss(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor):
        mass, momentum = self.residual(model, x, t)
        return mass.square().mean(), momentum.square().mean()

    def raw_loss(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor):
        mass_loss, momentum_loss = self.loss(model, x, t)
        mass_scale = self.data.scales.discharge_m3_s / self.data.scales.length_m
        momentum_scale = (
            self.gravity
            * self.data.scales.area_m2
            * self.data.scales.friction_slope
        )
        return mass_loss * max(mass_scale, self.epsilon) ** 2, momentum_loss * max(momentum_scale, self.epsilon) ** 2


def data_loss(model, data, initial_boundary_data):
    z_pred, q_pred = model(initial_boundary_data["x"], initial_boundary_data["t"])
    z_loss = ((z_pred - initial_boundary_data["z"]) / data.scales.depth_m).square().mean()
    q_loss = ((q_pred - initial_boundary_data["q"]) / data.scales.discharge_m3_s).square().mean()
    return z_loss, q_loss
