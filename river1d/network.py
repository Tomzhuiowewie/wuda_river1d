from __future__ import annotations

import torch
from torch import nn

from data import RiverData


activations = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


def initialize_weights(model, gain=1.0):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)

    
def set_output_scales(model, data):
    """从现有数据中设置坐标归一化尺度和 Z/Q 输出尺度。"""
    model.data = data
    model.geometry = data.geometry

    model.register_buffer("x_min_m", data.x_m[0].clone())
    model.register_buffer("length_m", (data.x_m[-1] - data.x_m[0]).clone())
    model.register_buffer("t_min_s", data.t_s[0].clone())
    model.register_buffer("duration_s", (data.t_s[-1] - data.t_s[0]).clone())

    q_range = data.q_grid.max(dim=0).values - data.q_grid.min(dim=0).values
    q_margin = 0.2 * q_range

    model.register_buffer("x_grid", data.x_m.clone())
    model.register_buffer("z_min_curve", data.z_grid.min(dim=0).values.clone())
    model.register_buffer("z_max_curve", data.z_grid.max(dim=0).values.clone())
    model.register_buffer("q_min_curve", (data.q_grid.min(dim=0).values - q_margin).clamp_min(1.0e-6))
    model.register_buffer("q_max_curve", data.q_grid.max(dim=0).values + q_margin)


def interpolate_by_x(model, x, values):
    idx = torch.searchsorted(model.x_grid.contiguous(), x.contiguous(), right=True) - 1
    idx = idx.clamp(0, model.x_grid.numel() - 2)
    x0 = model.x_grid[idx]
    x1 = model.x_grid[idx + 1]
    weight = (x - x0) / (x1 - x0)
    return (1.0 - weight) * values[idx] + weight * values[idx + 1]


class FlowNetwork(nn.Module):
    """用一个共享 MLP 同时预测水位 Z(x,t) 和流量 Q(x,t)。"""

    def __init__(self, data, hidden_layers, activation, initial_boundary_data,
                 dropout=0.0):
        super().__init__()

        widths = list(hidden_layers)
        layers = []
        in_features = 2

        for out_features in widths:
            layers.extend((nn.Linear(in_features, out_features), activations[activation]()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, 2))
        self.network = nn.Sequential(*layers)
        set_output_scales(self, data)
        initialize_weights(self)

    def _coordinates(self, x_m, t_s): # 归一化坐标
        shape = torch.broadcast_shapes(x_m.shape, t_s.shape)
        x = x_m.expand(shape)
        t = t_s.expand(shape)
        x_hat = 2.0 * (x - self.x_min_m) / self.length_m - 1.0
        t_hat = 2.0 * (t - self.t_min_s) / self.duration_s - 1.0
        return shape, x.reshape(-1), t.reshape(-1), x_hat.reshape(-1), t_hat.reshape(-1)

    def _features(self, x, t, x_hat, t_hat):    # 构造输入特征
        return torch.stack((x_hat, t_hat), dim=-1)

    def forward(self, x_m, t_s):
        shape, x, t, x_hat, t_hat = self._coordinates(x_m, t_s)
        raw = self.network(self._features(x, t, x_hat, t_hat))

        z_min = interpolate_by_x(self, x, self.z_min_curve)
        z_max = interpolate_by_x(self, x, self.z_max_curve)
        q_min = interpolate_by_x(self, x, self.q_min_curve)
        q_max = interpolate_by_x(self, x, self.q_max_curve)

        z_bed, z_top = self.geometry.stage_bounds(x)
        z_lower = torch.maximum(z_bed + 1.0e-3, z_min)
        z_upper = torch.minimum(z_top, z_max)
        z_upper = torch.maximum(z_upper, z_lower + 1.0e-3)

        z = z_lower + (z_upper - z_lower) * torch.sigmoid(raw[:, 0])
        z = torch.minimum(torch.maximum(z, z_lower), z_upper)

        q = q_min + (q_max - q_min) * torch.sigmoid(raw[:, 1])
        return z.reshape(shape), q.reshape(shape)


class AreaDischargeFlowNetwork(FlowNetwork):
    """共享 MLP 直接预测守恒变量 A(x,t) 和 Q(x,t)。"""

    def __init__(self, data, hidden_layers=(64, 64, 64, 64, 64), activation="tanh",
                 initial_boundary_data=None, dropout=0.0):
        super().__init__(
            data, hidden_layers, activation, initial_boundary_data,
            dropout,
        )
        boundary = data.initial_boundary_data(None) if initial_boundary_data is None else initial_boundary_data
        q_min, q_max = float(boundary["q"].min()), float(boundary["q"].max())
        q_range = max(q_max - q_min, 1.0e-6)
        self.register_buffer("q_lower", torch.tensor(max(0.0, q_min - 0.5 * q_range)))
        self.register_buffer("q_upper", torch.tensor(q_max + 0.5 * q_range))

    def forward(self, x_m, t_s):
        shape, x, t, x_hat, t_hat = self._coordinates(x_m, t_s)
        raw = self.network(self._features(x, t, x_hat, t_hat))
        z_lower, z_upper = self.geometry.stage_bounds(x)
        z_lower = z_lower + 0.05
        z_upper = torch.maximum(z_upper, z_lower + 1.0e-3)
        area_lower, _, _ = self.geometry(x, z_lower)
        area_upper, _, _ = self.geometry(x, z_upper)
        area_free = area_lower + (area_upper - area_lower) * torch.sigmoid(raw[:, 0])
        discharge_free = self.q_lower + (self.q_upper - self.q_lower) * torch.sigmoid(raw[:, 1])
        area = area_free.clamp_min(1.0e-6)
        discharge = discharge_free
        z = self.geometry.stage_from_area(x, area)
        return z.reshape(shape), discharge.reshape(shape)


class _GaussianFourierEncoding(nn.Module):
    def __init__(self, num_features=128, sigma_x=10.0, sigma_t=1.0):
        super().__init__()
        half = num_features // 2
        self.register_buffer("b_x", torch.randn(1, half) * sigma_x)
        self.register_buffer("b_t", torch.randn(1, half) * sigma_t)
        self.out_dim = 2 * num_features + 2

    def forward(self, xt):
        x_hat, t_hat = xt[:, 0:1], xt[:, 1:2]
        xp = 2.0 * torch.pi * (x_hat @ self.b_x)
        tp = 2.0 * torch.pi * (t_hat @ self.b_t)
        return torch.cat((xp.sin(), xp.cos(), tp.sin(), tp.cos(), x_hat, t_hat), dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, width, activation, dropout):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation1 = activation()
        self.activation2 = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        hidden = self.dropout(self.activation1(self.linear1(inputs)))
        return inputs + self.activation2(self.linear2(hidden))


class FourierResNetFlowNetwork(nn.Module):
    """Fourier-ResNet 水位、流量网络。"""

    def __init__(self, data, hidden_layers=(20, 20, 20, 20, 20, 20), activation="tanh",
                 fourier_features=128, sigma_x=10.0, sigma_t=1.0,
                 head_width=None, dropout=0.0, initial_boundary_data=None):
        super().__init__()
        widths = list(hidden_layers)
        width = widths[0]
        activation_type = activations[activation]
        self.fourier_encoding = _GaussianFourierEncoding(fourier_features, sigma_x, sigma_t)
        self.input_layer = nn.Linear(self.fourier_encoding.out_dim, width)
        self.input_activation = activation_type()
        self.residual_blocks = nn.ModuleList(
            _ResidualBlock(width, activation_type, dropout) for _ in widths
        )
        head_width = max(1, width // 2) if head_width is None else head_width
        self.z_head = nn.Sequential(
            nn.Linear(width, head_width), activation_type(), nn.Linear(head_width, 1)
        )
        self.q_head = nn.Sequential(
            nn.Linear(width, head_width), activation_type(), nn.Linear(head_width, 1)
        )
        set_output_scales(self, data)
        initialize_weights(self, gain=0.8)

    def forward(self, x_m, t_s):
        shape = torch.broadcast_shapes(x_m.shape, t_s.shape)
        x, t = x_m.expand(shape), t_s.expand(shape)
        x_hat = 2.0 * (x - self.x_min_m) / self.length_m - 1.0
        t_hat = 2.0 * (t - self.t_min_s) / self.duration_s - 1.0
        coordinates = torch.stack((x_hat.reshape(-1), t_hat.reshape(-1)), dim=-1)
        features = self.fourier_encoding(coordinates)
        hidden = self.input_activation(self.input_layer(features))
        for block in self.residual_blocks:
            hidden = block(hidden)
        z_raw = self.z_head(hidden).squeeze(-1)
        q_raw = self.q_head(hidden).squeeze(-1)
        x_flat = x.reshape(-1)
        z_min = interpolate_by_x(self, x_flat, self.z_min_curve)
        z_max = interpolate_by_x(self, x_flat, self.z_max_curve)
        q_min = interpolate_by_x(self, x_flat, self.q_min_curve)
        q_max = interpolate_by_x(self, x_flat, self.q_max_curve)

        z_bed, z_top = self.geometry.stage_bounds(x_flat)
        z_lower = torch.maximum(z_bed + 1.0e-3, z_min)
        z_upper = torch.minimum(z_top, z_max)
        z_upper = torch.maximum(z_upper, z_lower + 1.0e-3)
        z = z_lower + (z_upper - z_lower) * torch.sigmoid(z_raw)
        z = torch.minimum(torch.maximum(z, z_lower), z_upper)
        q = q_min + (q_max - q_min) * torch.sigmoid(q_raw)
        return z.reshape(shape), q.reshape(shape)


def build_flow_network(data, hidden_layers, activation, network_type="fourier_resnet", *,
                       fourier_features=128, sigma_x=10.0, sigma_t=1.0,
                       head_width=None, dropout=0.0, initial_boundary_data=None):
    if network_type == "mlp":
        return FlowNetwork(
            data, hidden_layers, activation, initial_boundary_data,
            dropout,
        )
    if network_type == "mlp_aq":
        return AreaDischargeFlowNetwork(
            data, hidden_layers, activation, initial_boundary_data,
            dropout,
        )
    if network_type == "fourier_resnet":
        return FourierResNetFlowNetwork(
            data, hidden_layers, activation, fourier_features=fourier_features,
            sigma_x=sigma_x, sigma_t=sigma_t, head_width=head_width, dropout=dropout,
            initial_boundary_data=initial_boundary_data,
        )
    raise ValueError("network_type must be 'mlp', 'mlp_aq' or 'fourier_resnet'")
