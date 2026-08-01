from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class TrainConfig:
    # 1. 输入数据与输出位置
    data_path: Path
    cross_section_path: Path
    output_dir: Path = Path("outputs/river1d_pinn")

    # 2. 网络结构
    network_type: str = "mlp"
    # MLP：每个数是一个隐藏层宽度；Fourier-ResNet：元组长度是残差块数量。
    hidden_layers: Tuple[int, ...] = (64, 64, 64, 64)
    activation: str = "gelu"

    # 3. Fourier-ResNet 网络专用参数
    fourier_features: int = 128
    fourier_sigma_x: float = 10.0
    fourier_sigma_t: float = 0.68
    fourier_head_width: int = 16
    dropout: float = 0.0

    # 4. 训练与采样
    epochs: int = 30000
    num_physics_points: int = 2000 # 每一轮训练采样多少物理点
    use_geometry_calibration: bool = False
    learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-6
    # 学习率
    scheduler_type: str = "warmup_exp"  # "cosine"、"warmup_exp"、"constant"
    warmup_epochs: int = 200
    lr_decay: float = 0.98

    # 采样策略
    sampling_strategy: str = "random"  # "random" 或 "causal"
    causal_start_fraction: float = 0.05
    causal_warmup_epochs: int = 100

    # 5. 损失函数权重
    data_weight: float = 1.0    # 手工权重
    z_weight: float = 1.0
    q_weight: float = 5.0

    physics_weight: float = 8.0e-2
    mass_weight: float = 1.0
    momentum_weight: float = 1.0

    ntk_weighting: bool = False # 是否启用 NTK 自动权重
    ntk_update_every: int = 20
    ntk_momentum: float = 0.9
    ntk_min_weight: float = 1.0e-8
    ntk_max_weight: float = 1.0e2

    interior_section_count: int = 0
    interior_section_ids: Tuple[int, ...] = (23, 46, 69)
    interior_data_weight: float = 1.0

    # 6. 运行、保存与复现
    grad_clip: float = 1.0
    print_every: int = 20
    save_every: int = 1_000
    seed: int = 2032
    dtype: str = "float32"
    device: str = "auto"

    def to_dict(self):
        result = asdict(self)
        result["data_path"] = str(self.data_path)
        result["cross_section_path"] = str(self.cross_section_path)
        result["output_dir"] = str(self.output_dir)
        result["hidden_layers"] = list(self.hidden_layers)
        result["interior_section_ids"] = list(self.interior_section_ids)
        return result


# 只在这里修改训练参数。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = TrainConfig(
    data_path=PROJECT_ROOT / "data/formodel/2024/FCSLPF.csv",
    cross_section_path=(
        PROJECT_ROOT
        / "data/1D_LYR_20260629/OUTPUT-2024汛期-东霞院-利津/OUTPUT/Qob5500.00/INICSProf.TXT"
    ),
)
