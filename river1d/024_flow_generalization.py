"""验证集幅值模态校准的预测锚点 AL-PINN。

固化实验 156。核心模型和 GPR 锚点与 023 相同，但增加 8 个空间
幅值模态，并用 40 个验证场景的完整 Q 场拟合模态系数。因此本脚本
不是严格单断面监督对照，结果必须标注“validation-calibrated”。
已复现实验结果约为 Z L2=3.706%、Q L2=9.928%。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODES = load_module("modes_024", "experiments/064_flow_amplitude_modes/search.py")
ENRICHED = load_module(
    "enriched_024", "experiments/114_zero_init_trace_enrichment/run.py"
)
ANCHOR = load_module("anchor_024", "experiments/109_predicted_anchor/run.py")
BASE, DEVICE = MODES.B, MODES.DEVICE
SEED = 2032
OUTPUT = ROOT / "outputs/flow_generalization_024/validation_calibrated_modes"
CHECKPOINT = ROOT / "outputs/experiments/114_zero_init_trace_enrichment/best.pt"
P023 = load_module("formal_023_training", "river1d/023_flow_generalization.py")


def split_training_validation(dataset):
    order = np.random.default_rng(SEED).permutation(len(dataset.train_files))
    return (
        [dataset.train_files[i] for i in order[40:]],
        [dataset.train_files[i] for i in order[:40]],
    )


@torch.no_grad()
def collect(model, dataset, standardizer, full_conditions, paths, time_count):
    rows = []
    for path in paths:
        raw = dataset.load_scenario(path)
        time_ids = np.linspace(0, len(raw["t"]) - 1, time_count).round().astype(int)
        scenario = {
            "t": raw["t"][time_ids],
            "x": raw["x"],
            "z": raw["z"][time_ids],
            "q": raw["q"][time_ids],
        }
        time_grid, space_grid = np.meshgrid(
            scenario["t"], scenario["x"], indexing="ij"
        )
        condition = BASE.to_tensor(
            standardizer.transform(dataset.get_condition_vector(path))
        )
        extra = BASE.to_tensor(full_conditions.get(dataset, path))
        z, q = model(
            BASE.to_tensor(space_grid.ravel()),
            BASE.to_tensor(time_grid.ravel()),
            condition,
            extra,
        )
        rows.append(
            (
                scenario,
                z.cpu().numpy().reshape(time_grid.shape),
                q.cpu().numpy().reshape(time_grid.shape),
                standardizer.transform(dataset.get_condition_vector(path)),
            )
        )
    return rows


def amplitude_features(scenario, discharge, predicted_anchor, q_scale):
    """构造不读取测试 x[46] 真值的 8 个空间幅值模态。"""
    x = scenario["x"]
    anchor_index = 46
    left = (x[: anchor_index + 1] - x[0]) / max(
        x[anchor_index] - x[0], 1.0e-6
    )
    right = (x[anchor_index:] - x[anchor_index]) / max(
        x[-1] - x[anchor_index], 1.0e-6
    )
    left_shape = 4.0 * left * (1.0 - left)
    right_shape = 4.0 * right * (1.0 - right)
    upstream = scenario["q"][:, 0]
    difference = (predicted_anchor - upstream) / q_scale
    anchor_amplitude = predicted_anchor / q_scale
    upstream_amplitude = upstream / q_scale
    downstream_difference = (discharge[:, -1] - predicted_anchor) / q_scale
    features = np.zeros(
        (len(scenario["t"]), len(x), 8), dtype=np.float32
    )
    features[:, : anchor_index + 1, 0] = difference[:, None] * left_shape
    features[:, : anchor_index + 1, 1] = anchor_amplitude[:, None] * left_shape
    features[:, : anchor_index + 1, 2] = upstream_amplitude[:, None] * left_shape
    features[:, anchor_index:, 3] = downstream_difference[:, None] * right_shape
    features[:, anchor_index:, 4] = anchor_amplitude[:, None] * right_shape
    features[:, anchor_index:, 5] = upstream_amplitude[:, None] * right_shape
    features[:, : anchor_index + 1, 6] = (
        difference[:, None] * left_shape * (2.0 * left[None, :] - 1.0)
    )
    features[:, anchor_index:, 7] = (
        downstream_difference[:, None]
        * right_shape
        * (2.0 * right[None, :] - 1.0)
    )
    return features.reshape(-1, 8)


def predict_anchor(row, fitted, source_time, length):
    scenario, _, _, condition = row
    mean, std, training_x, alpha, time_ids = fitted
    kernel = ANCHOR.rbf(
        ((condition - mean) / std)[None, :], training_x, length
    )[0]
    return np.interp(scenario["t"], source_time[time_ids], kernel @ alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="先训练 enriched AL-PINN 再进行024校准测试")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA 不可用，拒绝将 CPU 结果混入正式实验")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    dataset = BASE.ScenarioDataset(BASE.cache_directory, cache_limit=600)
    standardizer, scales = BASE.fit_condition_normalizer_and_scales(dataset)
    training_paths, validation_paths = split_training_validation(dataset)
    full_conditions = ENRICHED.FullTraceConditions(dataset, training_paths, 128)
    source_time = dataset.load_scenario(training_paths[0])["t"]
    reference = dataset.load_scenario(training_paths[0])
    geometry = BASE.CrossSectionGeometry(
        BASE.cross_section_profile_path,
        BASE.to_tensor(reference["x"]),
        device=DEVICE,
    )
    base = BASE.OperatorPINN(scales, geometry, code_dim=128, hidden=256).to(DEVICE)
    model = ENRICHED.EnrichedPINN(base, len(full_conditions.mean)).to(DEVICE)
    checkpoint = P023.train_checkpoint(args.epochs) if args.train else CHECKPOINT
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    training_rows = ANCHOR.collect(base, dataset, standardizer, training_paths)
    validation_rows = collect(
        model, dataset, standardizer, full_conditions, validation_paths, 128
    )
    candidates = []
    for length in (1.0, 2.0, 4.0):
        fitted = ANCHOR.fit_anchor(training_rows, length, 1.0e-3)
        predicted_z, predicted_q, true_z, true_q, feature_rows = [], [], [], [], []
        for row in validation_rows:
            scenario, z, q, _ = row
            anchor = predict_anchor(row, fitted, source_time, length)
            projected_z, projected_q = ANCHOR.input_project(
                scenario, z, q, anchor, 0.02, 0.08, 0.12, 0.08, 0.15
            )
            predicted_z.append(projected_z.ravel())
            predicted_q.append(projected_q.ravel())
            true_z.append(scenario["z"].ravel())
            true_q.append(scenario["q"].ravel())
            feature_rows.append(
                amplitude_features(scenario, projected_q, anchor, scales.q_std)
            )
        pz, pq, tz, tq, features = map(
            np.concatenate,
            (predicted_z, predicted_q, true_z, true_q, feature_rows),
        )
        # 注意：这里使用了验证集完整 Q 场，故不属于严格单断面监督。
        beta = np.linalg.lstsq(features, tq - pq, rcond=None)[0]
        candidates.append(
            {
                "length": length,
                "beta": beta.tolist(),
                "val_z_l2": MODES.S.l2(pz, tz),
                "val_q_l2": MODES.S.l2(pq + features @ beta, tq),
            }
        )

    candidates.sort(key=lambda row: max(row["val_z_l2"], row["val_q_l2"]))
    selected = candidates[0]
    length = selected["length"]
    beta = np.asarray(selected["beta"])
    fitted = ANCHOR.fit_anchor(training_rows, length, 1.0e-3)

    test_count = len(dataset.load_scenario(dataset.test_files[0])["t"])
    test_rows = collect(
        model, dataset, standardizer, full_conditions, dataset.test_files, test_count
    )
    predicted_z, predicted_q, true_z, true_q = [], [], [], []
    for row in test_rows:
        scenario, z, q, _ = row
        anchor = predict_anchor(row, fitted, source_time, length)
        projected_z, projected_q = ANCHOR.input_project(
            scenario, z, q, anchor, 0.02, 0.08, 0.12, 0.08, 0.15
        )
        features = amplitude_features(
            scenario, projected_q, anchor, scales.q_std
        )
        correction = (features @ beta).reshape(projected_q.shape)
        predicted_z.append(projected_z.ravel())
        predicted_q.append((projected_q + correction).ravel())
        true_z.append(scenario["z"].ravel())
        true_q.append(scenario["q"].ravel())
    pz, pq, tz, tq = map(
        np.concatenate, (predicted_z, predicted_q, true_z, true_q)
    )
    result = {
        "method": "enriched AL-PINN + predicted anchor + 8 amplitude modes",
        "selected": selected,
        "test_z_l2": MODES.S.l2(pz, tz),
        "test_q_l2": MODES.S.l2(pq, tq),
        "strict_single_section_supervision": False,
        "validation_full_field_calibration": True,
        "uses_test_internal_label": False,
        "anchor_training_labels": "training x[46] only",
        "checkpoint": str(checkpoint),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation_candidates.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8"
    )
    (OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
