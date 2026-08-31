"""023: fixed three-stage single-section PINN research experiment.

The script deliberately contains one protocol only: 30-epoch supervised
warm-up, 8-epoch AL-PINN training, and 20-epoch joint base/enrichment
fine-tuning. The only internal training label is x[46]. Model selection uses
the fixed validation split; the 100 test scenarios are evaluated once at the
end. No historical checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = __import__("river1d.021_flow_generalization", fromlist=["*"])
DEVICE = BASE.compute_device
_ENGINE_SPEC = importlib.util.spec_from_file_location(
    "isolated_023_engine", ROOT / "experiments/023_single_section_pinn/engine.py"
)
assert _ENGINE_SPEC.loader is not None
ENGINE = importlib.util.module_from_spec(_ENGINE_SPEC)
sys.modules[_ENGINE_SPEC.name] = ENGINE
_ENGINE_SPEC.loader.exec_module(ENGINE)
ANCHOR_SPEC = importlib.util.spec_from_file_location(
    "anchor_023_full", ROOT / "experiments/109_predicted_anchor/run.py"
)
ANCHOR = importlib.util.module_from_spec(ANCHOR_SPEC)
assert ANCHOR_SPEC.loader is not None
ANCHOR_SPEC.loader.exec_module(ANCHOR)

SEED = 2032
# Results are written below the dedicated 023 reproduction directory.
OUT_ROOT = ROOT / "outputs/flow_generalization_023/reproduction"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def split_paths(dataset, seed: int):
    order = np.random.default_rng(seed).permutation(len(dataset.train_files))
    validation = [dataset.train_files[i] for i in order[:40]]
    training = [dataset.train_files[i] for i in order[40:]]
    return training, validation


def fit_train_statistics(dataset, paths):
    conditions, z_values, q_values = [], [], []
    for path in paths:
        scenario = dataset.load_scenario(path)
        conditions.append(dataset.get_condition_vector(path))
        z_values.append(scenario["z"].reshape(-1))
        q_values.append(scenario["q"].reshape(-1))
    reference = dataset.load_scenario(paths[0])
    standardizer = BASE.ConditionStandardizer(np.stack(conditions))
    scales = BASE.PhysicalScales(
        reference, np.concatenate(z_values), np.concatenate(q_values)
    )
    return standardizer, scales


class FullTraceConditions:
    """440-dimensional known-condition encoder input, fit on train paths only."""

    def __init__(self, dataset, paths, ntime=128):
        self.ntime = ntime
        values = []
        for path in paths:
            s = dataset.load_scenario(path)
            ti = np.linspace(0, len(s["t"]) - 1, ntime).round().astype(int)
            values.append(np.concatenate((s["q"][ti, 0], s["z"][ti, -1], s["z"][0], s["q"][0])))
        raw = np.asarray(values, dtype=np.float32)
        self.mean = raw.mean(0).astype(np.float32)
        self.std = np.maximum(raw.std(0), 1.0e-6).astype(np.float32)

    def get(self, dataset, path):
        s = dataset.load_scenario(path)
        ti = np.linspace(0, len(s["t"]) - 1, self.ntime).round().astype(int)
        raw = np.concatenate((s["q"][ti, 0], s["z"][ti, -1], s["z"][0], s["q"][0]))
        return ((raw - self.mean) / self.std).astype(np.float32)


class EnrichedOperator(nn.Module):
    """Base 192-condition operator plus a 440-condition correction branch."""

    def __init__(self, base, extra_dim=440):
        super().__init__()
        self.base = base
        self.scales, self.geometry = base.scales, base.geometry
        self.extra = nn.Sequential(
            nn.LayerNorm(extra_dim),
            nn.Linear(extra_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128),
        )
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)
        self._extra = None

    def set_extra(self, extra):
        self._extra = extra

    def forward(self, x, t, condition):
        if condition.ndim == 1:
            condition = condition.expand(x.numel(), -1)
        if self._extra is None:
            raise RuntimeError("EnrichedOperator.set_extra must be called before forward")
        extra = self._extra
        if extra.ndim == 1:
            extra = extra.expand(x.numel(), -1)
        elif extra.shape[0] != x.numel():
            extra = extra[:1].expand(x.numel(), -1)
        xt = torch.stack((self.scales.x_normalize(x), self.scales.t_normalize(t)), 1)
        branch = self.base.branch(condition) + self.extra(extra)
        raw = self.base.head(torch.cat((self.base.trunk(xt), branch), 1))
        bed, _ = self.geometry.stage_bounds(x)
        z = bed + 5.0e-4 + 5.0 * torch.nn.functional.softplus(raw[:, 0])
        q = self.scales.q_denormalize(raw[:, 1])
        return z, q


def model_condition(model, extra):
    if isinstance(model, EnrichedOperator):
        model.set_extra(extra)


def weighted_terms(terms, q_weight, bc_q_weight, data_q_multiplier):
    factors = terms.new_tensor((1.0, q_weight, q_weight * bc_q_weight,
                                1.0, 1.0, q_weight * data_q_multiplier,
                                1.0, 1.0))
    return terms * factors


def validation_metrics(model, dataset, standardizer, full_conditions, paths, seed, points=2048):
    model.eval()
    rng = np.random.default_rng(seed)
    pred_z, pred_q, true_z, true_q = [], [], [], []
    with torch.no_grad():
        for path in paths:
            s = dataset.load_scenario(path)
            n = min(points, len(s["t"]) * len(s["x"]))
            ti = rng.integers(0, len(s["t"]), n)
            xi = rng.integers(0, len(s["x"]), n)
            c = BASE.to_tensor(standardizer.transform(dataset.get_condition_vector(path)))
            e = BASE.to_tensor(full_conditions.get(dataset, path)) if full_conditions else None
            model_condition(model, e)
            z, q = model(BASE.to_tensor(s["x"][xi]), BASE.to_tensor(s["t"][ti]), c)
            pred_z.append(z.cpu().numpy()); pred_q.append(q.cpu().numpy())
            true_z.append(s["z"][ti, xi]); true_q.append(s["q"][ti, xi])
    pz, pq, tz, tq = map(np.concatenate, (pred_z, pred_q, true_z, true_q))
    return metric_arrays(pz, pq, tz, tq)


def metric_arrays(pz, pq, tz, tq):
    return {
        "z_l2": float(100 * np.linalg.norm(pz - tz) / max(np.linalg.norm(tz), 1e-12)),
        "q_l2": float(100 * np.linalg.norm(pq - tq) / max(np.linalg.norm(tq), 1e-12)),
        "z_nse": float(1 - np.sum((pz - tz) ** 2) / max(np.sum((tz - tz.mean()) ** 2), 1e-12)),
        "q_nse": float(1 - np.sum((pq - tq) ** 2) / max(np.sum((tq - tq.mean()) ** 2), 1e-12)),
    }




def train_epoch(model, dataset, standardizer, full_conditions, geometry,
                pde_scales, paths, options, epoch, seed, optimizer, al_state,
                objective_kind, rng_seed_base=None):
    model.train()
    rng = np.random.default_rng(
        (seed * 100000 if rng_seed_base is None else rng_seed_base) + epoch)
    order = rng.permutation(len(paths))
    records = []
    for start in range(0, len(order), options.batch_scenarios):
        optimizer.zero_grad(set_to_none=True)
        batch_terms = []
        for index in order[start:start + options.batch_scenarios]:
            path = paths[index]
            scenario = dataset.load_scenario(path)
            condition = BASE.to_tensor(
                standardizer.transform(dataset.get_condition_vector(path)))
            extra = (BASE.to_tensor(full_conditions.get(dataset, path))
                     if full_conditions is not None else None)
            model_condition(model, extra)
            batch_terms.append(ENGINE.loss_terms(
                model, geometry, pde_scales, scenario, condition,
                options, rng, epoch))
        terms = torch.stack(batch_terms).mean(0)
        weighted = weighted_terms(
            terms, options.q_weight, options.bc_q_weight,
            options.data_q_multiplier)
        grouped = ENGINE.groups(weighted)

        if objective_kind == "supervised":
            objective = grouped[:3].sum()
            objective.backward()
            objective_value = objective.detach()
            pde_share = 0.0
            adaptive_weights = torch.tensor(
                [1., 1., options.bc_q_weight, options.bc_q_weight,
                 1., options.q_weight * options.data_q_multiplier, 0., 0.],
                device=DEVICE)
        elif objective_kind == "al":
            constraints = grouped[:3]
            lam, mu = al_state
            objective = (options.al_pde_multiplier * grouped[3]
                         + torch.dot(lam.detach(), constraints)
                         + 0.5 * (mu * constraints.square()).sum())
            objective.backward()
            with torch.no_grad():
                lam.add_(mu * constraints.detach()).clamp_(
                    -options.al_mu_max, options.al_mu_max)
            objective_value = objective.detach()
            pde_share = float(
                (grouped[3] / grouped.sum().clamp_min(1e-12)).detach().cpu())
            adaptive_weights = torch.ones(8, device=DEVICE)
        else:
            raise ValueError(f"unknown objective kind: {objective_kind}")

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        records.append({
            "objective": float(objective_value.cpu()),
            "raw_terms": terms.detach().cpu().tolist(),
            "weighted_terms": weighted.detach().cpu().tolist(),
            "grouped_terms": grouped.detach().cpu().tolist(),
            "pde_share": pde_share,
            "adaptive_weights": adaptive_weights.detach().cpu().tolist(),
        })
    return {
        "objective": float(np.mean([r["objective"] for r in records])),
        "raw_terms": np.asarray(
            [r["raw_terms"] for r in records]).mean(0).tolist(),
        "pde_share": float(np.mean([r["pde_share"] for r in records])),
        "projection_ratio": None,
        "adaptive_weights": np.asarray(
            [r["adaptive_weights"] for r in records]).mean(0).tolist(),
    }


def save_state(path, model, optimizer, al_state, epoch, best_score, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "al_lambda": al_state[0].detach().cpu(), "epoch": epoch,
        "best_score": best_score, "record": record,
    }, path)


def collect_rows(model, dataset, standardizer, full_conditions, paths):
    model.eval()
    rows = []
    with torch.no_grad():
        for path in paths:
            scenario = dataset.load_scenario(path)
            tg, xg = np.meshgrid(
                scenario["t"], scenario["x"], indexing="ij")
            condition = BASE.to_tensor(
                standardizer.transform(dataset.get_condition_vector(path)))
            model_condition(
                model,
                BASE.to_tensor(full_conditions.get(dataset, path))
                if full_conditions is not None else None)
            fx, ft = BASE.to_tensor(xg.ravel()), BASE.to_tensor(tg.ravel())
            pred_z, pred_q = [], []
            for start in range(0, fx.numel(), 8192):
                z, q = model(fx[start:start + 8192],
                             ft[start:start + 8192], condition)
                pred_z.append(z.cpu().numpy())
                pred_q.append(q.cpu().numpy())
            rows.append((
                scenario,
                np.concatenate(pred_z).reshape(tg.shape),
                np.concatenate(pred_q).reshape(tg.shape),
                standardizer.transform(
                    dataset.get_condition_vector(path))))
    return rows


def select_anchor(train_rows, validation_rows):
    candidates = []
    for length in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for ridge in (1e-3, 1e-2, 0.1, 1.0):
            fitted = ANCHOR.fit_anchor(train_rows, length, ridge)
            for left in (0.04, 0.08, 0.12, 0.2):
                for right in (0.08, 0.15, 0.2, 0.3):
                    pars = (length, 0.02, 0.08, 0.12, left, right)
                    z, q = ANCHOR.score(
                        validation_rows, fitted, pars, True, stride=20)
                    candidates.append({
                        "length": length, "ridge": ridge,
                        "left_scale": left, "right_scale": right,
                        "val_z_l2": z, "val_q_l2": q})
    candidates.sort(key=lambda item: max(
        item["val_z_l2"], item["val_q_l2"]))
    return candidates[0], candidates


def select_no_anchor(validation_rows):
    candidates = []
    for et in (0.01, 0.02, 0.05, 0.1):
        for eu in (0.04, 0.08, 0.12, 0.2):
            for ed in (0.04, 0.08, 0.12, 0.2):
                pars = (1.0, et, eu, ed, 0.08, 0.15)
                z, q = ANCHOR.score(
                    validation_rows, None, pars, False, stride=20)
                candidates.append({
                    "et": et, "eu": eu, "ed": ed,
                    "val_z_l2": z, "val_q_l2": q})
    candidates.sort(key=lambda item: max(
        item["val_z_l2"], item["val_q_l2"]))
    return candidates[0], candidates


def corrected_rows(rows, fitted, parameters, use_anchor, use_projection):
    output = []
    for row in rows:
        scenario, z, q, _ = row
        if use_projection:
            anchor = (ANCHOR.predict_anchor(
                row, fitted, parameters[0]) if use_anchor else None)
            z, q = ANCHOR.input_project(
                scenario, z, q, anchor, *parameters[1:])
        output.append((scenario, z, q))
    return output


def regional_metrics(rows):
    keys = ("supervised_x46", "upstream", "downstream",
            "unsupervised_internal")
    groups = {key: [[], [], [], []] for key in keys}
    all_values = [[], [], [], []]
    for scenario, pred_z, pred_q in rows:
        true_z, true_q = scenario["z"], scenario["q"]
        all_values[0].append(pred_z.ravel())
        all_values[1].append(pred_q.ravel())
        all_values[2].append(true_z.ravel())
        all_values[3].append(true_q.ravel())
        masks = {
            "supervised_x46": np.eye(len(scenario["x"]), dtype=bool)[46][None, :].repeat(len(scenario["t"]), 0),
            "upstream": np.eye(len(scenario["x"]), dtype=bool)[0][None, :].repeat(len(scenario["t"]), 0),
            "downstream": np.eye(len(scenario["x"]), dtype=bool)[-1][None, :].repeat(len(scenario["t"]), 0),
            "unsupervised_internal": np.ones_like(true_z, dtype=bool),
        }
        masks["unsupervised_internal"][:, [0, 46, -1]] = False
        for key, mask in masks.items():
            groups[key][0].append(pred_z[mask])
            groups[key][1].append(pred_q[mask])
            groups[key][2].append(true_z[mask])
            groups[key][3].append(true_q[mask])
    result = {"global": metric_arrays(*map(np.concatenate, all_values))}
    for key, values in groups.items():
        result[key] = metric_arrays(*map(np.concatenate, values))
    return result


def pde_diagnostics(model, dataset, standardizer, full_conditions,
                    geometry, pde_scales, paths, seed):
    rng = np.random.default_rng(seed)
    values = []
    for path in paths:
        scenario = dataset.load_scenario(path)
        x = BASE.to_tensor(rng.uniform(
            scenario["x"][0], scenario["x"][-1], 128))
        t = BASE.to_tensor(rng.uniform(
            scenario["t"][0], scenario["t"][-1], 128))
        condition = BASE.to_tensor(
            standardizer.transform(dataset.get_condition_vector(path)))
        model_condition(
            model,
            BASE.to_tensor(full_conditions.get(dataset, path))
            if full_conditions is not None else None)
        mass, momentum = ENGINE.pde_pointwise(
            model, geometry, pde_scales, x, t, condition)
        values.append(torch.stack(
            (mass.abs(), momentum.abs()), 1).detach().cpu().numpy())
    values = np.concatenate(values)
    result = {}
    for index, name in enumerate(("mass", "momentum")):
        value = values[:, index]
        result[name] = {
            "mse": float(np.mean(value * value)),
            "mean": float(value.mean()), "median": float(np.median(value)),
            "p90": float(np.quantile(value, .9)),
            "p95": float(np.quantile(value, .95)),
            "max": float(value.max())}
    return result


def build_base(scales, geometry):
    return ENGINE.DenseOperator(
        scales, geometry, BASE.condition_input_dim,
        code_dim=128, hidden=256).to(DEVICE)


def warmup_options():
    return ENGINE.Options(
        strategy="pcgrad_warmup", sampling="rar", epochs=30,
        warmup_epochs=30, batch_scenarios=4, time_points=512,
        physics_points=512, lr=7e-4, q_weight=3, bc_q_weight=5,
        pde_target=.5, data_q_multiplier=1, grad_ratio=1.0,
        physics_lr_scale=.1)


def al_options():
    return ENGINE.Options(
        strategy="al_pinn", sampling="rar", epochs=8,
        batch_scenarios=4, time_points=256, physics_points=256,
        lr=5e-5, q_weight=3, bc_q_weight=1, data_q_multiplier=5,
        al_mu=10, al_pde_multiplier=.1)


def enrichment_options():
    return ENGINE.Options(
        strategy="al_pinn", sampling="uniform", epochs=20,
        batch_scenarios=2, time_points=256, physics_points=256,
        lr=3e-4, q_weight=3, bc_q_weight=1, data_q_multiplier=3,
        al_mu=10, al_pde_multiplier=.1)


def train_stage(model, stage_name, dataset, standardizer, full_conditions,
                geometry, train_paths, validation_paths, pde_scales,
                epochs, options, objective_kind, output_dir, seed,
                trainable=None, initial_state=None, rng_seed_base=None,
                weight_decay=1.0e-4):
    if initial_state is not None:
        model.load_state_dict(initial_state)
    if trainable is not None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in trainable:
            parameter.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=options.lr, weight_decay=weight_decay)
    al_state = [
        torch.zeros(3, device=DEVICE),
        torch.full((3,), float(options.al_mu), device=DEVICE)]
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    history, best_score, best_state = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        record = train_epoch(
            model, dataset, standardizer, full_conditions, geometry,
            pde_scales, train_paths, options, epoch, seed, optimizer,
            al_state, objective_kind, rng_seed_base=rng_seed_base)
        validation = validation_metrics(
            model, dataset, standardizer, full_conditions,
            validation_paths, seed + 1, points=1024)
        score = max(validation["z_l2"], validation["q_l2"])
        full_record = {
            "stage": stage_name, "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **record, "validation": validation}
        history.append(full_record)
        save_state(stage_dir / "last.pt", model, optimizer,
                   al_state, epoch, best_score, full_record)
        if score < best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            save_state(stage_dir / "best.pt", model, optimizer,
                       al_state, epoch, best_score, full_record)
        print(json.dumps(full_record, ensure_ascii=False), flush=True)
    model.load_state_dict(best_state)
    (stage_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    return best_state, history


def run_experiment(args):
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA unavailable; formal results require GPU")
    start_time = time.time()
    seed_everything(args.seed)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    dataset = BASE.ScenarioDataset(BASE.cache_directory, cache_limit=600)
    train_paths, validation_paths = split_paths(dataset, args.seed)
    standardizer, scales = fit_train_statistics(dataset, train_paths)
    full_conditions = FullTraceConditions(dataset, train_paths, ntime=128)
    reference = dataset.load_scenario(train_paths[0])
    geometry = BASE.CrossSectionGeometry(
        BASE.cross_section_profile_path,
        BASE.to_tensor(reference["x"]), device=DEVICE)
    pde_scales = ENGINE.physical_pde_scales(
        dataset, geometry, scales)
    output_dir = OUT_ROOT / "clean_strict_paper" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        "pipeline": "strict_paper_clean", "seed": args.seed,
        "device": str(DEVICE), "train_scenarios": len(train_paths),
        "validation_scenarios": len(validation_paths),
        "test_scenarios": len(dataset.test_files),
        "internal_supervision": "x[46] only",
        "stages": ["30 supervised warmup",
                   "8 AL-PINN base", "20 joint AL-PINN enrichment"],
        "loaded_historical_checkpoint": False,
        "test_used_for_selection": False,
    }, indent=2), encoding="utf-8")

    base = build_base(scales, geometry)
    state1, history1 = train_stage(
        base, "stage1_supervised", dataset, standardizer, None,
        geometry, train_paths, validation_paths, pde_scales,
        args.stage1_epochs, warmup_options(), "supervised",
        output_dir, args.seed, weight_decay=1.0e-2)
    state2, history2 = train_stage(
        base, "stage2_al_pinn", dataset, standardizer, None,
        geometry, train_paths, validation_paths, pde_scales,
        args.stage2_epochs, al_options(), "al", output_dir, args.seed,
        initial_state=state1, weight_decay=1.0e-2)

    enriched = EnrichedOperator(base).to(DEVICE)
    enriched.load_state_dict(
        {"base." + key: value for key, value in state2.items()},
        strict=False)
    state3, history3 = train_stage(
        enriched, "stage3_joint_enrichment", dataset, standardizer,
        full_conditions, geometry, train_paths, validation_paths,
        pde_scales, args.stage3_epochs, enrichment_options(), "al",
        output_dir, args.seed, trainable=list(enriched.parameters()),
        rng_seed_base=args.seed * 100, weight_decay=1.0e-4)
    enriched.load_state_dict(state3)

    train_rows = collect_rows(
        enriched, dataset, standardizer, full_conditions, train_paths)
    validation_rows = collect_rows(
        enriched, dataset, standardizer, full_conditions, validation_paths)
    selected_anchor, anchor_candidates = select_anchor(
        train_rows, validation_rows)
    fitted_anchor = ANCHOR.fit_anchor(
        train_rows, selected_anchor["length"], selected_anchor["ridge"])
    anchor_parameters = (
        selected_anchor["length"], .02, .08, .12,
        selected_anchor["left_scale"], selected_anchor["right_scale"])
    selected_no_anchor, no_anchor_candidates = select_no_anchor(
        validation_rows)
    no_anchor_parameters = (
        1.0, selected_no_anchor["et"], selected_no_anchor["eu"],
        selected_no_anchor["ed"], .08, .15)
    test_rows = collect_rows(
        enriched, dataset, standardizer, full_conditions, dataset.test_files)
    anchored_rows = corrected_rows(
        test_rows, fitted_anchor, anchor_parameters, True,
        not args.no_projection)
    no_anchor_rows = corrected_rows(
        test_rows, None, no_anchor_parameters, False,
        not args.no_projection)
    anchor_score = ANCHOR.score(
        test_rows, fitted_anchor, anchor_parameters, True, stride=1)
    no_anchor_score = ANCHOR.score(
        test_rows, None, no_anchor_parameters, False, stride=1)
    pde = pde_diagnostics(
        enriched, dataset, standardizer, full_conditions,
        geometry, pde_scales, dataset.test_files, args.seed + 7)
    result = {
        "pipeline": "strict_paper_clean", "seed": args.seed,
        "device": str(DEVICE), "loaded_historical_checkpoint": False,
        "strict_single_section_supervision": True,
        "anchor_training_labels": "training x[46] only",
        "selected_anchor": selected_anchor,
        "selected_no_anchor": selected_no_anchor,
        "test": {"z_l2": float((no_anchor_score if args.no_anchor else anchor_score)[0]),
                 "q_l2": float((no_anchor_score if args.no_anchor else anchor_score)[1])},
        "anchored_test": {"z_l2": float(anchor_score[0]),
                          "q_l2": float(anchor_score[1])},
        "no_anchor_test": {"z_l2": float(no_anchor_score[0]),
                           "q_l2": float(no_anchor_score[1])},
        "regional_metrics": {
            "anchor": regional_metrics(anchored_rows),
            "no_anchor": regional_metrics(no_anchor_rows)},
        "pde_diagnostics": pde,
        "best_epoch": {
            "stage1": int(min(history1, key=lambda item: max(
                item["validation"]["z_l2"], item["validation"]["q_l2"]))["epoch"]),
            "stage2": int(min(history2, key=lambda item: max(
                item["validation"]["z_l2"], item["validation"]["q_l2"]))["epoch"]),
            "stage3": int(min(history3, key=lambda item: max(
                item["validation"]["z_l2"], item["validation"]["q_l2"]))["epoch"])},
        "elapsed_seconds": time.time() - start_time,
        "parameter_count": int(sum(
            parameter.numel() for parameter in enriched.parameters())),
        "peak_memory_mb": float(
            torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)),
        "history": {"stage1": history1, "stage2": history2, "stage3": history3},
    }
    for name, payload in (
        ("anchor_candidates.json", anchor_candidates),
        ("no_anchor_candidates.json", no_anchor_candidates),
        ("regional_metrics.json", result["regional_metrics"]),
        ("pde_diagnostics.json", pde),
        ("result.json", result)):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "test": result["test"], "no_anchor_test": result["no_anchor_test"],
        "best_epoch": result["best_epoch"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="023 单断面 PINN：固定科研流程")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--stage3-epochs", type=int, default=20)
    parser.add_argument("--no-anchor", action="store_true",
                        help="将无锚点结果作为主测试结果")
    parser.add_argument("--no-projection", action="store_true",
                        help="关闭 IC/BC/GPR 推理投影")
    run_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
