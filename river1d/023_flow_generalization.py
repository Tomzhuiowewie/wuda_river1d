"""From-scratch 023 staged/joint single-section PINN experiment.

This file is self contained with respect to model initialization and training:
it never loads the historical 023/114 checkpoints.  The staged protocol
faithfully reproduces the historical 30-supervised + 10-PCGrad/RAR +
20-AL(all-401) + 20-enrichment chain; consequently its AL stage deliberately
reuses the 40 screening scenarios, exactly as the historical experiment did.
``--pipeline simplified`` runs the paper-oriented two-stage version: the
same 40-epoch first stage followed by 20 epochs of AL-PINN, with the separate
enrichment stage removed. ``--pipeline all`` also runs the strict 361/40
Joint control.  The only
internal spatial training label is x[46]; GPR is a post-training anchor
surrogate fitted from the 361-scenario training split.

``--pipeline strict_paper`` is the strict paper-oriented protocol validated
from scratch: 30 supervised PCGrad/RAR epochs, 8 validation-selected
AL-PINN epochs, and 20 epochs of joint base-plus-known-condition enrichment,
all using only the 361 training scenarios for optimization.
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
# Keep the earlier strict 361/40 experiment intact.  The default 023 output
# now records the faithful historical 10.48%-chain reproduction separately.
OUT_ROOT = ROOT / "outputs/flow_generalization_023/reproduction"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def initialize_xavier(module: nn.Module, gain: float = 1.0) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


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

    def __init__(self, base, extra_dim=440, zero_last=True, extra_gain=None):
        super().__init__()
        self.base = base
        self.scales, self.geometry = base.scales, base.geometry
        self.extra = nn.Sequential(
            nn.LayerNorm(extra_dim),
            nn.Linear(extra_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128),
        )
        if extra_gain is not None:
            initialize_xavier(self.extra, extra_gain)
        if zero_last:
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


def train_epoch(model, dataset, standardizer, full_conditions, geometry, pde_scales,
                paths, options, epoch, seed, optimizer, al_state, objective_kind,
                rng_seed_base=None, balance_state=None):
    model.train()
    # Keep one deterministic RNG stream per epoch, matching the historical
    # 023 engine.  RAR candidates and supervised time samples therefore see
    # exactly the same random-number ordering as the original run.
    rng = np.random.default_rng((seed * 100000 if rng_seed_base is None else rng_seed_base) + epoch)
    order = rng.permutation(len(paths))
    batch_size = options.batch_scenarios
    records, ratios = [], []
    for start in range(0, len(order), batch_size):
        optimizer.zero_grad(set_to_none=True)
        terms_batch = []
        for index in order[start:start + batch_size]:
            path = paths[index]
            s = dataset.load_scenario(path)
            c = BASE.to_tensor(standardizer.transform(dataset.get_condition_vector(path)))
            e = BASE.to_tensor(full_conditions.get(dataset, path)) if full_conditions else None
            model_condition(model, e)
            terms_batch.append(ENGINE.loss_terms(model, geometry, pde_scales, s, c, options,
                                                 rng, epoch))
        terms = torch.stack(terms_batch).mean(0)
        weighted = weighted_terms(terms, options.q_weight, options.bc_q_weight,
                                  options.data_q_multiplier)
        grouped = ENGINE.groups(weighted)
        pcgrad_active = objective_kind in ("pcgrad", "pde_anneal") or (
            objective_kind == "pcgrad_warmup" and epoch > options.warmup_epochs
        )
        if pcgrad_active:
            weights = torch.ones(4, device=DEVICE)
            sup = grouped[:3].sum().detach()
            pde = grouped[3].detach().clamp_min(1e-12)
            if objective_kind == "pde_anneal":
                progress = min(max(float(epoch), 0.0), 40.0) / 40.0
                target = 0.05 + (options.pde_target - 0.05) * 0.5 * (
                    1.0 - np.cos(np.pi * progress)
                )
            else:
                target = options.pde_target
            weights[3] = (target / max(1.0 - target, 1e-12) * sup / pde).clamp(1e-6, 1e6)
            ratio, _ = ENGINE.project_pde_against_supervision(grouped, model, weights, 1.0)
            ratios.append(ratio)
            objective_value = (weights * grouped).sum().detach()
            adaptive_weights = torch.cat((torch.ones(6, device=DEVICE),
                                          weights[3].repeat(2)))
        elif objective_kind == "relobralo":
            if balance_state is None:
                raise RuntimeError("ReLoBRaLo requires persistent balance_state")
            current = weighted.detach().clamp_min(1e-12)
            if balance_state.get("initial") is None:
                balance_state["initial"] = current.clone()
                balance_state["previous"] = current.clone()
                balance_state["weights"] = torch.ones_like(current)
            temperature = float(balance_state["temperature"])

            def balanced(reference):
                logits = current / (temperature * reference.clamp_min(1e-12))
                logits = logits - logits.max()
                return current.numel() * torch.softmax(logits, dim=0)

            rho = float(rng.random() < balance_state["rho_probability"])
            bal_initial = balanced(balance_state["initial"])
            bal_previous = balanced(balance_state["previous"])
            history = rho * balance_state["weights"] + (1.0 - rho) * bal_initial
            adaptive_weights = (balance_state["alpha"] * history
                                + (1.0 - balance_state["alpha"]) * bal_previous)
            adaptive_weights = adaptive_weights.detach()
            balance_state["previous"] = current.clone()
            balance_state["weights"] = adaptive_weights.clone()
            objective = (adaptive_weights * weighted).sum()
            objective.backward()
            objective_value = objective.detach()
        elif objective_kind == "pcgrad_warmup":
            # Historical protocol: learn the supervised/IC/BC mapping first;
            # physics and gradient projection start only after epoch 30.
            objective = grouped[:3].sum()
            objective.backward()
            objective_value = objective.detach()
            adaptive_weights = torch.tensor([1., 1., options.bc_q_weight,
                                             options.bc_q_weight, 1.,
                                             options.q_weight * options.data_q_multiplier,
                                             0., 0.], device=DEVICE)
        else:
            constraints = grouped[:3]
            lam, mu = al_state
            objective = (options.al_pde_multiplier * grouped[3]
                         + torch.dot(lam.detach(), constraints)
                         + 0.5 * (mu * constraints.square()).sum())
            objective.backward()
            with torch.no_grad():
                lam.add_(mu * constraints.detach()).clamp_(-options.al_mu_max, options.al_mu_max)
            objective_value = objective.detach()
            adaptive_weights = torch.ones(8, device=DEVICE)
        if pcgrad_active:
            # project_pde_against_supervision has already populated p.grad.
            pass
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        records.append({
            "objective": float(objective_value.cpu()),
            "raw_terms": terms.detach().cpu().tolist(),
            "weighted_terms": weighted.detach().cpu().tolist(),
            "grouped_terms": grouped.detach().cpu().tolist(),
            "pde_share": (float((grouped[3] / grouped.sum().clamp_min(1e-12)).detach().cpu())
                          if pcgrad_active or objective_kind in ("al", "relobralo") else 0.0),
            "adaptive_weights": adaptive_weights.detach().cpu().tolist(),
        })
    mean = np.asarray([r["raw_terms"] for r in records]).mean(0)
    return {
        "objective": float(np.mean([r["objective"] for r in records])),
        "raw_terms": mean.tolist(),
        "pde_share": float(np.mean([r["pde_share"] for r in records])),
        "projection_ratio": float(np.mean(ratios)) if ratios else None,
        "adaptive_weights": np.asarray([r["adaptive_weights"] for r in records]).mean(0).tolist(),
    }


def save_state(path, model, optimizer, al_state, epoch, best_score, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "al_lambda": al_state[0].detach().cpu(), "epoch": epoch,
        "best_score": best_score, "config": config,
        "numpy_state": np.random.get_state(), "torch_state": torch.get_rng_state(),
    }, path)


def load_model_state(model, path, optimizer=None, al_state=None):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if al_state is not None and "al_lambda" in state:
        al_state[0].copy_(state["al_lambda"].to(DEVICE))
    return state


def collect_rows(model, dataset, standardizer, full_conditions, paths):
    model.eval()
    rows = []
    with torch.no_grad():
        for path in paths:
            s = dataset.load_scenario(path)
            tg, xg = np.meshgrid(s["t"], s["x"], indexing="ij")
            c = BASE.to_tensor(standardizer.transform(dataset.get_condition_vector(path)))
            e = BASE.to_tensor(full_conditions.get(dataset, path)) if full_conditions else None
            model_condition(model, e)
            fx, ft = BASE.to_tensor(xg.ravel()), BASE.to_tensor(tg.ravel())
            pz, pq = [], []
            for i in range(0, fx.numel(), 8192):
                z, q = model(fx[i:i + 8192], ft[i:i + 8192], c)
                pz.append(z.cpu().numpy()); pq.append(q.cpu().numpy())
            rows.append((s, np.concatenate(pz).reshape(tg.shape),
                         np.concatenate(pq).reshape(tg.shape),
                         standardizer.transform(dataset.get_condition_vector(path))))
    return rows


def select_anchor(train_rows, validation_rows):
    candidates = []
    for length in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        for ridge in (1e-3, 1e-2, 0.1, 1.0):
            fitted = ANCHOR.fit_anchor(train_rows, length, ridge)
            for left in (0.04, 0.08, 0.12, 0.2):
                for right in (0.08, 0.15, 0.2, 0.3):
                    pars = (length, 0.02, 0.08, 0.12, left, right)
                    z, q = ANCHOR.score(validation_rows, fitted, pars, True, stride=20)
                    candidates.append({"length": length, "ridge": ridge,
                                       "left_scale": left, "right_scale": right,
                                       "val_z_l2": z, "val_q_l2": q})
    candidates.sort(key=lambda r: max(r["val_z_l2"], r["val_q_l2"]))
    return candidates[0], candidates


def select_no_anchor(validation_rows):
    candidates = []
    for et in (0.01, 0.02, 0.05, 0.1):
        for eu in (0.04, 0.08, 0.12, 0.2):
            for ed in (0.04, 0.08, 0.12, 0.2):
                pars = (1.0, et, eu, ed, 0.08, 0.15)
                z, q = ANCHOR.score(validation_rows, None, pars, False, stride=20)
                candidates.append({"et": et, "eu": eu, "ed": ed,
                                   "val_z_l2": z, "val_q_l2": q})
    candidates.sort(key=lambda r: max(r["val_z_l2"], r["val_q_l2"]))
    return candidates[0], candidates


def corrected_rows(rows, fitted, anchor_parameters, use_anchor, use_projection=True):
    output = []
    for row in rows:
        s, z, q, _ = row
        if not use_projection:
            zh, qh = z, q
        else:
            anchor = ANCHOR.predict_anchor(row, fitted, anchor_parameters[0]) if use_anchor else None
            zh, qh = ANCHOR.input_project(s, z, q, anchor, *anchor_parameters[1:])
        output.append((s, zh, qh))
    return output


def regional_metrics(rows):
    result = {"global": None, "supervised_x46": None, "upstream": None,
              "downstream": None, "unsupervised_internal": None}
    all_pz, all_pq, all_tz, all_tq = [], [], [], []
    groups = {k: [[], [], [], []] for k in result if k != "global"}
    for s, pz, pq in rows:
        tz, tq = s["z"], s["q"]
        all_pz.append(pz.ravel()); all_pq.append(pq.ravel()); all_tz.append(tz.ravel()); all_tq.append(tq.ravel())
        masks = {
            "supervised_x46": np.eye(len(s["x"]), dtype=bool)[46][None, :].repeat(len(s["t"]), 0),
            "upstream": np.eye(len(s["x"]), dtype=bool)[0][None, :].repeat(len(s["t"]), 0),
            "downstream": np.eye(len(s["x"]), dtype=bool)[-1][None, :].repeat(len(s["t"]), 0),
            "unsupervised_internal": np.ones_like(tz, dtype=bool),
        }
        masks["unsupervised_internal"][:, [0, 46, -1]] = False
        for key, mask in masks.items():
            groups[key][0].append(pz[mask]); groups[key][1].append(pq[mask])
            groups[key][2].append(tz[mask]); groups[key][3].append(tq[mask])
    result["global"] = metric_arrays(*map(np.concatenate, (all_pz, all_pq, all_tz, all_tq)))
    for key, values in groups.items():
        result[key] = metric_arrays(*map(np.concatenate, values))
    return result


def pde_diagnostics(model, dataset, standardizer, full_conditions, geometry, pde_scales, paths, seed):
    rng = np.random.default_rng(seed)
    values = []
    for path in paths:
        s = dataset.load_scenario(path); n = 128
        x = BASE.to_tensor(rng.uniform(s["x"][0], s["x"][-1], n))
        t = BASE.to_tensor(rng.uniform(s["t"][0], s["t"][-1], n))
        c = BASE.to_tensor(standardizer.transform(dataset.get_condition_vector(path)))
        e = BASE.to_tensor(full_conditions.get(dataset, path)) if full_conditions else None
        model_condition(model, e)
        mass, mom = ENGINE.pde_pointwise(model, geometry, pde_scales, x, t, c)
        values.append(torch.stack((mass.abs(), mom.abs()), 1).detach().cpu().numpy())
    array = np.concatenate(values)
    output = {}
    for i, name in enumerate(("mass", "momentum")):
        v = array[:, i]
        output[name] = {"mse": float(np.mean(v * v)), "mean": float(v.mean()),
                        "median": float(np.median(v)), "p90": float(np.quantile(v, .9)),
                        "p95": float(np.quantile(v, .95)), "max": float(v.max())}
    return output


def build_base(scales, geometry):
    model = ENGINE.DenseOperator(scales, geometry, BASE.condition_input_dim, code_dim=128, hidden=256).to(DEVICE)
    # Match the established OperatorPINN initialization (PyTorch Linear
    # defaults).  This is still a fresh random initialization and avoids
    # conflating the staged/joint comparison with an initialization ablation.
    return model


def warmup_pcgrad_options(epochs=40):
    """Canonical 30-supervised + 10-PCGrad/RAR configuration."""
    return ENGINE.Options(
        strategy="pcgrad_warmup", sampling="rar", epochs=epochs,
        warmup_epochs=30, batch_scenarios=4, time_points=512,
        physics_points=512, lr=7e-4, q_weight=3, bc_q_weight=5,
        pde_target=.5, data_q_multiplier=1, grad_ratio=1.0,
        physics_lr_scale=.1,
    )


def adaptive_base_options(strategy, epochs=40):
    """Joint eight-term base PINN used by the weighting ablation."""
    return ENGINE.Options(
        strategy=strategy, sampling="rar", epochs=epochs,
        warmup_epochs=0, batch_scenarios=4, time_points=512,
        physics_points=512, lr=7e-4, q_weight=3, bc_q_weight=5,
        pde_target=.5, data_q_multiplier=1, grad_ratio=1.0,
        physics_lr_scale=.1,
    )


def train_stage(model, stage_name, dataset, standardizer, full_conditions, geometry,
                train_paths, val_paths, pde_scales, epochs, options, objective_kind,
                output_dir, seed, trainable=None, initial_state=None,
                resume_state=None, select_last=False, rng_seed_base=None,
                weight_decay=1.0e-4):
    if initial_state is not None:
        model.load_state_dict(initial_state)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"] if "model" in resume_state else resume_state)
    if trainable is not None:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in trainable:
            p.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    effective_lr = options.lr * (options.physics_lr_scale if objective_kind == "pcgrad" else 1.0)
    optimizer = torch.optim.AdamW(parameters, lr=effective_lr, weight_decay=weight_decay)
    al_state = [torch.zeros(3, device=DEVICE), torch.full((3,), float(options.al_mu), device=DEVICE)]
    balance_state = ({"initial": None, "previous": None, "weights": None,
                      "alpha": 0.999, "temperature": 1.0,
                      "rho_probability": 0.999}
                     if objective_kind == "relobralo" else None)
    if resume_state is not None:
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "al_lambda" in resume_state:
            al_state[0].copy_(resume_state["al_lambda"].to(DEVICE))
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = int(resume_state.get("epoch", 0)) + 1 if resume_state is not None else 1
    best_score = float(resume_state.get("best_score", float("inf"))) if resume_state is not None else float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if resume_state is not None else None
    history = []
    for epoch in range(start_epoch, start_epoch + epochs):
        if objective_kind == "pcgrad_warmup":
            optimizer.param_groups[0]["lr"] = options.lr * (
                options.physics_lr_scale if epoch > options.warmup_epochs else 1.0
            )
        train_record = train_epoch(model, dataset, standardizer, full_conditions, geometry,
                                   pde_scales, train_paths, options, epoch, seed,
                                   optimizer, al_state, objective_kind, rng_seed_base,
                                   balance_state)
        validation = (validation_metrics(model, dataset, standardizer, full_conditions,
                                         val_paths, seed + 1, points=1024)
                      if val_paths else None)
        score = (max(validation["z_l2"], validation["q_l2"])
                 if validation is not None else float("inf"))
        record = {"stage": stage_name, "epoch": epoch,
                  "learning_rate": float(optimizer.param_groups[0]["lr"]),
                  **train_record, "validation": validation}
        history.append(record)
        save_state(stage_dir / "last.pt", model, optimizer, al_state, epoch, best_score, record)
        if select_last or score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_state(stage_dir / "best.pt", model, optimizer, al_state, epoch, best_score, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if best_state is None:
        raise RuntimeError(f"{stage_name} failed to produce a checkpoint")
    model.load_state_dict(best_state)
    (stage_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_state, history


def run_pipeline(name, args):
    start_time = time.time()
    seed_everything(args.seed)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    dataset = BASE.ScenarioDataset(BASE.cache_directory, cache_limit=600)
    train_paths, val_paths = split_paths(dataset, args.seed)
    # The historical 10.48% chain standardized on all 401 development
    # scenarios and reused the 40 screening scenarios during its AL stage.
    # Keep Joint strict (361-only statistics) while making Staged faithful.
    if name in ("staged", "simplified", "relobralo", "pde_anneal"):
        standardizer, scales = BASE.fit_condition_normalizer_and_scales(dataset)
    else:
        standardizer, scales = fit_train_statistics(dataset, train_paths)
    full_conditions = FullTraceConditions(dataset, train_paths, 128)
    reference = dataset.load_scenario(train_paths[0])
    geometry = BASE.CrossSectionGeometry(BASE.cross_section_profile_path,
                                        BASE.to_tensor(reference["x"]), device=DEVICE)
    pde_scales = ENGINE.physical_pde_scales(dataset, geometry, scales)
    output_dir = OUT_ROOT / name / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"pipeline": name, "seed": args.seed, "device": str(DEVICE),
              "train": len(train_paths), "validation": len(val_paths), "test": len(dataset.test_files),
              "strict_single_section": True, "loaded_historical_checkpoint": False,
              "joint_base_and_enrichment_from_stage3_epoch1": bool(name == "strict_paper"),
              "protocol": ("strict_paper_30sup_8al_20joint_known_condition_enrichment"
                           if name == "strict_paper" else
                           "historical_reproduction_30sup_10pcgrad_20al401_20enrichment"
                           if name == "staged" else
                           "paper_compact_30sup_10pcgrad_20al401_gpr" if name == "simplified"
                           else "relobralo40_20al401_gpr" if name == "relobralo"
                           else "pde_cosine_05_to_50_40_20al401_gpr" if name == "pde_anneal"
                           else "strict_joint_361_40"),
              "normalization_scenarios": (401 if name in ("staged", "simplified", "relobralo", "pde_anneal") else 361),
              "al_training_scenarios": (401 if name in ("staged", "simplified", "relobralo", "pde_anneal") else 361),
              "validation_reused_by_staged_al": bool(name in ("staged", "simplified", "relobralo", "pde_anneal")),
              "relobralo": ({"alpha": .999, "temperature": 1.0,
                              "rho_probability": .999, "terms": 8}
                             if name == "relobralo" else None),
              "pde_annealing": ({"minimum": .05, "maximum": .5,
                                  "schedule": "half_cosine", "epochs": 40}
                                 if name == "pde_anneal" else None),
              "pde_scales": list(map(float, pde_scales))}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    base = build_base(scales, geometry)
    eval_checkpoint = args.checkpoint or args.resume
    if args.eval_only:
        if not eval_checkpoint:
            raise SystemExit("--eval-only必须提供--checkpoint路径")
        final_model = (base if name in ("simplified", "relobralo", "pde_anneal") else
                       EnrichedOperator(base, zero_last=False, extra_gain=.1).to(DEVICE))
        state = torch.load(eval_checkpoint, map_location=DEVICE, weights_only=False)
        final_model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        stage_history = {}
        best_epoch = {"loaded_checkpoint": str(eval_checkpoint)}
    elif args.resume:
        # Resume the final trainable stage while restoring the available
        # checkpoint state. The compact pipeline may transition directly
        # from its completed stage-1 checkpoint into stage 2.
        state = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        checkpoint_text = str(args.resume)
        if "stage1_pcgrad_rar" in checkpoint_text and name == "simplified":
            base.load_state_dict(state["model"])
            opt2 = ENGINE.Options(strategy="al_pinn", sampling="rar",
                                  batch_scenarios=4, time_points=256,
                                  physics_points=256, lr=5e-5, q_weight=3,
                                  bc_q_weight=1, data_q_multiplier=5,
                                  al_mu=10, al_pde_multiplier=.1)
            state2, h2 = train_stage(
                base, "stage2_al_pinn_all401", dataset, standardizer,
                None, geometry, list(dataset.train_files), [], pde_scales,
                args.stage2_epochs, opt2, "al", output_dir, args.seed,
                initial_state=state["model"], select_last=True,
                weight_decay=1.0e-2)
            base.load_state_dict(state2)
            final_model = base
            stage1_history_path = output_dir / "stage1_pcgrad_rar" / "history.json"
            h1 = (json.loads(stage1_history_path.read_text(encoding="utf-8"))
                  if stage1_history_path.exists() else [])
            stage_history = {"stage1": h1, "stage2": h2}
            best_epoch = {
                "stage1": (int(min(h1, key=lambda r: max(
                    r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"])
                           if h1 else int(state.get("epoch", 0))),
                "stage2": int(h2[-1]["epoch"]),
            }
        elif "stage2_al_pinn_all401" in checkpoint_text and name == "staged":
            completed = int(state.get("epoch", 0))
            remaining = max(int(args.stage2_epochs) - completed, 0)
            opt2 = ENGINE.Options(strategy="al_pinn", sampling="rar",
                                  batch_scenarios=4, time_points=256,
                                  physics_points=256, lr=5e-5, q_weight=3,
                                  bc_q_weight=1, data_q_multiplier=5,
                                  al_mu=10, al_pde_multiplier=.1)
            state2, h2 = train_stage(
                base, "stage2_al_pinn_all401_resume", dataset, standardizer,
                None, geometry, list(dataset.train_files), [], pde_scales,
                remaining, opt2, "al", output_dir, args.seed,
                resume_state=state, select_last=True, weight_decay=1.0e-2)
            enriched = EnrichedOperator(base, zero_last=True, extra_gain=None).to(DEVICE)
            enriched.load_state_dict({"base." + k: v for k, v in state2.items()}, strict=False)
            state3, h3 = train_stage(
                enriched, "stage3_full_trace", dataset, standardizer,
                full_conditions, geometry, train_paths, val_paths, pde_scales,
                args.stage3_epochs,
                ENGINE.Options(strategy="al_pinn", sampling="uniform",
                               batch_scenarios=2, time_points=256,
                               physics_points=256, lr=3e-4, q_weight=3,
                               bc_q_weight=1, data_q_multiplier=3,
                               al_mu=10, al_pde_multiplier=.1),
                "al", output_dir, args.seed,
                trainable=list(enriched.extra.parameters()),
                rng_seed_base=args.seed * 100)
            enriched.load_state_dict(state3)
            final_model = enriched
            stage1_history_path = output_dir / "stage1_pcgrad_rar" / "history.json"
            h1 = (json.loads(stage1_history_path.read_text(encoding="utf-8"))
                  if stage1_history_path.exists() else [])
            stage_history = {"stage1": h1, "stage2_resumed": h2, "stage3": h3}
            best_epoch = {
                "stage1": (int(min(h1, key=lambda r: max(r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"])
                           if h1 else None),
                "stage2": completed + remaining,
                "stage3": int(min(h3, key=lambda r: max(r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"]),
            }
        elif "stage3_full_trace" in checkpoint_text and name == "staged":
            final_model = EnrichedOperator(base, zero_last=True, extra_gain=None).to(DEVICE)
            final_model.load_state_dict(state["model"])
            state_r, h_r = train_stage(
                final_model, "stage3_full_trace_resume", dataset, standardizer,
                full_conditions, geometry, train_paths, val_paths, pde_scales,
                args.stage3_epochs, ENGINE.Options(strategy="al_pinn", sampling="rar",
                batch_scenarios=2, time_points=256, physics_points=256, lr=3e-4,
                q_weight=3, bc_q_weight=1, data_q_multiplier=3, al_mu=10,
                al_pde_multiplier=.1), "al", output_dir, args.seed,
                trainable=list(final_model.extra.parameters()), resume_state=state)
            final_model.load_state_dict(state_r)
            stage_history = {"resumed_stage3": h_r}
            best_epoch = {"resumed_stage3": int(min(h_r, key=lambda r: max(r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"])}
        elif "joint" in checkpoint_text and name == "joint":
            final_model = EnrichedOperator(base, zero_last=False, extra_gain=.1).to(DEVICE)
            final_model.load_state_dict(state["model"])
            opt_resume = ENGINE.Options(strategy="al_pinn", sampling="rar", epochs=args.joint_epochs,
                                        batch_scenarios=4, time_points=256, physics_points=256,
                                        lr=3e-4, q_weight=3, bc_q_weight=1, data_q_multiplier=5,
                                        al_mu=10, al_pde_multiplier=.1)
            state_r, h_r = train_stage(final_model, "joint_resume", dataset, standardizer,
                                       full_conditions, geometry, train_paths, val_paths,
                                       pde_scales, args.joint_epochs, opt_resume, "al",
                                       output_dir, args.seed, trainable=list(final_model.parameters()),
                                       resume_state=state)
            final_model.load_state_dict(state_r)
            stage_history = {"resumed_joint": h_r}
            best_epoch = {"resumed_joint": int(min(h_r, key=lambda r: max(r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"])}
        else:
            raise SystemExit("检查点与pipeline不匹配；simplified支持stage1，staged支持stage2/stage3，joint支持joint检查点")
    elif name in ("relobralo", "pde_anneal"):
        objective_kind = name
        opt1 = adaptive_base_options(name, args.stage1_epochs)
        state1, h1 = train_stage(
            base, f"stage1_{name}", dataset, standardizer, None, geometry,
            train_paths, val_paths, pde_scales, args.stage1_epochs, opt1,
            objective_kind, output_dir, args.seed, weight_decay=1.0e-2)
        base.load_state_dict(state1)
        opt2 = ENGINE.Options(strategy="al_pinn", sampling="rar",
                              batch_scenarios=4, time_points=256,
                              physics_points=256, lr=5e-5, q_weight=3,
                              bc_q_weight=1, data_q_multiplier=5,
                              al_mu=10, al_pde_multiplier=.1)
        state2, h2 = train_stage(
            base, "stage2_al_pinn_all401", dataset, standardizer, None,
            geometry, list(dataset.train_files), [], pde_scales,
            args.stage2_epochs, opt2, "al", output_dir, args.seed,
            initial_state=state1, select_last=True, weight_decay=1.0e-2)
        base.load_state_dict(state2)
        final_model = base
        stage_history = {"stage1": h1, "stage2": h2}
        best_epoch = {
            "stage1": int(min(h1, key=lambda r: max(
                r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"]),
            "stage2": int(h2[-1]["epoch"]),
        }
    elif name == "simplified":
        opt1 = warmup_pcgrad_options(args.stage1_epochs)
        state1, h1 = train_stage(
            base, "stage1_pcgrad_rar", dataset, standardizer, None, geometry,
            train_paths, val_paths, pde_scales, args.stage1_epochs, opt1,
            "pcgrad_warmup", output_dir, args.seed, weight_decay=1.0e-2)
        base.load_state_dict(state1)
        opt2 = ENGINE.Options(strategy="al_pinn", sampling="rar",
                              batch_scenarios=4, time_points=256,
                              physics_points=256, lr=5e-5, q_weight=3,
                              bc_q_weight=1, data_q_multiplier=5,
                              al_mu=10, al_pde_multiplier=.1)
        state2, h2 = train_stage(
            base, "stage2_al_pinn_all401", dataset, standardizer, None,
            geometry, list(dataset.train_files), [], pde_scales,
            args.stage2_epochs, opt2, "al", output_dir, args.seed,
            initial_state=state1, select_last=True, weight_decay=1.0e-2)
        base.load_state_dict(state2)
        final_model = base
        stage_history = {"stage1": h1, "stage2": h2}
        best_epoch = {
            "stage1": int(min(
                h1, key=lambda r: max(r["validation"]["z_l2"],
                                      r["validation"]["q_l2"])
            )["epoch"]),
            "stage2": int(h2[-1]["epoch"]),
        }
    elif name == "strict_paper":
        # Paper-oriented strict protocol: all neural/PDE training uses the
        # 361 training scenarios; the 40 validation scenarios select the AL
        # and enrichment checkpoints and are never used for optimization.
        opt1 = warmup_pcgrad_options(args.stage1_epochs)
        state1, h1 = train_stage(
            base, "stage1_pcgrad_rar", dataset, standardizer, None, geometry,
            train_paths, val_paths, pde_scales, args.stage1_epochs,
            opt1, "pcgrad_warmup", output_dir, args.seed,
            weight_decay=1.0e-2)
        opt2 = ENGINE.Options(
            strategy="al_pinn", sampling="rar", epochs=args.stage2_epochs,
            batch_scenarios=4, time_points=256, physics_points=256,
            lr=5e-5, q_weight=3, bc_q_weight=1, data_q_multiplier=5,
            al_mu=10, al_pde_multiplier=.1)
        state2, h2 = train_stage(
            base, "stage2_al_pinn_strict", dataset, standardizer, None,
            geometry, train_paths, val_paths, pde_scales, args.stage2_epochs,
            opt2, "al", output_dir, args.seed, initial_state=state1,
            select_last=False, weight_decay=1.0e-2)
        enriched = EnrichedOperator(base, zero_last=True, extra_gain=None).to(DEVICE)
        enriched.load_state_dict({"base." + k: v for k, v in state2.items()},
                                 strict=False)
        opt3 = ENGINE.Options(
            strategy="al_pinn", sampling="uniform", batch_scenarios=2,
            time_points=256, physics_points=256, lr=3e-4, q_weight=3,
            bc_q_weight=1, data_q_multiplier=3, al_mu=10,
            al_pde_multiplier=.1)
        state3, h3 = train_stage(
            enriched, "stage3_known_condition_enrichment", dataset,
            standardizer, full_conditions, geometry, train_paths, val_paths,
            pde_scales, args.stage3_epochs, opt3, "al", output_dir, args.seed,
            # Joint enrichment ablation: the base PINN and the new condition
            # branch are both optimized from the first fine-tuning epoch.
            trainable=list(enriched.parameters()), rng_seed_base=args.seed * 100)
        enriched.load_state_dict(state3)
        final_model = enriched
        stage_history = {"stage1": h1, "stage2": h2, "stage3": h3}
        best_epoch = {
            "stage1": int(min(h1, key=lambda r: max(
                r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"]),
            "stage2": int(min(h2, key=lambda r: max(
                r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"]),
            "stage3": int(min(h3, key=lambda r: max(
                r["validation"]["z_l2"], r["validation"]["q_l2"]))["epoch"]),
        }
    elif name == "staged":
        opt1 = warmup_pcgrad_options(args.stage1_epochs)
        state1, h1 = train_stage(base, "stage1_pcgrad_rar", dataset, standardizer, None, geometry,
                                  train_paths, val_paths, pde_scales, args.stage1_epochs,
                                  opt1, "pcgrad_warmup", output_dir, args.seed,
                                  weight_decay=1.0e-2)
        opt2 = ENGINE.Options(strategy="al_pinn", sampling="rar", epochs=args.stage2_epochs,
                              batch_scenarios=4, time_points=256, physics_points=256,
                              lr=5e-5, q_weight=3, bc_q_weight=1, data_q_multiplier=5,
                              al_mu=10, al_pde_multiplier=.1)
        # Faithful historical reproduction: this stage deliberately trains on
        # all 401 development scenarios and keeps the final epoch, as did
        # al_all401/q5.  The external 100-scenario test set remains untouched.
        state2, h2 = train_stage(base, "stage2_al_pinn_all401", dataset, standardizer,
                                  None, geometry, list(dataset.train_files), [], pde_scales,
                                  args.stage2_epochs, opt2, "al", output_dir, args.seed,
                                  initial_state=state1, select_last=True,
                                  weight_decay=1.0e-2)
        enriched = EnrichedOperator(base, zero_last=True, extra_gain=None).to(DEVICE)
        enriched.load_state_dict({"base." + k: v for k, v in state2.items()}, strict=False)
        # The base is frozen; only the zero-initialized full-trace correction is trained.
        state3, h3 = train_stage(enriched, "stage3_full_trace", dataset, standardizer, full_conditions,
                                  geometry, train_paths, val_paths, pde_scales, args.stage3_epochs,
                                  ENGINE.Options(strategy="al_pinn", sampling="uniform", batch_scenarios=2,
                                                time_points=256, physics_points=256, lr=3e-4,
                                                q_weight=3, bc_q_weight=1, data_q_multiplier=3,
                                                al_mu=10, al_pde_multiplier=.1), "al", output_dir,
                                  args.seed, trainable=list(enriched.extra.parameters()), initial_state=None,
                                  rng_seed_base=args.seed * 100)
        final_model = enriched
        stage_history = {"stage1": h1, "stage2": h2, "stage3": h3}
        best_epoch = {"stage1": int(np.argmin([max(r["validation"]["z_l2"], r["validation"]["q_l2"]) for r in h1]) + 1),
                      "stage2": int(h2[-1]["epoch"]),
                      "stage3": int(np.argmin([max(r["validation"]["z_l2"], r["validation"]["q_l2"]) for r in h3]) + 1)}
    else:
        final_model = EnrichedOperator(base, zero_last=False, extra_gain=.1).to(DEVICE)
        opt = ENGINE.Options(strategy="al_pinn", sampling="rar", batch_scenarios=4,
                             time_points=256, physics_points=256, lr=3e-4, q_weight=3,
                             bc_q_weight=1, data_q_multiplier=5, al_mu=10,
                             al_pde_multiplier=.1)
        state, hj = train_stage(final_model, "joint", dataset, standardizer, full_conditions,
                                geometry, train_paths, val_paths, pde_scales, args.joint_epochs,
                                opt, "al", output_dir, args.seed,
                                trainable=list(final_model.parameters()), initial_state=None)
        final_model.load_state_dict(state)
        stage_history = {"joint": hj}
        best_epoch = {"joint": int(np.argmin([max(r["validation"]["z_l2"], r["validation"]["q_l2"]) for r in hj]) + 1)}

    train_rows = collect_rows(final_model, dataset, standardizer, full_conditions, train_paths)
    val_rows = collect_rows(final_model, dataset, standardizer, full_conditions, val_paths)
    selected_anchor, anchor_candidates = select_anchor(train_rows, val_rows)
    fitted = ANCHOR.fit_anchor(train_rows, selected_anchor["length"], selected_anchor["ridge"])
    anchor_pars = (selected_anchor["length"], .02, .08, .12,
                   selected_anchor["left_scale"], selected_anchor["right_scale"])
    selected_no_anchor, no_anchor_candidates = select_no_anchor(val_rows)
    no_anchor_pars = (1.0, selected_no_anchor["et"], selected_no_anchor["eu"], selected_no_anchor["ed"], .08, .15)
    test_rows = collect_rows(final_model, dataset, standardizer, full_conditions, dataset.test_files)
    # Always build the anchored control explicitly; --no-anchor only changes
    # which control is reported as the primary ``test`` result.
    anchored = corrected_rows(test_rows, fitted, anchor_pars, True, args.no_projection)
    no_anchor = corrected_rows(test_rows, None, no_anchor_pars, False, args.no_projection)
    anchor_probe = ANCHOR.predict_anchor(test_rows[0], fitted, anchor_pars[0])
    if args.no_anchor:
        anchored = no_anchor
    metrics_anchor = regional_metrics(anchored)
    metrics_no_anchor = regional_metrics(no_anchor)
    # Independent score path guards against any array-aliasing in the
    # regional report and is the canonical global L2 comparison.
    score_anchor = ANCHOR.score(test_rows, fitted, anchor_pars, True, stride=1)
    score_no_anchor = ANCHOR.score(test_rows, None, no_anchor_pars, False, stride=1)
    metrics_anchor["global"]["z_l2"], metrics_anchor["global"]["q_l2"] = map(float, score_anchor)
    metrics_no_anchor["global"]["z_l2"], metrics_no_anchor["global"]["q_l2"] = map(float, score_no_anchor)
    # The canonical anchor effect is the change in the independently
    # evaluated global Q score.  Computing it from the regional arrays can
    # hide the correction when those arrays share views or are projected.
    anchor_delta = float(abs(score_anchor[1] - score_no_anchor[1]))
    anchor_delta_mean = anchor_delta
    pde = pde_diagnostics(final_model, dataset, standardizer, full_conditions,
                          geometry, pde_scales, dataset.test_files, args.seed + 7)
    final = {
        "pipeline": name, "seed": args.seed, "device": str(DEVICE),
        "loaded_historical_checkpoint": False, "strict_single_section_supervision": True,
        "joint_base_and_enrichment_from_stage3_epoch1": bool(name == "strict_paper"),
        "anchor_training_labels": "361 training scenarios x[46] only",
        "pinn_training_labels": ("stage1: 361 scenarios x[46]; stage2 AL: 361 scenarios x[46]; "
                                 "stage3: 361 scenarios x[46]" if name == "strict_paper"
                                 else "stage1: 361 scenarios x[46]; stage2 AL: all 401 scenarios x[46]; "
                                 "stage3: 361 scenarios x[46]" if name == "staged"
                                 else "stage1: 361 scenarios x[46]; stage2 AL: all 401 scenarios x[46]"
                                 if name in ("simplified", "relobralo", "pde_anneal")
                                 else "361 training scenarios x[46] only"),
        "historical_validation_reuse": bool(name in ("staged", "simplified", "relobralo", "pde_anneal")),
        "selected_anchor": selected_anchor, "selected_no_anchor": selected_no_anchor,
        "no_anchor_requested": bool(args.no_anchor),
        "projection_requested": not bool(args.no_projection),
        "anchor_effect": {"max_abs_q_delta": anchor_delta, "mean_abs_q_delta": anchor_delta_mean},
        "anchor_probe": {"min": float(anchor_probe.min()), "max": float(anchor_probe.max())},
        "test": metrics_anchor["global"], "no_anchor_test": metrics_no_anchor["global"],
        "regional_metrics": {"anchor": metrics_anchor, "no_anchor": metrics_no_anchor},
        "pde_diagnostics": pde,
        "best_epoch": best_epoch, "elapsed_seconds": time.time() - start_time,
        "parameter_count": int(sum(p.numel() for p in final_model.parameters())),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)) if DEVICE.type == "cuda" else None,
        "history": stage_history,
    }
    (output_dir / "anchor_candidates.json").write_text(json.dumps(anchor_candidates, indent=2), encoding="utf-8")
    (output_dir / "no_anchor_candidates.json").write_text(json.dumps(no_anchor_candidates, indent=2), encoding="utf-8")
    (output_dir / "regional_metrics.json").write_text(json.dumps(final["regional_metrics"], indent=2), encoding="utf-8")
    (output_dir / "pde_diagnostics.json").write_text(json.dumps(pde, indent=2), encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def write_comparison(results):
    comparison = {k: {"test": v["test"], "no_anchor_test": v["no_anchor_test"],
                      "best_epoch": v["best_epoch"], "elapsed_seconds": v["elapsed_seconds"],
                      "regional": v["regional_metrics"]} for k, v in results.items()}
    (OUT_ROOT / "staged_joint_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    staged, joint = results.get("staged"), results.get("joint")
    lines = ["# 023 Staged / Joint 从头训练对照", "", "| 指标 | Staged | Joint | Joint-Staged |", "|---|---:|---:|---:|"]
    for label, key in (("测试 Z L2", "z_l2"), ("测试 Q L2", "q_l2"), ("无锚点 Z L2", "z_l2"), ("无锚点 Q L2", "q_l2")):
        a = (staged["test"] if "无锚点" not in label else staged["no_anchor_test"])[key]
        b = (joint["test"] if "无锚点" not in label else joint["no_anchor_test"])[key]
        lines.append(f"| {label} | {a:.4f}% | {b:.4f}% | {b-a:+.4f} pp |")
    lines.extend(["", "测试集只在各自验证检查点和GPR参数冻结后评价；两种流程均未加载历史权重。"])
    (OUT_ROOT / "STAGED_JOINT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("staged", "strict_paper", "joint", "simplified", "relobralo", "pde_anneal", "all"), default="simplified")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stage1-epochs", type=int, default=40)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage3-epochs", type=int, default=20)
    parser.add_argument("--joint-epochs", type=int, default=80)
    parser.add_argument("--no-anchor", action="store_true",
                        help="最终主结果关闭GPR锚点；仍输出同一模型的无锚点配对结果")
    parser.add_argument("--no-projection", action="store_true",
                        help="关闭IC/BC推理投影，仅评价神经网络原始输出")
    parser.add_argument("--resume", default="",
                        help="继续最终阶段；simplified也可从stage1最佳检查点进入AL阶段")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评价已有最终检查点；需同时指定--checkpoint")
    parser.add_argument("--checkpoint", default="",
                        help="--eval-only使用的最终base/stage3/joint检查点")
    args = parser.parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA不可用，拒绝将CPU结果混入正式实验")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    names = ("staged", "joint") if args.pipeline == "all" else (args.pipeline,)
    results = {}
    for name in names:
        results[name] = run_pipeline(name, args)
    if len(results) == 2:
        write_comparison(results)
    print(json.dumps({k: {"test": v["test"], "no_anchor_test": v["no_anchor_test"]} for k, v in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
