"""Three training stages: forced-route learning, utility estimation and regret."""
from __future__ import annotations
import random
from dataclasses import dataclass
import torch
from torch.nn import functional as F

from .data import synthetic_degradation
from .detection import detection_loss
from .probe import semantic_alignment_loss, pairwise_reliability_loss
from .routing import enumerate_choices, predicted_value, LatencyProfile
from .model import RDFuse


@torch.no_grad()
def true_forced_loss(model: RDFuse, rgb, tir, targets, mode: str, indices=None) -> torch.Tensor:
    """IMPORTANT: returns a *real joint execution* detection loss, NOT sum of u_i."""
    out = model(rgb, tir, forced=mode, windows=indices)
    return detection_loss(out.predictions, targets).detach()


def set_trainable(model: RDFuse, phase: str):
    # Phase 2/3: fixed detector and probe, train predictors only, as described.
    model.requires_grad_(phase == 'forced')
    if phase != 'forced':
        model.global_utility.requires_grad_(True)
        model.local_utility.requires_grad_(True)
        model.eval()
        model.global_utility.train()
        model.local_utility.train()
    else:
        model.train()


def forced_step(model: RDFuse, rgb, tir, targets, *, alpha=.1, beta=.05,
                align_global=.5, align_local=.5, margin=.1, info_temp=.07,
                hard_neg=8, degradation_probability=.5):
    mode = random.choice(['L', 'R', 'T', 'C0', 'C'])
    k = random.choice([2, 4, 8, 12, 16]) if mode == 'C' else 0
    out = model(rgb, tir, forced=mode, forced_k=k)
    det = detection_loss(out.predictions, targets)
    gl, loc = semantic_alignment_loss(out.probe, info_temp, hard_neg)
    alignment = align_global * gl + align_local * loc
    # Genuine same-modality pairwise ranking; don't pretend scores calibrated.
    deg_rgb, deg_tir, mods = synthetic_degradation(rgb, tir, degradation_probability, with_label=True)
    mods = [mods] if isinstance(mods, (str, type(None))) else mods
    rel_terms = []
    if 'R' in mods:
        sdeg = model.rgb.shallow(deg_rgb)
        dg = model.probe._single(sdeg, model.probe.rgb_projection)[3]
        mask = torch.tensor([v == 'R' for v in mods], device=rgb.device, dtype=torch.bool)
        rel_terms.append(pairwise_reliability_loss(out.probe.rgb_reliability[mask], dg[mask], margin))
    if 'T' in mods:
        sdeg = model.tir.shallow(deg_tir)
        dg = model.probe._single(sdeg, model.probe.tir_projection)[3]
        mask = torch.tensor([v == 'T' for v in mods], device=rgb.device, dtype=torch.bool)
        rel_terms.append(pairwise_reliability_loss(out.probe.tir_reliability[mask], dg[mask], margin))
    rel = torch.stack(rel_terms).mean() if rel_terms else det * 0
    return det + alpha * alignment + beta * rel, {'mode': mode,
                                                    'det': float(det.detach()),
                                                    'align': float(alignment.detach()),
                                                    'rel': float(rel.detach())}


def utility_step(model: RDFuse, rgb, tir, targets, sampled_windows: int = 4, cache: dict | None = None):
    # Predict on one shallow pass, fixed representation: model is in eval mode,
    # and only predictors require_grad.
    r2, t2 = model.rgb.shallow(rgb), model.tir.shallow(tir)
    probe = model.probe(r2, t2)
    gg = model.global_utility(probe.global_state)
    lg = model.local_utility(probe.global_state, probe.local_state)
    key = targets[0].get('sample_index')
    if cache is not None and key is not None and key in cache:
        rgains, indices, per_window = [x.to(lg.device) for x in cache[key]]
    else:
        with torch.no_grad():
            baseline = true_forced_loss(model, rgb, tir, targets, 'L')
            rgains = torch.stack([baseline - true_forced_loss(model, rgb, tir, targets, mode)
                                 for mode in ['R', 'T', 'C0']])
            c0 = baseline - rgains[2]
            indices = torch.randperm(lg.shape[1], device=lg.device)[:sampled_windows]
            per_window = torch.stack([c0 - true_forced_loss(model, rgb, tir, targets, 'C', [int(i)])
                                      for i in indices])
        if cache is not None and key is not None:
            cache[key] = tuple(x.detach().cpu() for x in (rgains, indices, per_window))
    loss = F.smooth_l1_loss(gg[0], rgains) + F.smooth_l1_loss(lg[0, indices], per_window)
    return loss, {'global_gain': [float(v) for v in rgains],
                  'sampled_windows': [int(i) for i in indices],
                  'loss': float(loss.detach())}


def regret_step(model: RDFuse, rgb, tir, targets, profile: LatencyProfile,
                budget_ms: float, lam: float = .005, temperature: float = .2,
                max_candidates: int = 6):
    """Eq. 11: true combined execution loss for sampled feasible candidates.

    Predictions are soft over a fixed sampled feasible set; choice membership
    and top-k window IDs are discrete/no-grad, predicted gains are differentiable.
    """
    r2, t2 = model.rgb.shallow(rgb), model.tir.shallow(tir)
    p = model.probe(r2, t2)
    gg = model.global_utility(p.global_state)[0]
    lg = model.local_utility(p.global_state, p.local_state)[0]
    choices = enumerate_choices(gg, lg, profile, budget_ms, lam)
    # Include inexpensive paths and multiple C options if feasible. All are
    # evaluated using joint execution; never replace true loss with sum of u_i.
    if len(choices) > max_candidates:
        cheap = [c for c in choices if c.mode == 'L'][:1]
        combos = [c for c in choices if c.mode == 'C'][:2]
        others = [c for c in choices if c not in cheap and c not in combos]
        choices = (cheap + combos + others)[:max_candidates]
    predicted = torch.stack([predicted_value(gg, lg, c.mode, c.k, c.cost_ms, lam) for c in choices])
    with torch.no_grad():
        realized = torch.stack([true_forced_loss(model, rgb, tir, targets, c.mode, list(c.indices)) +
                                lam*c.cost_ms for c in choices])
        regret_targets = (realized - realized.min()).detach()
    pz = F.softmax(predicted / temperature, dim=0)
    loss = (pz * regret_targets).sum()
    return loss, {'candidates': [f'{c.mode}:{c.k}' for c in choices],
                  'realized_costs': [float(x) for x in realized],
                  'regret': float(loss.detach())}
