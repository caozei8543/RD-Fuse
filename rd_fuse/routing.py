"""Budget-feasible utility-cost routing (manuscript Eqs. 8–10)."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional
import torch
from torch import nn


COUNTS = (0, 2, 4, 8, 12, 16)


@dataclass(frozen=True)
class Choice:
    mode: str
    k: int
    indices: tuple[int, ...]
    cost_ms: float
    value: float


class LatencyProfile:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if 'path_ms' not in d or 'windows_ms' not in d:
            raise ValueError('Profile requires path_ms and windows_ms')
        self.raw = d
        self.is_verified = bool(d.get('validated_for_target_hardware', False))

    def cost(self, mode: str, k: int = 0) -> float:
        if mode not in ('L', 'R', 'T', 'C0', 'C'):
            raise ValueError(f'Invalid mode {mode}')
        base = float(self.raw.get('shared_ms', 0.0)) + float(self.raw['path_ms']['C0' if mode == 'C' else mode])
        if mode == 'C':
            base += float(self.raw['windows_ms'][str(k)])
        return base

    def budget(self, ratio: float) -> float:
        return ratio * self.cost('C', 16)


class GlobalUtility(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(131, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class LocalUtility(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(260, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, global_state: torch.Tensor, local_state: torch.Tensor) -> torch.Tensor:
        n = local_state.shape[1]
        return self.net(torch.cat([global_state[:, None].expand(-1, n, -1), local_state], dim=-1)).squeeze(-1)


def predicted_value(global_gains: torch.Tensor, local_gains: torch.Tensor,
                    mode: str, k: int, cost_ms: float, lam: float = .005) -> torch.Tensor:
    """Differentiable Eq. (9); chosen window indices are discrete, gains stay differentiable."""
    if mode == 'L':
        gain = global_gains.new_zeros(())
    elif mode in ('R', 'T', 'C0'):
        gain = global_gains[{'R': 0, 'T': 1, 'C0': 2}[mode]]
    elif mode == 'C':
        gain = global_gains[2] + local_gains[torch.argsort(local_gains, descending=True, stable=True)[:k]].sum()
    else:
        raise ValueError(mode)
    return gain - lam * cost_ms


def enumerate_choices(gg: torch.Tensor, lg: torch.Tensor, profile: LatencyProfile,
                      budget_ms: float, lam: float = .005, counts=COUNTS) -> list[Choice]:
    if gg.ndim != 1 or gg.shape[0] != 3 or lg.ndim != 1:
        raise ValueError('Router currently requires batch=1 and global(3)/local(64)')
    choices = []
    for mode, k in [('L', 0), ('R', 0), ('T', 0), ('C0', 0)] + [('C', x) for x in counts if x]:
        if k > lg.numel():
            continue
        ms = profile.cost(mode, k)
        if ms > budget_ms + 1e-9:
            continue
        idx = tuple(torch.argsort(lg, descending=True, stable=True)[:k].tolist()) if k else ()
        score = predicted_value(gg, lg, mode, k, ms, lam).item()
        choices.append(Choice(mode, k, idx, ms, score))
    if not choices:
        raise ValueError(f'Budget {budget_ms:.4f} ms admits no feasible path')
    return sorted(choices, key=lambda c: (-c.value, c.cost_ms, c.k, c.mode))


def select_choice(gg: torch.Tensor, lg: torch.Tensor, profile: LatencyProfile,
                  budget_ms: float, lam: float = .005, counts=COUNTS) -> Choice:
    return enumerate_choices(gg.detach(), lg.detach(), profile, budget_ms, lam, counts)[0]
