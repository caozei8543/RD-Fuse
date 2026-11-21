"""Shared-space shallow state, global/local disagreement and ranked reliability."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ProbeState:
    global_state: torch.Tensor  # B x 131
    local_state: torch.Tensor   # B x 64 x 129
    rgb_semantic: torch.Tensor  # B x 64 x S x S
    tir_semantic: torch.Tensor
    rgb_reliability: torch.Tensor
    tir_reliability: torch.Tensor
    rgb_windows: torch.Tensor  # B x 64 x 64
    tir_windows: torch.Tensor


def cosine_disagreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)) * 0.5


class SemanticProbe(nn.Module):
    def __init__(self, semantic_size: int = 40, grid: int = 8):
        super().__init__()
        if semantic_size % grid:
            raise ValueError('semantic_size must be divisible by grid')
        self.semantic_size, self.grid = semantic_size, grid
        self.rgb_projection = nn.Conv2d(64, 64, 1)
        self.tir_projection = nn.Conv2d(64, 64, 1)
        self.reliability = nn.Sequential(nn.Linear(65, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def _entropy(self, sem: torch.Tensor) -> torch.Tensor:
        # Scalar spatial activation entropy divided by log(number of windows).
        pooled = F.adaptive_avg_pool2d(sem.abs().mean(dim=1, keepdim=True), (self.grid, self.grid))
        p = F.softmax(pooled.flatten(1), dim=-1).clamp_min(1e-8)
        return -(p * p.log()).sum(dim=-1, keepdim=True) / math.log(self.grid ** 2)

    def _single(self, s2: torch.Tensor, projection: nn.Module):
        sem = F.adaptive_avg_pool2d(projection(s2), (self.semantic_size, self.semantic_size))
        gl = sem.mean(dim=(2, 3))
        windows = F.adaptive_avg_pool2d(sem, (self.grid, self.grid)).flatten(2).transpose(1, 2)
        reliability = self.reliability(torch.cat([gl, self._entropy(sem)], dim=-1))
        return sem, gl, windows, reliability

    def forward(self, rgb_s2: torch.Tensor, tir_s2: torch.Tensor) -> ProbeState:
        r, gr, wr, rr = self._single(rgb_s2, self.rgb_projection)
        t, gt, wt, rt = self._single(tir_s2, self.tir_projection)
        dg = cosine_disagreement(gr, gt).unsqueeze(-1)
        di = cosine_disagreement(wr, wt).unsqueeze(-1)
        gs = torch.cat([gr, gt, rr, rt, dg], dim=-1)
        ls = torch.cat([wr, wt, di], dim=-1)
        assert gs.shape[-1] == 131 and ls.shape[-1] == 129
        return ProbeState(gs, ls, r, t, rr, rt, wr, wt)


def pairwise_reliability_loss(clean: torch.Tensor, degraded: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Same-modality original should outrank synthetically degraded view."""
    return (margin - (clean - degraded)).clamp_min(0).mean()


def symmetric_infonce(a: torch.Tensor, b: torch.Tensor, temperature: float = .07) -> torch.Tensor:
    """Global scene or flattened local-window contrastive objective.

    A single image has no global in-batch negatives, so that term is zero.
    Local windows remain usable as negatives for B=1.
    """
    if a.shape[0] < 2:
        return a.new_zeros(())
    aa, bb = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = aa @ bb.T / temperature
    y = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) * 0.5


def semantic_alignment_loss(p: ProbeState, temperature: float = .07, hard_neg: int = 8):
    # Global InfoNCE over batch. Local InfoNCE across corresponding windows
    # within each sample; hard negatives are top similar NON-matching regions.
    gr, gt = p.rgb_semantic.mean((2, 3)), p.tir_semantic.mean((2, 3))
    gl = symmetric_infonce(gr, gt, temperature)
    wr, wt = F.normalize(p.rgb_windows, dim=-1), F.normalize(p.tir_windows, dim=-1)
    sim = (wr @ wt.transpose(1, 2)) / temperature  # B x 64 x 64
    n = sim.shape[-1]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)[None]
    neg = sim.masked_fill(eye, -1e6)
    k = min(hard_neg, n - 1)
    # Matched positive vs k most similar non-matching windows.
    pos = sim.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    loss_rt = F.cross_entropy(torch.cat([pos, neg.topk(k, dim=-1).values], dim=-1).flatten(0, 1),
                              torch.zeros(sim.shape[0] * n, dtype=torch.long, device=sim.device))
    trans = sim.transpose(1, 2)
    neg2 = trans.masked_fill(eye, -1e6)
    pos2 = trans.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    loss_tr = F.cross_entropy(torch.cat([pos2, neg2.topk(k, dim=-1).values], dim=-1).flatten(0, 1),
                              torch.zeros(sim.shape[0] * n, dtype=torch.long, device=sim.device))
    return gl, 0.5 * (loss_rt + loss_tr)
