"""RD-Fuse conditional routes, semantic-state probe and fusion model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

from .backbones import make_backbone, conv_block
from .probe import SemanticProbe, ProbeState
from .routing import Choice, GlobalUtility, LocalUtility, LatencyProfile, select_choice


class SelectiveCrossAttention(nn.Module):
    """Gather only selected nonoverlapping windows, interact, scatter to dense map.

    Attention is per window; selected windows are processed together as a batch
    of independent windows. Exact paper attention details were not specified.
    """
    def __init__(self, dim: int = 128, out_dim: int = 64, heads: int = 4, grid: int = 8):
        super().__init__()
        self.rgb_proj = nn.Conv2d(64, dim, 1)
        self.tir_proj = nn.Conv2d(64, dim, 1)
        self.r_to_t = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.t_to_r = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Conv2d(dim, out_dim, 1)
        self.grid = grid
        self.last_executed = 0

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, indices: list[int] | tuple[int, ...]):
        B, _, H, W = rgb.shape
        if H % self.grid or W % self.grid:
            raise ValueError('Semantic map must be divisible by window grid')
        self.last_executed = len(indices)
        if not indices:
            return rgb.new_zeros((B, self.out.out_channels, H, W))
        if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= self.grid**2:
            raise ValueError('Window IDs must be unique and in [0,grid^2)')
        kh, kw = H // self.grid, W // self.grid
        r = self.rgb_proj(rgb)
        t = self.tir_proj(tir)
        pr = F.unfold(r, kernel_size=(kh, kw), stride=(kh, kw))
        pt = F.unfold(t, kernel_size=(kh, kw), stride=(kh, kw))
        ids = torch.tensor(indices, device=rgb.device)
        # gather selected patch tensors only; unselected attention is NEVER run
        def prep(x):
            x = x.index_select(2, ids).transpose(1, 2)
            return x.reshape(B * len(indices), r.shape[1], kh*kw).transpose(1, 2)
        qr, qt = prep(pr), prep(pt)
        ar = self.r_to_t(qr, qt, qt, need_weights=False)[0]
        at = self.t_to_r(qt, qr, qr, need_weights=False)[0]
        pieces = ((ar + at) * .5).transpose(1, 2).reshape(B, len(indices), -1).transpose(1, 2)
        dense = r.new_zeros(pr.shape)
        dense.index_copy_(2, ids, pieces)
        updated = F.fold(dense, output_size=(H, W), kernel_size=(kh, kw), stride=(kh, kw))
        output = self.out(updated)
        # The output projection has a bias: explicitly mask non-selected windows.
        # Otherwise, those locations would receive a constant interaction offset.
        selected = torch.zeros((1, 1, self.grid * self.grid), device=rgb.device, dtype=output.dtype)
        selected[:, :, ids] = 1
        mask = F.fold(selected.expand(1, kh * kw, -1), output_size=(H, W),
                      kernel_size=(kh, kw), stride=(kh, kw))
        return output * mask


class AnchorFreeHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.towers = nn.ModuleList([conv_block(64, 64) for _ in range(3)])
        self.classes = nn.ModuleList([nn.Conv2d(64, num_classes, 1) for _ in range(3)])
        self.dist = nn.ModuleList([nn.Conv2d(64, 4, 1) for _ in range(3)])
        self.center = nn.ModuleList([nn.Conv2d(64, 1, 1) for _ in range(3)])
        for layer in self.classes:
            nn.init.constant_(layer.bias, -4.6)  # low initial object prior

    def forward(self, features: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        out = []
        for x, tower, c, b, t in zip(features, self.towers, self.classes, self.dist, self.center):
            x = tower(x)
            out.append({'cls': c(x), 'ltrb': F.softplus(b(x)), 'center': t(x)})
        return out


@dataclass
class ForwardResult:
    predictions: list[dict[str, torch.Tensor]]
    mode: str
    window_indices: tuple[int, ...]
    probe: ProbeState
    global_utility: torch.Tensor
    local_utility: torch.Tensor
    cost_ms: Optional[float]


class RDFuse(nn.Module):
    def __init__(self, num_classes: int = 1, backbone: str = 'tiny', pretrained: bool = False,
                 semantic_size: int = 40, grid: int = 8):
        super().__init__()
        self.rgb = make_backbone(backbone, 3, pretrained=pretrained)
        self.tir = make_backbone(backbone, 1, pretrained=pretrained)
        self.probe = SemanticProbe(semantic_size=semantic_size, grid=grid)
        self.global_utility = GlobalUtility()
        self.local_utility = LocalUtility()
        self.cross = SelectiveCrossAttention(dim=128, out_dim=64, heads=4, grid=grid)
        self.comp3_r = nn.Conv2d(64, 160, 1)
        self.comp3_t = nn.Conv2d(64, 160, 1)
        self.comp4_r = nn.Conv2d(64, 256, 1)
        self.comp4_t = nn.Conv2d(64, 256, 1)
        self.merge2 = nn.Conv2d(128, 64, 1)
        self.merge3 = nn.Conv2d(320, 64, 1)
        self.merge4 = nn.Conv2d(512, 64, 1)
        self.head = AnchorFreeHead(num_classes)
        self.num_classes = num_classes
        self.backbone_name = backbone

    def _merged(self, rgb2: torch.Tensor, tir2: torch.Tensor, state: ProbeState,
                mode: str, indices: tuple[int, ...]) -> list[torch.Tensor]:
        # Stage 2 is ALWAYS executed. Deep RGB/T branches truly skip execution.
        s2 = self.merge2(torch.cat([rgb2, tir2], 1))
        need_r = mode in ('R', 'C0', 'C')
        need_t = mode in ('T', 'C0', 'C')
        r3 = r4 = t3 = t4 = None
        if need_r:
            r3, r4 = self.rgb.deep(rgb2)
        if need_t:
            t3, t4 = self.tir.deep(tir2)
        s3_shape = (rgb2.shape[-2] // 2, rgb2.shape[-1] // 2)
        s4_shape = (rgb2.shape[-2] // 4, rgb2.shape[-1] // 4)
        if r3 is None:
            r3 = self.comp3_r(F.adaptive_avg_pool2d(rgb2, s3_shape))
            r4 = self.comp4_r(F.adaptive_avg_pool2d(rgb2, s4_shape))
        if t3 is None:
            t3 = self.comp3_t(F.adaptive_avg_pool2d(tir2, s3_shape))
            t4 = self.comp4_t(F.adaptive_avg_pool2d(tir2, s4_shape))
        s3 = self.merge3(torch.cat([r3, t3], 1))
        s4 = self.merge4(torch.cat([r4, t4], 1))
        if mode == 'C' and indices:
            local = self.cross(state.rgb_semantic, state.tir_semantic, indices)
            s3 = s3 + F.interpolate(local, size=s3.shape[-2:], mode='bilinear', align_corners=False)
        else:
            self.cross.last_executed = 0
        return [s2, s3, s4]

    def forward(self, rgb: torch.Tensor, tir: torch.Tensor, *,
                forced: Optional[str] = None, windows: Optional[list[int] | tuple[int, ...]] = None,
                profile: Optional[LatencyProfile] = None, budget_ms: Optional[float] = None,
                lam: float = .005, forced_k: int = 2) -> ForwardResult:
        if rgb.ndim != 4 or tir.ndim != 4 or rgb.shape[1] != 3 or tir.shape[1] != 1:
            raise ValueError('Expected RGB [B,3,H,W] and thermal [B,1,H,W]')
        if rgb.size(0) != tir.size(0) or rgb.shape[-2:] != tir.shape[-2:]:
            raise ValueError('Paired inputs must have the same batch and spatial dimensions')
        r2 = self.rgb.shallow(rgb)
        t2 = self.tir.shallow(tir)
        state = self.probe(r2, t2)
        gg = self.global_utility(state.global_state)
        lg = self.local_utility(state.global_state, state.local_state)
        cost = None
        if forced is None:
            if rgb.size(0) != 1 or profile is None or budget_ms is None:
                raise ValueError('Dynamic inference needs batch=1, profile, and budget_ms')
            ch = select_choice(gg[0], lg[0], profile, budget_ms, lam)
            mode, indices, cost = ch.mode, ch.indices, ch.cost_ms
        else:
            mode = forced
            if mode not in ('L', 'R', 'T', 'C0', 'C'):
                raise ValueError('Invalid forced mode')
            if mode != 'C' and windows:
                raise ValueError('Interaction windows only allowed in C mode')
            indices = tuple(int(i) for i in (windows or []))
            if mode == 'C' and not indices:
                indices = tuple(int(v) for v in lg[0].topk(forced_k).indices.tolist())
            if profile is not None:
                cost = profile.cost(mode, len(indices))
        feats = self._merged(r2, t2, state, mode, indices)
        return ForwardResult(self.head(feats), mode, indices, state, gg, lg, cost)
