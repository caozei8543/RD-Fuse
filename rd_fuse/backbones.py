"""Backbones for a transparent RD-Fuse *reference* implementation.

PVTv2-B0 uses timm's stage modules directly so skipped deep stages are not
executed. The small CNN exists only to run offline CPU smoke tests; it is NOT
an alternative claiming the paper's accuracy or computational results.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.GELU(),
    )


class TinyStages(nn.Module):
    """Synthetic demonstration backbone, NOT the manuscript's PVTv2-B0."""
    def __init__(self, in_chans: int = 3):
        super().__init__()
        self.s1 = conv_block(in_chans, 32, 4)
        self.s2 = conv_block(32, 64, 2)
        self.s3 = conv_block(64, 160, 2)
        self.s4 = conv_block(160, 256, 2)
        self.deep_calls = 0

    def shallow(self, x: torch.Tensor) -> torch.Tensor:
        return self.s2(self.s1(x))

    def deep(self, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.deep_calls += 1
        s3 = self.s3(s2)
        return s3, self.s4(s3)


class PVTv2Stages(nn.Module):
    """Adapter for timm pvt_v2_b0; two shallow stages, then two deep stages.

    Uses timm's currently exposed .patch_embed and .stages. Version should
    be pinned and validated at release time by the authors.
    """
    def __init__(self, in_chans: int, pretrained: bool):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("PVTv2-B0 requires `pip install timm`; use backbone=tiny for offline smoke tests") from exc
        self.net = timm.create_model('pvt_v2_b0', pretrained=pretrained)
        assert hasattr(self.net, 'patch_embed') and len(self.net.stages) == 4, 'Unexpected timm PVTv2 API'
        if in_chans == 1:
            old = self.net.patch_embed.proj
            new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                if new.bias is not None:
                    new.bias.copy_(old.bias)
            self.net.patch_embed.proj = new
        self.deep_calls = 0

    def shallow(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.patch_embed(x)
        x = self.net.stages[0](x)
        return self.net.stages[1](x)

    def deep(self, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.deep_calls += 1
        s3 = self.net.stages[2](s2)
        return s3, self.net.stages[3](s3)


def make_backbone(name: str, in_chans: int, pretrained: bool = False) -> nn.Module:
    if name == 'tiny':
        return TinyStages(in_chans)
    if name == 'pvt_v2_b0':
        return PVTv2Stages(in_chans, pretrained)
    raise ValueError(f'Unsupported backbone: {name}')
