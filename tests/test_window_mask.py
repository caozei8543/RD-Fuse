import torch
from rd_fuse.model import SelectiveCrossAttention


def test_interaction_projection_cannot_leak_to_inactive_windows():
    model = SelectiveCrossAttention(dim=16, out_dim=64, heads=4, grid=2).eval()
    with torch.no_grad():
        model.out.bias.fill_(3.0)
        rgb = torch.randn(1,64,8,8)
        tir = torch.randn(1,64,8,8)
        out = model(rgb, tir, [0])
    assert out.shape == (1,64,8,8)
    assert torch.count_nonzero(out[:, :, :4, 4:]) == 0
    assert torch.count_nonzero(out[:, :, 4:, :]) == 0
    assert torch.count_nonzero(out[:, :, :4, :4]) > 0
