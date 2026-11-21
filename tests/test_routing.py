import torch
import pytest
from rd_fuse import RDFuse,LatencyProfile
from rd_fuse.routing import enumerate_choices, predicted_value
from rd_fuse.probe import pairwise_reliability_loss,semantic_alignment_loss


def profile():
    return LatencyProfile('configs/demo_profile.json')


def test_probe_dims_and_64_windows():
    m=RDFuse(num_classes=2,backbone='tiny',semantic_size=8)
    with torch.no_grad():
        out=m(torch.rand(1,3,64,64),torch.rand(1,1,64,64),forced='L')
    assert out.probe.global_state.shape == (1,131)
    assert out.probe.local_state.shape == (1,64,129)
    assert out.global_utility.shape == (1,3)
    assert out.local_utility.shape == (1,64)
    assert all(x['cls'].shape[1]==2 for x in out.predictions)


def test_shallow_skip_modes_and_selected_attention():
    m=RDFuse(backbone='tiny',semantic_size=8)
    m.eval();r=torch.rand(1,3,64,64);t=torch.rand(1,1,64,64)
    with torch.no_grad():
        m(r,t,forced='L')
        assert (m.rgb.deep_calls,m.tir.deep_calls)==(0,0)
        m(r,t,forced='R')
        assert (m.rgb.deep_calls,m.tir.deep_calls)==(1,0)
        m(r,t,forced='T')
        assert (m.rgb.deep_calls,m.tir.deep_calls)==(1,1)
        m(r,t,forced='C0')
        assert (m.rgb.deep_calls,m.tir.deep_calls)==(2,2)
        assert m.cross.last_executed==0
        out=m(r,t,forced='C',windows=[1,3,5,7])
        assert len(out.window_indices)==4
        assert m.cross.last_executed==4
        assert (m.rgb.deep_calls,m.tir.deep_calls)==(3,3)


def test_budget_feasible_and_degenerate_budget():
    g=torch.tensor([.1,.1,.1]);l=torch.randn(64)
    p=profile();b=p.budget(.7)
    choices=enumerate_choices(g,l,p,b)
    assert len(choices)>2
    assert all(c.cost_ms<=b+1e-8 for c in choices)
    assert any(c.mode=='C' and c.k>1 for c in choices)
    with pytest.raises(ValueError,match='no feasible path'):
        enumerate_choices(g,l,p,.01)


def test_cost_and_sign_and_reliability():
    p=profile()
    assert p.cost('C',16)>p.cost('C0')
    gain=predicted_value(torch.tensor([.5,.2,.4]),torch.full((64,),.1),'C',4,1.3,lam=.005)
    assert gain.item() == pytest.approx(.4+.4-.005*1.3)
    good=pairwise_reliability_loss(torch.tensor([.8]),torch.tensor([.2]),margin=.1)
    bad=pairwise_reliability_loss(torch.tensor([.2]),torch.tensor([.8]),margin=.1)
    assert good < bad


def test_alignment_losses_finite():
    m=RDFuse(backbone='tiny',semantic_size=8)
    x=m(torch.rand(2,3,64,64),torch.rand(2,1,64,64),forced='L')
    global_loss,local_loss=semantic_alignment_loss(x.probe)
    assert torch.isfinite(global_loss) and torch.isfinite(local_loss)
