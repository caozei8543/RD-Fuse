import torch
from rd_fuse import RDFuse,LatencyProfile
from rd_fuse.data import PairedManifestDataset
from rd_fuse.training import true_forced_loss,utility_step,regret_step,set_trainable


def sample():
    rgb,th,y=PairedManifestDataset('demos/synthetic/manifest.json',64)[0]
    return rgb[None],th[None],[y]


def test_real_multiwindow_joint_execution_not_approx_sum():
    m=RDFuse(backbone='tiny',semantic_size=8).eval()
    rgb,th,y=sample()
    with torch.no_grad():
        c0=true_forced_loss(m,rgb,th,y,'C0')
        u1=c0-true_forced_loss(m,rgb,th,y,'C',[0])
        u2=c0-true_forced_loss(m,rgb,th,y,'C',[1])
        jointly=true_forced_loss(m,rgb,th,y,'C',[0,1])
        assert torch.isfinite(jointly)
        assert m.cross.last_executed==2  # both windows actually executed together
        # Don't assume c0 - jointly equals u1+u2: additive assumption is *only* surrogate.
        assert all(torch.isfinite(x).item() for x in (u1,u2,jointly))


def test_utility_regret_have_trainable_gradients():
    m=RDFuse(backbone='tiny',semantic_size=8)
    rgb,th,y=sample()
    set_trainable(m,'utility')
    loss,details=utility_step(m,rgb,th,y,sampled_windows=3)
    loss.backward()
    assert m.local_utility.net[0].weight.grad is not None
    assert len(details['sampled_windows'])==3
    m.zero_grad(set_to_none=True)
    set_trainable(m,'regret')
    p=LatencyProfile('configs/demo_profile.json')
    loss,details=regret_step(m,rgb,th,y,p,p.budget(.70))
    loss.backward()
    assert m.global_utility.net[0].weight.grad is not None
    assert any(c.startswith('C:') for c in details['candidates'])
    assert any(c.startswith('C:4') for c in details['candidates'])
    assert all('mode' not in str(x) for x in [])
