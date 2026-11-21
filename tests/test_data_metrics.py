import torch
from rd_fuse.data import PairedManifestDataset,synthetic_degradation
from rd_fuse.metrics import map_scores
from rd_fuse.detection import detection_loss,decode_predictions
from rd_fuse import RDFuse


def test_manifest_pair_normalization():
    ds=PairedManifestDataset('demos/synthetic/manifest.json',64)
    rgb,th,gt=ds[0]
    assert rgb.shape==(3,64,64) and th.shape==(1,64,64)
    assert gt['boxes'].shape==(1,4)
    assert gt['boxes'].min()>=0 and gt['boxes'].max()<=1


def test_synthetic_degradation_pair():
    torch.manual_seed(123)
    r,t=torch.full((1,3,8,8),.8),torch.full((1,1,8,8),.8)
    rd,td,mod=synthetic_degradation(r,t,probability=1.,with_label=True)
    assert mod in ('R','T')
    if mod=='R':
        assert torch.equal(t,td) and not torch.equal(r,rd)
    else:
        assert torch.equal(r,rd) and not torch.equal(t,td)


def test_detection_loss_decode():
    ds=PairedManifestDataset('demos/synthetic/manifest.json',64)
    r,t,y=ds[0]
    m=RDFuse(backbone='tiny',semantic_size=8)
    out=m(r[None],t[None],forced='C0')
    loss=detection_loss(out.predictions,[y]);loss.backward()
    assert torch.isfinite(loss)
    decoded=decode_predictions(out.predictions)
    assert len(decoded)==1


def test_ap_perfect():
    gt=[{'boxes':[[.1,.1,.5,.5]],'labels':[0]}]
    pred=[{'boxes':[[.1,.1,.5,.5]],'labels':[0],'scores':[.9]}]
    ap=map_scores(pred,gt,1)
    assert abs(ap['mAP50']-1)<1e-9
    assert abs(ap['mAP50_95']-1)<1e-9
