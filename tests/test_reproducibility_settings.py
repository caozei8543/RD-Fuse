import csv, statistics
import pytest, torch, yaml
from rd_fuse.routing import LatencyProfile,enumerate_choices
from rd_fuse.data import PairedManifestDataset
from rd_fuse.training import utility_step,set_trainable
from rd_fuse import RDFuse

def test_author_confirmed_profile():
    p=LatencyProfile('configs/author_confirmed_profile.json')
    assert [p.cost(m) for m in ('L','R','T','C0')]==pytest.approx([12,15.2,15.6,17.8])
    assert [p.cost('C',k) for k in (2,4,8,12,16)]==pytest.approx([18.35,19.05,20.8,24.3,30])
    assert [p.budget(r) for r in (.7,.85,1.)]==pytest.approx([21,25.5,30])
    assert not p.is_verified
    g=torch.zeros(3); l=torch.ones(64)
    assert {c.k for c in enumerate_choices(g,l,p,p.budget(.7)) if c.mode=='C'}=={2,4,8}
    assert {c.k for c in enumerate_choices(g,l,p,p.budget(.85)) if c.mode=='C'}=={2,4,8,12}
    assert next(c.indices for c in enumerate_choices(g,l,p,p.budget(.7)) if c.mode=='C' and c.k==4)==(0,1,2,3)

def test_author_confirmed_seed_arithmetic():
    with open('docs/author_confirmed_seed_summary.csv',newline='') as f: rows=list(csv.DictReader(f))
    for d,mean,sd in [('M3FD',62.7,.22),('LLVIP',62.2,.20)]:
        selected=[r for r in rows if r['dataset']==d]
        assert [int(r['seed']) for r in selected]==[42,3407,2026]
        vals=[float(r['mAP50_95_percent']) for r in selected]
        assert statistics.mean(vals)==pytest.approx(mean)
        assert statistics.stdev(vals)==pytest.approx(sd)

def test_dataset_configs():
    for name,n,epochs in [('m3fd',6,[70,35,55]),('llvip',1,[50,25,45])]:
        with open(f'configs/{name}.yaml') as f: c=yaml.safe_load(f)
        t=c['training'];assert c['model']['num_classes']==n and t['stage_epochs']==epochs
        for stage in ('forced','utility','regret'):
            assert t['physical_batch_size'][stage]*t['accumulation_steps'][stage]==16

def test_synchronized_flip_and_cache():
    ds=PairedManifestDataset('demos/synthetic/manifest.json',64)
    f=PairedManifestDataset('demos/synthetic/manifest.json',64,flip_probability=1.)
    r,t,y=ds[0];rf,tf,yf=f[0]
    assert torch.equal(r.flip(-1),rf) and torch.equal(t.flip(-1),tf)
    assert torch.allclose(1-y['boxes'][:,[2,0]],yf['boxes'][:,[0,2]])
    m=RDFuse(backbone='tiny',semantic_size=8);set_trainable(m,'utility');cache={}
    loss1,log1=utility_step(m,r[None],t[None],[y],cache=cache)
    loss2,log2=utility_step(m,r[None],t[None],[y],cache=cache)
    assert len(cache)==1 and log1['sampled_windows']==log2['sampled_windows']
    assert torch.allclose(loss1,loss2)

def test_optional_coco_perfect():
    pytest.importorskip('pycocotools')
    from rd_fuse.coco_metrics import coco_scores
    gt=[{'boxes':[[.1,.1,.5,.5]],'labels':[0]}]
    p=[{'boxes':[[.1,.1,.5,.5]],'labels':[0],'scores':[.9]}]
    assert coco_scores(p,gt,1,640)['mAP50_95']==pytest.approx(1.)
