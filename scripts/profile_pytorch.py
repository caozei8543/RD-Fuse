"""Measure reference-model *PyTorch* end-to-end timings; NOT TensorRT.

Output can inform a PyTorch demo latency model but MUST NOT be labeled as
Jetson Orin NX TensorRT data from the manuscript.
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from rd_fuse import RDFuse
from rd_fuse.data import PairedManifestDataset


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def benchmark(fn, device, warmup: int, runs: int):
    with torch.inference_mode():
        for _ in range(warmup):fn()
        sync(device)
        samples=[]
        for _ in range(runs):
            sync(device); start=time.perf_counter();fn();sync(device)
            samples.append((time.perf_counter()-start)*1000)
    samples.sort()
    return {'mean_ms':statistics.mean(samples),'median_ms':statistics.median(samples),
            'p95_ms':samples[min(len(samples)-1,int(.95*len(samples)))]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--warmup',type=int,default=200)
    p.add_argument('--runs',type=int,default=1000)
    p.add_argument('--out',default='artifacts/profiles/reference_pytorch.json')
    a=p.parse_args()
    if a.runs<2 or a.warmup<0:
        p.error('runs>=2, warmup>=0')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cfg=ck['config']['model'];device=torch.device(a.device)
    m=RDFuse(num_classes=cfg['num_classes'],backbone=cfg['backbone'],
             pretrained=False,semantic_size=cfg['semantic_size'],grid=cfg['grid']).to(device).eval()
    m.load_state_dict(ck['model'])
    ds=PairedManifestDataset(a.manifest,cfg['image_size'])
    rgb,tir,_=ds[0];rgb=rgb[None].to(device);tir=tir[None].to(device)
    timing={}
    for mode,k in [('L',0),('R',0),('T',0),('C0',0)]+[('C',n) for n in [2,4,8,12,16]]:
        idx=list(range(k)) if k else []
        measured=benchmark(lambda:m(rgb,tir,forced=mode,windows=idx),device,a.warmup,a.runs)
        timing[f'{mode}:{k}']=measured
        print(mode,k,measured)
    c0=timing['C0:0']['mean_ms']
    increments={str(k): max(0.,timing[f'C:{k}']['mean_ms']-c0) for k in [2,4,8,12,16]}
    result={'description':'REAL MEASURED PYTORCH REFERENCE LATENCY, NOT TENSORRT OR PAPER MEASUREMENTS',
            'device':str(device),'runtime':f'PyTorch {torch.__version__}',
            'measured_at_utc':datetime.now(timezone.utc).isoformat(),
            'python':platform.python_version(),'checkpoint':a.checkpoint,'manifest':a.manifest,
            'validated_for_target_hardware':False,
            'shared_ms':0.0,
            'path_ms':{mode:timing[f'{mode}:0']['mean_ms'] for mode in ['L','R','T','C0']},
            'windows_ms':{'0':0.,**increments},
            'raw_measurements':timing,
            'caution':'Zero shared cost is a reparameterization, not an estimate of shared latency; noisy incremental costs may be clamped to zero.'}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Wrote PYTORCH-ONLY profile',out)

if __name__=='__main__':main()
