"""Run the three reference stages using dataset-specific config and one seed."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['m3fd','llvip'],required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--seed',type=int,choices=[42,3407,2026],default=42)
    p.add_argument('--device',default='cuda');p.add_argument('--out-dir',default='artifacts/runs')
    a=p.parse_args();out=Path(a.out_dir)/a.dataset/str(a.seed);out.mkdir(parents=True,exist_ok=True)
    previous=None
    for phase in ('forced','utility','regret'):
        ck=out/(phase+'.pt')
        cmd=[sys.executable,'-m','scripts.train','--config',f'configs/{a.dataset}.yaml','--manifest',a.manifest,'--profile',a.profile,'--seed',str(a.seed),'--device',a.device,'--phase',phase,'--out',str(ck)]
        if previous: cmd+=['--resume',str(previous)]
        subprocess.run(cmd,check=True);previous=ck
if __name__=='__main__':main()
