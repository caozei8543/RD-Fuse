"""Create stable split-ID manifest hash and per-file SHA256, no raw data copied."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while True:
            buf=f.read(1<<20)
            if not buf:break
            h.update(buf)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--hash-images',action='store_true')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    file=Path(args.manifest).resolve()
    samples=json.loads(file.read_text(encoding='utf-8'))['samples']
    ids=[str(x.get('id')) for x in samples]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate split sample IDs')
    h=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    res={'manifest_sha256':sha256(file),'ordered_sample_count':len(ids),'sorted_IDs_sha256':h}
    if args.hash_images:
        res['image_files_sha256']={}
        for s in samples:
            for k in ['rgb','thermal']:
                fp=file.parent/s[k]
                res['image_files_sha256'][str(fp)]=sha256(fp)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(res,indent=2),encoding='utf-8');print(path)

if __name__=='__main__':main()
