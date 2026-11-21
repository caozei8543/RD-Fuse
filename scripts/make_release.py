"""Rebuild a clean and integrity-checked source archive and integrity file.

Runs from repository root: python -m scripts.make_release --out /path/archive.zip
"""
import argparse
import hashlib
from pathlib import Path
import zipfile

EXCLUDE_DIRS={'__pycache__','.pytest_cache','.git','.venv'}


def wanted(path: Path, root: Path) -> bool:
    rel=path.relative_to(root)
    if any(part in EXCLUDE_DIRS or part.endswith('.egg-info') for part in rel.parts):return False
    if path.suffix in ('.pyc','.pyo','.aux','.log','.out','.toc','.fls','.fdb_latexmk','.synctex.gz'):return False
    if rel.parts[0]=='artifacts':
        return False
    if len(rel.parts)>=2 and rel.parts[0]=='artifacts' and rel.parts[1] in ('logs','profiles'):
        return False
    if rel.parts[0]=='data' and rel.name!='README.md':return False
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='../RD-Fuse_IEEE_Access_Clean.zip')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent.parent
    files=sorted((p for p in root.rglob('*') if p.is_file() and wanted(p,root)),key=lambda p:str(p.relative_to(root)))
    # Manifest is recomputed from concrete content; exclude manifest from its own list.
    checks=[]
    for p in files:
        if p.name=='SHA256SUMS.txt':continue
        checks.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root)))
    (root/'SHA256SUMS.txt').write_text('\n'.join(checks)+'\n',encoding='utf-8')
    files=sorted((p for p in root.rglob('*') if p.is_file() and wanted(p,root)),key=lambda p:str(p.relative_to(root)))
    outfile=Path(args.out).resolve();outfile.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(outfile,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as zf:
        for p in files:
            arc='RD-Fuse-IEEE-Access-Package/'+p.relative_to(root).as_posix()
            zf.write(p,arc)
    with zipfile.ZipFile(outfile) as zf:
        bad=zf.testzip()
        if bad:raise RuntimeError(f'Corrupted member: {bad}')
        print(f'ZIP OK: {outfile} ({len(zf.namelist())} files; {outfile.stat().st_size/2**20:.2f} MiB)')
    print('SHA256(zip):',hashlib.sha256(outfile.read_bytes()).hexdigest())

if __name__=='__main__':main()
