"""Convert paired RGB/TIR COCO-style annotations to the manifest schema.

Images must match by RGB file_name stem and thermal file stem; verify RGB/TIR
registration and category maps before using a benchmark.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--coco',required=True,help='COCO instances JSON for paired images')
    p.add_argument('--rgb-dir',required=True)
    p.add_argument('--thermal-dir',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--class-map',default=None,help='JSON mapping original category_id strings to zero-based IDs')
    args=p.parse_args()
    data=json.loads(Path(args.coco).read_text())
    rdir,tdir=Path(args.rgb_dir).resolve(),Path(args.thermal_dir).resolve()
    thermo={p.stem:p for p in tdir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')}
    classes=json.loads(Path(args.class_map).read_text()) if args.class_map else {str(c['id']):i for i,c in enumerate(sorted(data['categories'], key=lambda x:x['id']))}
    anns=defaultdict(list)
    for ann in data['annotations']:
        key=str(ann['category_id'])
        if key not in classes:continue
        anns[ann['image_id']].append({'category_id':classes[key],'bbox':ann['bbox']})
    samples=[];not_found=[]
    for item in data['images']:
        name=Path(item['file_name']).name;rp=rdir/name;tp=thermo.get(Path(name).stem)
        if not rp.exists() or tp is None:
            not_found.append(name);continue
        samples.append({'id':item['id'],'rgb':str(rp),'thermal':str(tp),'objects':anns[item['id']]})
    if not_found:
        raise ValueError(f'{len(not_found)} paired images missing (first 10): {not_found[:10]}')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'source_annotation':args.coco,'class_mapping':classes,
                               'samples':samples},indent=2),encoding='utf-8')
    print('Created',out,'pairs',len(samples),'categories',classes)

if __name__=='__main__':main()
