"""Checkpoint inference and JSON + optional visualization, batch=1."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from rd_fuse import RDFuse,LatencyProfile
from rd_fuse.data import PairedManifestDataset
from rd_fuse.detection import decode_predictions


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--sample',type=int,default=0)
    ap.add_argument('--profile',default=None)
    ap.add_argument('--budget-ratio',type=float,default=.7)
    ap.add_argument('--forced',choices=['L','R','T','C0','C'],default=None)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--out-dir',default='artifacts/logs/inference')
    args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True);cfg=ck['config']['model']; ecfg=ck['config'].get('evaluation',{})
    model=RDFuse(num_classes=cfg['num_classes'],backbone=cfg['backbone'],pretrained=False,
                 semantic_size=cfg['semantic_size'],grid=cfg['grid']).to(args.device).eval()
    model.load_state_dict(ck['model'])
    prof=LatencyProfile(args.profile) if args.profile else None
    if not args.forced and not prof:ap.error('Supply --forced or --profile')
    ds=PairedManifestDataset(args.manifest,cfg['image_size'])
    rgb,thermal,target=ds[args.sample]
    with torch.inference_mode():
        out=model(rgb[None].to(args.device),thermal[None].to(args.device),
                  forced=args.forced,profile=prof,
                  budget_ms=prof.budget(args.budget_ratio) if prof else None)
        det=decode_predictions(out.predictions,score_thresh=ecfg.get('score_threshold',.2),iou_thresh=ecfg.get('nms_iou_threshold',.5),max_per_level=ecfg.get('max_per_level',100),max_detections=ecfg.get('max_detections',300))[0]
    outdir=Path(args.out_dir);outdir.mkdir(parents=True,exist_ok=True)
    data={'mode':out.mode,'selected_windows':list(out.window_indices),
          'boxes':det['boxes'].cpu().tolist(),'scores':det['scores'].cpu().tolist(),
          'labels':det['labels'].cpu().tolist(),
          'run_scope':ck.get('run_scope','unspecified')}
    (outdir/'result.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    arr=(rgb.permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')
    img=Image.fromarray(arr);d=ImageDraw.Draw(img)
    s=cfg['image_size']
    for xyxy,score in zip(data['boxes'],data['scores']):
        if score >=.2:
            d.rectangle(tuple(int(v*s) for v in xyxy),outline=(0,255,0),width=2)
    img.save(outdir/'prediction.png')
    print(json.dumps({'mode':data['mode'],'windows':data['selected_windows'],'count':len(data['boxes'])},indent=2))
    print('Saved to',outdir)

if __name__=='__main__':main()
