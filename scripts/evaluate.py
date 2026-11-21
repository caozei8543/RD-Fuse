"""Evaluation requires real paired annotations and an actual checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from rd_fuse import RDFuse,LatencyProfile
from rd_fuse.data import PairedManifestDataset,paired_collate
from rd_fuse.detection import decode_predictions
from rd_fuse.metrics import map_scores


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--checkpoint',required=True)
    a.add_argument('--manifest',required=True)
    a.add_argument('--profile',default=None)
    a.add_argument('--budget-ratio',type=float,default=.7)
    a.add_argument('--forced',choices=['L','R','T','C0','C'],default=None)
    a.add_argument('--device',default='cpu')
    a.add_argument('--metric',choices=['compact','coco'],default='compact')
    a.add_argument('--out',default='artifacts/logs/eval_reference.json')
    args=a.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=ck['config']['model']; ecfg=ck['config'].get('evaluation',{})
    m=RDFuse(num_classes=cfg['num_classes'],backbone=cfg['backbone'],pretrained=False,
             semantic_size=cfg['semantic_size'],grid=cfg['grid'])
    m.load_state_dict(ck['model'],strict=True)
    m=m.to(args.device).eval()
    prof=LatencyProfile(args.profile) if args.profile else None
    if not args.forced and not prof:
        a.error('Supply --forced or a measured --profile')
    if prof and not prof.is_verified:
        print('The selected profile has no bundled measurement evidence; see profile provenance.')
    ds=PairedManifestDataset(args.manifest,cfg['image_size'])
    loader=DataLoader(ds,batch_size=1,collate_fn=paired_collate)
    predictions=[]; labels=[]; routes=[]
    with torch.no_grad():
        for rgb,tir,gt in loader:
            out=m(rgb.to(args.device),tir.to(args.device),forced=args.forced,
                  profile=prof,budget_ms=prof.budget(args.budget_ratio) if prof else None)
            pred=decode_predictions(out.predictions, score_thresh=ecfg.get('score_threshold',.2),iou_thresh=ecfg.get('nms_iou_threshold',.5),max_per_level=ecfg.get('max_per_level',100),max_detections=ecfg.get('max_detections',300))[0]
            predictions.append({k:v.cpu().numpy().tolist() for k,v in pred.items()})
            labels.append({'boxes':gt[0]['boxes'].numpy().tolist(),'labels':gt[0]['labels'].numpy().tolist()})
            routes.append({'sample_id':gt[0]['id'],'route':out.mode,'k':len(out.window_indices),'indices':list(out.window_indices),'estimated_cost_ms':out.cost_ms,'global_utility':out.global_utility[0].cpu().tolist(),'local_utility':out.local_utility[0].cpu().tolist()})
    if args.metric == 'coco':
        from rd_fuse.coco_metrics import coco_scores
        result=coco_scores(predictions,labels,cfg['num_classes'],cfg['image_size'])
    else: result=map_scores(predictions,labels,cfg['num_classes'])
    result['metric_protocol']=args.metric
    result['predictions']=[{'sample_id':r['sample_id'],**p} for r,p in zip(routes,predictions)]
    from collections import Counter
    result['route_counts']=dict(Counter(f"{r['route']}:{r['k']}" for r in routes))
    result.update({'run_scope':ck.get('run_scope', 'unspecified'),
                   'manifest':args.manifest,
                   'number_of_images':len(ds),'routes':routes})
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('routes','predictions')},indent=2))

if __name__=='__main__':main()
