"""Pure-python AP@IoU evaluator; not a substitute for a verified COCO evaluator.

Computes 101-point interpolation and aggregates over classes with GT positives.
Please cross-check with the exact evaluation library used in the manuscript.
"""
from __future__ import annotations
import numpy as np


def iou_xyxy(a, b):
    lt=np.maximum(np.array(a[:2]),np.array(b[:2]))
    rb=np.minimum(np.array(a[2:]),np.array(b[2:]))
    inter=np.maximum(rb-lt,0).prod()
    aa=np.maximum(np.array(a[2:])-np.array(a[:2]),0).prod()
    bb=np.maximum(np.array(b[2:])-np.array(b[:2]),0).prod()
    return float(inter/max(aa+bb-inter,1e-12))


def ap_at(pred: list[dict], gt: list[dict], nclasses: int, iou_threshold: float):
    aps=[]
    for c in range(nclasses):
        truths={}
        for j,g in enumerate(gt):
            truths[j]=[list(bb) for bb,ll in zip(g['boxes'],g['labels']) if int(ll)==c]
        total=sum(map(len, truths.values()))
        if total==0:
            continue
        detections=[]
        for j,p in enumerate(pred):
            for box, score, lab in zip(p['boxes'],p['scores'],p['labels']):
                if int(lab)==c:
                    detections.append((float(score),j,list(box)))
        detections.sort(key=lambda x:x[0], reverse=True)
        used={j:set() for j in truths}
        tp=[]; fp=[]
        for _,j,box in detections:
            candidates=[(iou_xyxy(box,tbox),i) for i,tbox in enumerate(truths[j]) if i not in used[j]]
            if candidates and max(candidates)[0]>=iou_threshold:
                i=max(candidates)[1]; used[j].add(i);tp.append(1);fp.append(0)
            else:
                tp.append(0); fp.append(1)
        if not tp:
            aps.append(0.0);continue
        tp=np.cumsum(tp); fp=np.cumsum(fp)
        prec=tp/np.maximum(tp+fp,1)
        recall=tp/total
        # COCO-like 101-point recall interpolation, not area trapezoid AP.
        aps.append(float(np.mean([max(prec[recall>=r],default=0.0) for r in np.linspace(0,1,101)])))
    return float(np.mean(aps)) if aps else float('nan')


def map_scores(pred: list[dict], gt: list[dict], nclasses: int):
    return {'mAP50':ap_at(pred,gt,nclasses,.5),
            'mAP50_95':float(np.nanmean([ap_at(pred,gt,nclasses,float(x)) for x in np.arange(.5,.951,.05)]))}
