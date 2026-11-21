"""Optional pycocotools all-area AP on the shared letterboxed canvas.
The manifest lacks original crowd/ignore fields; not an original benchmark harness.
"""
def coco_scores(predictions, targets, num_classes, size):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    def xywh(b):
        x,y,x2,y2=b
        return [x*size,y*size,(x2-x)*size,(y2-y)*size]
    images=[{'id':i+1,'width':size,'height':size} for i in range(len(targets))]
    categories=[{'id':i+1,'name':str(i)} for i in range(num_classes)]
    anns=[]; detections=[]
    for i,(gt,pred) in enumerate(zip(targets,predictions),1):
        for b,l in zip(gt['boxes'],gt['labels']):
            box=xywh(b); anns.append({'id':len(anns)+1,'image_id':i,'category_id':l+1,'bbox':box,'area':box[2]*box[3],'iscrowd':0})
        for b,l,s in zip(pred['boxes'],pred['labels'],pred['scores']):
            detections.append({'image_id':i,'category_id':l+1,'bbox':xywh(b),'score':s})
    if not anns: raise ValueError('COCO AP undefined without ground truth')
    if not detections: return {'mAP50':0.,'mAP50_95':0.}
    gt=COCO(); gt.dataset={'images':images,'categories':categories,'annotations':anns,'info':{},'licenses':[]};gt.createIndex()
    ev=COCOeval(gt,gt.loadRes(detections),'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    return {'mAP50':float(ev.stats[1]),'mAP50_95':float(ev.stats[0])}
