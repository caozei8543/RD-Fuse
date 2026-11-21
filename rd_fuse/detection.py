"""Minimal anchor-free focal/GIoU/centerness head supervision and box decoder.

This provides a fully runnable baseline; the manuscript does not specify all
assignment, centerness, decoding, or NMS choices. These MUST be reconciled with
the original author code before claiming reproduction of published AP.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def boxes_from_ltrb(ltrb: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                    height: int, width: int) -> torch.Tensor:
    """LTRB distances are measured in strides of this feature map."""
    cx, cy = (cols + .5) / width, (rows + .5) / height
    l, t, r, b = (ltrb[..., i] for i in range(4))
    return torch.stack([cx - l / width, cy - t / height, cx + r / width, cy + b / height], -1)


def giou_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(a[..., :2], b[..., :2]); x2 = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (x2-x1).clamp_min(0).prod(-1)
    aa = (a[..., 2:]-a[..., :2]).clamp_min(0).prod(-1)
    bb = (b[..., 2:]-b[..., :2]).clamp_min(0).prod(-1)
    union = (aa + bb - inter).clamp_min(1e-8)
    xmin = torch.minimum(a[..., :2], b[..., :2]); xmax = torch.maximum(a[..., 2:], b[..., 2:])
    enclosure = (xmax-xmin).clamp_min(0).prod(-1).clamp_min(1e-8)
    return 1 - (inter / union - (enclosure - union)/enclosure)


def build_targets(preds: list[dict[str, torch.Tensor]], targets: list[dict]) -> list[dict]:
    """One center location per object; scale selected by image-relative size.

    This is a documented simplification (no FCOS center sampling/ATSS).
    targets are [{'boxes': Tensor[N,4] xyxy normalized, 'labels': Tensor[N]}].
    """
    result = []
    for p in preds:
        B, C, H, W = p['cls'].shape
        result.append({'cls': p['cls'].new_zeros((B, C, H, W)),
                       'center': p['cls'].new_zeros((B, 1, H, W)),
                       'positives': []})
    for b, obj in enumerate(targets):
        boxes = obj['boxes'].to(preds[0]['cls'].device)
        labels = obj['labels'].to(preds[0]['cls'].device)
        for box, label in zip(boxes, labels):
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue
            area_side = max((box[2]-box[0]).item(), (box[3]-box[1]).item())
            scale = 0 if area_side < .10 else (1 if area_side < .25 else 2)
            H, W = preds[scale]['cls'].shape[-2:]
            cx, cy = (box[0]+box[2]) * .5, (box[1]+box[3]) * .5
            x = min(W-1, max(0, int(cx.item() * W)))
            y = min(H-1, max(0, int(cy.item() * H)))
            cls_id = int(label.item())
            if cls_id < 0 or cls_id >= preds[scale]['cls'].shape[1]:
                raise ValueError(f'Class ID {cls_id} out of range')
            result[scale]['cls'][b, cls_id, y, x] = 1.
            result[scale]['center'][b, 0, y, x] = 1.
            result[scale]['positives'].append((b, y, x, box))
    return result


def focal_bce(logits: torch.Tensor, target: torch.Tensor, alpha: float = .25, gamma: float = 2.0):
    p = logits.sigmoid()
    pt = torch.where(target > .5, p, 1-p)
    a = torch.where(target > .5, alpha, 1-alpha)
    return (a * (1-pt).pow(gamma) * F.binary_cross_entropy_with_logits(logits, target, reduction='none')).sum()


def detection_loss(preds: list[dict[str, torch.Tensor]], targets: list[dict]) -> torch.Tensor:
    built = build_targets(preds, targets)
    n = max(1, sum(len(t['positives']) for t in built))
    class_loss = sum(focal_bce(p['cls'], t['cls']) for p, t in zip(preds, built)) / n
    bbox_losses = []; cent_losses = []
    for p, t in zip(preds, built):
        H, W = p['cls'].shape[-2:]
        for b, y, x, box in t['positives']:
            rows = torch.tensor(y, device=box.device, dtype=box.dtype)
            cols = torch.tensor(x, device=box.device, dtype=box.dtype)
            decoded = boxes_from_ltrb(p['ltrb'][b, :, y, x], rows, cols, H, W)
            bbox_losses.append(giou_loss(decoded, box))
            cent_losses.append(F.binary_cross_entropy_with_logits(p['center'][b, 0, y, x],
                                                                  torch.ones_like(p['center'][b, 0, y, x])))
    boxloss = torch.stack(bbox_losses).sum()/n if bbox_losses else class_loss * 0
    centloss = torch.stack(cent_losses).sum()/n if cent_losses else class_loss * 0
    return class_loss + 2.0 * boxloss + centloss


@torch.no_grad()
def decode_predictions(preds: list[dict[str, torch.Tensor]], score_thresh: float = .20,
                       max_per_level: int = 100, iou_thresh: float = .50, max_detections: int = 300) -> list[dict]:
    B = preds[0]['cls'].shape[0]
    all_results = []
    for b in range(B):
        bb, ss, ll = [], [], []
        for p in preds:
            _, nc, H, W = p['cls'].shape
            probs = p['cls'][b].sigmoid() * p['center'][b].sigmoid()
            top = probs.flatten().topk(min(max_per_level, probs.numel()))
            good = top.values >= score_thresh
            if good.sum().item() == 0:
                continue
            ind, scores = top.indices[good], top.values[good]
            cls = ind // (H*W)
            iy, ix = (ind % (H*W)) // W, ind % W
            distances = p['ltrb'][b, :, iy, ix].T
            boxes = boxes_from_ltrb(distances, iy.float(), ix.float(), H, W).clamp(0, 1)
            bb.append(boxes); ss.append(scores); ll.append(cls)
        if not bb:
            all_results.append({'boxes': preds[0]['cls'].new_zeros((0, 4)),
                                'scores': preds[0]['cls'].new_zeros((0,)),
                                'labels': torch.empty(0, device=preds[0]['cls'].device, dtype=torch.long)})
            continue
        boxes, scores, labels = torch.cat(bb), torch.cat(ss), torch.cat(ll)
        # Pure PyTorch class-aware greedy NMS avoids environment-dependent torchvision native ops.
        keep = []
        for label in labels.unique():
            ids = (labels == label).nonzero().flatten()
            ids = ids[scores[ids].argsort(descending=True)]
            while ids.numel():
                first = ids[0]
                keep.append(first.item())
                if ids.numel() == 1:
                    break
                a, rest = boxes[first], boxes[ids[1:]]
                lt = torch.maximum(rest[:, :2], a[:2]); rb = torch.minimum(rest[:, 2:], a[2:])
                inter = (rb-lt).clamp_min(0).prod(1)
                area0 = (a[2:]-a[:2]).clamp_min(0).prod()
                area1 = (rest[:, 2:]-rest[:, :2]).clamp_min(0).prod(1)
                iou = inter/(area0+area1-inter).clamp_min(1e-8)
                ids = ids[1:][iou <= iou_thresh]
        ks = torch.tensor(keep, dtype=torch.long, device=boxes.device)
        ks = ks[scores[ks].argsort(descending=True)][:max_detections]
        all_results.append({'boxes': boxes[ks], 'scores': scores[ks], 'labels': labels[ks]})
    return all_results
