"""Paired RGB/Thermal manifest loader. Does not bundle benchmark datasets.

Input manifest: {"samples":[{"rgb":"relative/path.png",
"thermal":"relative/path.png", "objects":[{"category_id":0,
"bbox":[x,y,w,h]}]}]} with COCO pixel-space xywh boxes.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedManifestDataset(Dataset):
    def __init__(self, manifest: str, image_size: int = 640, root: str | None = None, flip_probability: float = 0., seed: int = 42):
        self.file = Path(manifest).resolve()
        data = json.loads(self.file.read_text(encoding='utf-8'))
        self.samples = data['samples']
        self.root = Path(root).resolve() if root else self.file.parent
        self.image_size = image_size
        self.flip_probability = flip_probability
        self.seed = seed
        self.epoch = 0
        if not self.samples:
            raise ValueError('Manifest has no samples')

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        rp, tp = self.root / sample['rgb'], self.root / sample['thermal']
        with Image.open(rp) as im:
            rgb = im.convert('RGB')
        with Image.open(tp) as im:
            tir = im.convert('L')
        if tir.size != rgb.size:
            raise ValueError(f'Unregistered size pair {rp} {tp}')
        iw, ih = rgb.size
        size = self.image_size
        scale = min(size / iw, size / ih)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        pad_x, pad_y = (size - nw)//2, (size - nh)//2
        def resize(im, mode):
            canvas = Image.new(mode, (size, size))
            canvas.paste(im.resize((nw, nh), Image.Resampling.BILINEAR), (pad_x, pad_y))
            a = np.array(canvas, dtype=np.float32) / 255.
            return torch.from_numpy(a).unsqueeze(0) if mode == 'L' else torch.from_numpy(a).permute(2, 0, 1)
        rgb_t, tir_t = resize(rgb, 'RGB'), resize(tir, 'L')
        boxes, labels = [], []
        for obj in sample.get('objects', []):
            x, y, w, h = [float(z) for z in obj['bbox']]
            if w <= 0 or h <= 0:
                continue
            x1 = (pad_x + x*scale)/size; y1 = (pad_y + y*scale)/size
            x2 = (pad_x + (x+w)*scale)/size; y2 = (pad_y + (y+h)*scale)/size
            boxes.append([max(0, x1), max(0, y1), min(1, x2), min(1, y2)])
            labels.append(int(obj['category_id']))
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  'labels': torch.tensor(labels, dtype=torch.long),
                  'id': sample.get('id', str(index))}
        if random.Random((self.seed * 1000003 + self.epoch) * 1000003 + index).random() < self.flip_probability:
            rgb_t, tir_t = rgb_t.flip(-1), tir_t.flip(-1)
            target["boxes"][:, [0, 2]] = 1 - target["boxes"][:, [2, 0]]
        target["sample_index"] = index
        return rgb_t, tir_t, target


def paired_collate(items):
    return torch.stack([x[0] for x in items]), torch.stack([x[1] for x in items]), [x[2] for x in items]


def synthetic_degradation(rgb: torch.Tensor, tir: torch.Tensor,
                          probability: float = .5, with_label: bool = False):
    """Independent per-sample corruption, with one modality modified at most.

    When with_label is True, return one string for B=1 or list[str|None]
    for B>1; None means no degradation for that sample.
    """
    out_r, out_t = rgb.clone(), tir.clone()
    modes = []
    for i in range(rgb.size(0)):
        if torch.rand(()).item() >= probability:
            modes.append(None)
        elif torch.rand(()).item() < .5:
            scale = torch.empty(()).uniform_(.6, .9).item()
            out_r[i] *= scale
            modes.append('R')
        else:
            sigma = torch.empty(()).uniform_(.01, .04).item()
            out_t[i] = (tir[i] + sigma*torch.randn_like(tir[i])).clamp(0, 1)
            modes.append('T')
    if with_label:
        return out_r, out_t, modes[0] if len(modes) == 1 else modes
    return out_r, out_t


def target_to_device(items: list[dict], device: torch.device):
    return [{'boxes': a['boxes'].to(device), 'labels': a['labels'].to(device), 'id': a.get('id'), 'sample_index': a.get('sample_index')} for a in items]
