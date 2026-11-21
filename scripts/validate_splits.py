"""Strict, auditable paired-dataset manifest validation for RD-Fuse.

Validation does NOT grant rights to dataset files, confirm public benchmark
annotations, or verify paper-reported accuracy. It prevents common pairing,
box, duplicate-ID and train/validation/test-leakage mistakes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict

PAPER_COUNTS = {'M3FD': {'train': 3024, 'val': 336, 'test': 840},
                'LLVIP': {'train': 10822, 'val': 1203, 'test': 3463}}


def read_manifest(path: str | Path) -> dict:
    file = Path(path)
    doc = json.loads(file.read_text(encoding='utf8'))
    if not isinstance(doc, dict) or not isinstance(doc.get('samples'), list):
        raise ValueError(f'{file}: expected JSON object with samples list')
    return doc


def inspect_splits(manifests: dict[str, str | Path], *, check_files: bool = False,
                   dataset: str | None = None, strict_paper_counts: bool = False) -> dict:
    """Return deterministic report; non-empty errors means the manifests FAIL."""
    errors, warnings, ids = [], [], defaultdict(list)
    data = {}
    for split, manifest_path in manifests.items():
        mf = Path(manifest_path).resolve()
        doc = read_manifest(mf)
        if doc.get('data_provenance', '').upper().startswith('SYNTHETIC'):
            warnings.append(f'{split}: synthetic data, not a benchmark split')
        samples = doc['samples']
        if not samples:
            errors.append(f'{split}: empty manifest')
        data[split] = {'path': str(mf), 'samples': len(samples),
                       'sha256': hashlib.sha256(mf.read_bytes()).hexdigest(),
                       'synthetic': doc.get('data_provenance', '').upper().startswith('SYNTHETIC')}
        seen_local = set()
        for idx, item in enumerate(samples):
            label = f'{split}[{idx}]'
            if not isinstance(item, dict):
                errors.append(f'{label}: sample is not a JSON object'); continue
            for field in ('rgb', 'thermal'):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    errors.append(f'{label}: missing {field} path')
            if 'rgb' not in item or 'thermal' not in item:
                continue
            rgb, thermal = str(item['rgb']), str(item['thermal'])
            sid = str(item.get('id') or '').strip()
            key = ('id', sid) if sid else ('rgb_thermal', rgb, thermal)
            if key in seen_local:
                errors.append(f'{label}: duplicate sample key inside split: {key}')
            seen_local.add(key)
            for field, value in [('id', sid), ('rgb_path', rgb), ('thermal_path', thermal)]:
                if value:
                    ids[(field, value.replace('\\', '/'))].append((split, idx))
            if not isinstance(item.get('objects', []), list):
                errors.append(f'{label}: objects must be a list'); continue
            for j, obj in enumerate(item.get('objects', [])):
                if not isinstance(obj, dict) or not isinstance(obj.get('category_id'), int) or obj['category_id'] < 0:
                    errors.append(f'{label}.objects[{j}]: invalid category_id')
                box = obj.get('bbox') if isinstance(obj, dict) else None
                if not isinstance(box, list) or len(box) != 4 or any(
                        not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                    errors.append(f'{label}.objects[{j}]: bbox must be finite [x,y,w,h]'); continue
                x, y, w, h = box
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    errors.append(f'{label}.objects[{j}]: invalid xywh dimensions')
            if check_files:
                from PIL import Image
                paths = [(mf.parent / rgb).resolve(), (mf.parent / thermal).resolve()]
                if any(not p.is_file() for p in paths):
                    errors.append(f'{label}: at least one image file is missing')
                else:
                    try:
                        with Image.open(paths[0]) as v, Image.open(paths[1]) as t:
                            if v.size != t.size:
                                errors.append(f'{label}: RGB/TIR dimensions differ')
                            iw, ih = v.size
                        for j, obj in enumerate(item.get('objects', [])):
                            if isinstance(obj, dict) and isinstance(obj.get('bbox'), list) and len(obj['bbox']) == 4:
                                x, y, w, h = obj['bbox']
                                if x + w > iw + 1e-5 or y + h > ih + 1e-5:
                                    errors.append(f'{label}.objects[{j}]: box outside image dimensions')
                    except Exception as exc:
                        errors.append(f'{label}: image read failed: {type(exc).__name__}')
    for key, positions in ids.items():
        if len({s for s, _ in positions}) > 1:
            errors.append(f'cross-split leakage by {key[0]}={key[1]}: {positions}')
    if dataset:
        if dataset not in PAPER_COUNTS:
            raise ValueError(f'Unknown dataset {dataset}; choose M3FD or LLVIP')
        for split, count in PAPER_COUNTS[dataset].items():
            seen = data.get(split, {}).get('samples', 0)
            if seen != count:
                msg = f'{dataset} PDF-declared {split} size={count}; manifest has {seen}'
                if strict_paper_counts: errors.append(msg)
                else: warnings.append(msg)
    return {'schema': 'rd-fuse-manifest-review-v1', 'dataset': dataset,
            'passed': not errors, 'splits': data, 'errors': errors, 'warnings': warnings,
            'statement': 'Structural and leakage checks only; does not validate official dataset provenance.'}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for split in ('train', 'val', 'test'):
        p.add_argument('--'+split, required=True, help='Path to manifest JSON')
    p.add_argument('--dataset', choices=['M3FD', 'LLVIP'])
    p.add_argument('--strict-paper-counts', action='store_true')
    p.add_argument('--check-files', action='store_true')
    p.add_argument('--out', help='Save evidence report JSON')
    a = p.parse_args()
    report = inspect_splits({'train': a.train, 'val': a.val, 'test': a.test},
                            check_files=a.check_files, dataset=a.dataset,
                            strict_paper_counts=a.strict_paper_counts)
    if a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['passed'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
