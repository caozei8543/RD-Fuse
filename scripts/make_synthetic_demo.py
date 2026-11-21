"""Create artificial RGB/TIR pairs for functional tests ONLY, never for claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='demos/synthetic')
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--size', type=int, default=64)
    args = p.parse_args()
    rng = np.random.default_rng(42)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(args.count):
        n = args.size
        bg = rng.integers(10, 40, size=(n,n,3), dtype=np.uint8)
        rgb = Image.fromarray(bg, 'RGB')
        thermal = Image.fromarray(rng.integers(10, 50, size=(n,n), dtype=np.uint8), 'L')
        cx,cy = int(n*.35)+i%3, int(n*.4)+i%2
        ww, hh = max(6, n//6), max(6, n//5)
        ImageDraw.Draw(rgb).rectangle((cx,cy,cx+ww,cy+hh), fill=(220,85,28))
        ImageDraw.Draw(thermal).rectangle((cx,cy,cx+ww,cy+hh), fill=225)
        name_r = f'rgb_{i:03d}.png'; name_t = f'thermal_{i:03d}.png'
        rgb.save(out / name_r); thermal.save(out / name_t)
        items.append({'id': f'DEMO_{i:03d}', 'rgb': name_r, 'thermal': name_t,
                      'objects': [{'category_id':0,'bbox':[cx,cy,ww,hh]}]})
    (out/'manifest.json').write_text(json.dumps({'data_provenance':'synthetic_demo',
                                                 'samples':items}, indent=2), encoding='utf-8')
    print(f'Generated synthetic demo: {out/"manifest.json"}')


if __name__ == '__main__':
    main()
