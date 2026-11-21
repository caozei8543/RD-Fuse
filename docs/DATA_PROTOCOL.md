# Dataset and annotation protocol

**No M3FD or LLVIP image data/annotations are distributed in this archive.** Obtain datasets from their owners and comply with their licenses.

## Paper-declared splits

- M3FD: 4,200 registered pairs; six categories. Fixed pair-disjoint split, seed 42: 3,024 training, 336 validation, 840 test.
- LLVIP: 15,488 registered pairs for pedestrians. Declared original split 12,025 training / 3,463 testing; 1,203 of original training pairs reserved for validation; 10,822 for model development.
- Evaluation input 640×640 via aspect-preserving letterbox. Three seeds 42, 3407, 2026.

**Do not claim the same split by regenerating a random split with the same seed**: shuffle order, sorting, category handling, grouping, missing pairs and source file IDs all matter. Export the author-used exact file IDs and include SHA256 hashes.

## Manifest JSON schema

```json
{
  "samples": [
    {
      "id": "original_dataset_image_id",
      "rgb": "/data/M3FD/RGB/0001.png",
      "thermal": "/data/M3FD/TIR/0001.png",
      "objects": [{"category_id": 0, "bbox": [100.0, 120.0, 40.0, 50.0]}]
    }
  ]
}
```

`bbox` is **absolute original image pixel coordinates** in COCO `xywh` order. `category_id` must be contiguous zero-based `[0, num_classes)`. The loader maps paired images to the same letterboxed square canvas and yields normalized `xyxy` targets. Thermal is loaded as a one-channel grayscale image. Real thermal calibration/normalization must be verified against the original detector code.

Converter for COCO annotations that use same RGB/TIR file stems:

```bash
python -m scripts.coco_to_manifest --coco /path/instances.json --rgb-dir /path/rgb --thermal-dir /path/thermal --out data/M3FD_train_author_verified.json
```

If your file names differ across modalities, **do not silently match incorrect pairs**. Modify converter with the actual pair-ID mapping.

Before reporting paper-level metrics, compare the decoding/IoU assignment/evaluation to the original published protocol (e.g., official COCO metrics for mAP50:95). The included 101-point AP evaluator is a minimal reference, not pycocotools, and does not implement all ignore/crowd/maxDet/area behavior.
