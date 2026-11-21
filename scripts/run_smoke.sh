#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python -m scripts.make_synthetic_demo --out demos/synthetic --count 4 --size 64
python -m scripts.train --config configs/smoke.yaml --manifest demos/synthetic/manifest.json --phase forced --epochs 1 --max-steps 1 --out artifacts/checkpoints/demo_forced.pt
python -m scripts.train --config configs/smoke.yaml --manifest demos/synthetic/manifest.json --phase utility --resume artifacts/checkpoints/demo_forced.pt --epochs 1 --max-steps 1 --out artifacts/checkpoints/demo_utility.pt
python -m scripts.train --config configs/smoke.yaml --manifest demos/synthetic/manifest.json --phase regret --resume artifacts/checkpoints/demo_utility.pt --epochs 1 --max-steps 1 --out artifacts/checkpoints/demo_regret.pt
python -m scripts.evaluate --checkpoint artifacts/checkpoints/demo_regret.pt --manifest demos/synthetic/manifest.json --forced L --out artifacts/logs/demo_eval.json
python -m scripts.infer --checkpoint artifacts/checkpoints/demo_regret.pt --manifest demos/synthetic/manifest.json --profile configs/demo_profile.json
printf '\nSynthetic software-pipeline test complete.\n'
