"""Check profile shape and provenance markers; NOT a hardware measurement tool."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

COUNTS = (0, 2, 4, 8, 12, 16)
PATHS = ('L', 'R', 'T', 'C0')


def validate(profile: dict) -> dict:
    errors, warnings = [], []
    for field in ('shared_ms',):
        value = profile.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append(f'{field} must be nonnegative finite milliseconds')
    base = profile.get('path_ms', {})
    windows = profile.get('windows_ms', {})
    for key in PATHS:
        v = base.get(key) if isinstance(base, dict) else None
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            errors.append(f'path_ms.{key} must be nonnegative finite milliseconds')
    previous = -1.0
    for k in COUNTS:
        v = windows.get(str(k)) if isinstance(windows, dict) else None
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            errors.append(f'windows_ms.{k} must be nonnegative finite milliseconds')
            continue
        if v < previous - 1e-9:
            errors.append(f'windows_ms not nondecreasing at {k}')
        previous = v
    if isinstance(windows, dict) and windows.get('0') != 0:
        errors.append('windows_ms.0 must equal 0 exactly')
    is_marked = profile.get('validated_for_target_hardware') is True
    if is_marked:
        if 'Orin NX' not in str(profile.get('device', '')):
            warnings.append('validated_for_target_hardware=true but device lacks Orin NX identifier')
        if 'TensorRT' not in str(profile.get('runtime', '')):
            warnings.append('validated_for_target_hardware=true but runtime lacks TensorRT identifier')
        warnings.append('A self-declared verified flag does not independently establish authenticity of raw measurements.')
    else:
        warnings.append('Illustrative or unverified profile: unusable to support Table II.')
    return {'passed': not errors, 'errors': errors, 'warnings': warnings,
            'self_declared_hardware_validation': is_marked,
            'original_jetson_measurements_independently_verified': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('profile', help='A latency profile JSON')
    ap.add_argument('--out')
    args = ap.parse_args()
    doc = json.loads(Path(args.profile).read_text(encoding='utf8'))
    report = validate(doc)
    if args.out:
        p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
