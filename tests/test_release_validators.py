"""Deterministic tests for checks that guard research-integrity claims."""
import json
from pathlib import Path
import pytest
from scripts.validate_splits import inspect_splits
from scripts.validate_latency_profile import validate


def make_manifest(path: Path, ids: list[str]):
    path.write_text(json.dumps({'samples':[
        {'id': idx, 'rgb': f'{idx}.png', 'thermal': f'{idx}_tir.png',
         'objects': [{'category_id': 0, 'bbox': [1, 1, 10, 10]}]}
        for idx in ids]}), encoding='utf8')


def test_distinct_manifest_splits_pass(tmp_path):
    m = {x: tmp_path / f'{x}.json' for x in ['train','val','test']}
    for x,i in zip(m,['a','b','c']):make_manifest(m[x],[i])
    out=inspect_splits(m)
    assert out['passed'] and not out['errors']
    assert all(len(v['sha256'])==64 for v in out['splits'].values())


def test_reused_id_between_split_is_rejected(tmp_path):
    m = {x: tmp_path / f'{x}.json' for x in ['train','val','test']}
    for x,i in zip(m,['a','a','c']):make_manifest(m[x],[i])
    out=inspect_splits(m)
    assert not out['passed'] and any('leakage' in msg for msg in out['errors'])


def test_duplicate_within_split_rejected(tmp_path):
    m = {x: tmp_path / f'{x}.json' for x in ['train','val','test']}
    for x,i in zip(m,[['a','a'],['b'],['c']]):make_manifest(m[x],i)
    out=inspect_splits(m)
    assert not out['passed'] and any('duplicate' in msg for msg in out['errors'])


def test_paper_counts_are_not_silently_assumed(tmp_path):
    m = {x: tmp_path / f'{x}.json' for x in ['train','val','test']}
    for x,i in zip(m,['a','b','c']):make_manifest(m[x],[i])
    assert inspect_splits(m,dataset='M3FD')['passed']
    assert not inspect_splits(m,dataset='M3FD',strict_paper_counts=True)['passed']


def test_profile_valid_demo_but_not_verified():
    f=Path(__file__).parent.parent/'configs'/'demo_profile.json'
    doc=json.loads(f.read_text())
    res=validate(doc)
    assert res['passed'] and not res['original_jetson_measurements_independently_verified']


def test_profile_rejects_unphysical_costs():
    p={'shared_ms':0,'path_ms':{'L':1,'R':2,'T':2,'C0':3},
       'windows_ms':{'0':0,'2':.5,'4':.4,'8':1.,'12':2.,'16':3.}}
    out=validate(p)
    assert not out['passed'] and any('nondecreasing' in x for x in out['errors'])


def test_profile_rejects_missing_values():
    assert not validate({'path_ms': {}, 'windows_ms': {}})['passed']
