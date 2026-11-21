"""Summarize verified per-seed CSV; never auto-populates unpublished results."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--csv',required=True,help='dataset,seed,mAP50,mAP50_95,precision,recall')
    a.add_argument('--out',default='artifacts/logs/three_seed_summary.json')
    args=a.parse_args()
    data=defaultdict(list)
    with open(args.csv,newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            data[row['dataset']].append(row)
    result={}
    for dataset,rows in data.items():
        seeds=[r['seed'] for r in rows]
        if len(rows)!=3 or len(set(seeds))!=3:
            raise ValueError(f'{dataset}: expected exactly 3 distinct seeds, got {seeds}')
        result[dataset]={}
        for metric in ['mAP50','mAP50_95','precision','recall']:
            vals=[float(row[metric]) for row in rows]
            result[dataset][metric]={'mean':statistics.mean(vals),'sample_std':statistics.stdev(vals),
                                     'values':dict(zip(seeds,vals))}
    if not result: raise ValueError('No actual per-seed records found')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
