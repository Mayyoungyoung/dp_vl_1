"""Replicate ordinary RGB-D heads with/without the endpoint grounding auxiliary."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2])
    a=p.parse_args(); data=Path(a.data); output=Path(a.output)
    failed=0
    for seed in a.seeds:
        for name, weight in [('plain',0.),('aux',.02)]:
            run_id=name+'_seed'+str(seed)
            command=[sys.executable,'scripts/record_job.py','--output',str(output),'--run-id',run_id,'--',
                sys.executable,'-m','scripts.train_observed_geometry','--observations',str(data/'observations.jsonl'),
                '--supervision',str(data/'supervision.jsonl'),'--cache-dir',str(data/'qwen_cache'),
                '--output',str(output/run_id),'--threads','1','--grounding-weight',str(weight),'--seed',str(seed)]
            failed += subprocess.call(command)!=0
    sys.exit(int(failed>0))


if __name__=='__main__':main()
