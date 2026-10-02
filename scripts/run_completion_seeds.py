"""Sequential paired completion seeds with durable per-seed job records."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--steps',type=int,default=2500)
    a=p.parse_args(); failed=0
    for seed in a.seeds:
        command=[sys.executable,'scripts/record_job.py','--output',a.output,'--run-id','completion_seed%d'%seed,'--',
                 sys.executable,'-m','scripts.train_completion','--data',a.data,'--output',str(Path(a.output)/('seed%d'%seed)),
                 '--mechanism','both','--steps',str(a.steps),'--seed',str(seed),'--eval-every','500','--device','cuda']
        failed += subprocess.call(command)!=0
    sys.exit(int(failed>0))


if __name__=='__main__':main()
