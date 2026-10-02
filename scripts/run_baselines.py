"""Sequential single-GPU experiments inside this project only."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/routes.npz')
    p.add_argument('--features')
    p.add_argument('--output',default='runs/main')
    p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0])
    p.add_argument('--models',nargs='+',default=['independent','set_diffusion','regressor'])
    p.add_argument('--width',type=int,default=192)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--cover-weight',type=float,default=0.0)
    a=p.parse_args()
    if hasattr(os,'nice'): os.nice(10)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.setdefault('OMP_NUM_THREADS','4')
    env.setdefault('OPENBLAS_NUM_THREADS','4')
    shared=['--data',a.data]
    if a.features: shared+=['--features',a.features]
    for seed in a.seeds:
        for kind in a.models:
            target=out/(kind+'_seed'+str(seed))
            target.mkdir(parents=True,exist_ok=True)
            train=[sys.executable,'-m','routeset.train',*shared,'--model',kind,'--output',str(target),
                   '--steps',str(a.steps),'--seed',str(seed),'--width',str(a.width),'--eval-every',str(a.eval_every)]
            if a.cover_weight and kind=='set_diffusion': train+=['--cover-weight',str(a.cover_weight)]
            print('TRAIN '+kind+' seed='+str(seed),flush=True)
            with open(target/'train.log','w',encoding='utf-8') as log:
                subprocess.run(train,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
            evaluate=[sys.executable,'-m','routeset.evaluate',*shared,'--checkpoint',str(target/'best.pt'),
                      '--output',str(target/'evaluation'),'--splits','train','val','test','ood']
            print('EVALUATE '+kind+' seed='+str(seed),flush=True)
            with open(target/'evaluate.log','w',encoding='utf-8') as log:
                subprocess.run(evaluate,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
            metrics=json.loads((target/'evaluation'/'metrics.json').read_text(encoding='utf-8'))
            print(json.dumps({'model':kind,'seed':seed,'test':metrics['test']['metrics'],'ood':metrics['ood']['metrics']}),flush=True)


if __name__=='__main__': main()
