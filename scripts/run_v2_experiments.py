"""Recorded, sequential experiment queue; subprocess failures remain evidence."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--budgets', nargs='+', type=int, default=[4])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--objectives', nargs='+', default=['subset','positive'])
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    os.nice(10)
    record = {'pid':os.getpid(),'start_utc':timestamp(),'code_commit':os.environ.get('CODE_COMMIT'),
              'status':'running','jobs':[],'host':'wzy3090','gpu_uuid':os.environ.get('RESEARCH_GPU_UUID')}
    def save():
        (out/'queue_status.json').write_text(json.dumps(record, indent=2))
    save()
    failures = 0
    for seed in a.seeds:
        for k in a.budgets:
            for objective in a.objectives:
                run_id = '{}_k{}_seed{}'.format(objective,k,seed)
                target = out/run_id; target.mkdir(parents=True, exist_ok=True)
                if (target/'status.json').exists() and json.loads((target/'status.json').read_text()).get('status') == 'completed':
                    print('Already completed '+run_id, flush=True); continue
                command = [sys.executable,'-m','routeset.train_v2','--data',a.data,'--output',str(target),
                           '--objective',objective,'--candidates',str(k),'--seed',str(seed),
                           '--steps',str(a.steps),'--eval-every',str(a.eval_every)]
                if a.resume and (target/'last.pt').exists(): command += ['--resume']
                job = {'run_id':run_id,'command':command,'start_utc':timestamp(), 'status':'running',
                       'log':str(target/'train.log'),'resume_command':command+([] if '--resume' in command else ['--resume'])}
                record['jobs'].append(job)
                with open(target/'train.log','a',encoding='utf-8') as log:
                    child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    job['pid'] = child.pid; save()
                    print(json.dumps(job),flush=True)
                    code = child.wait()
                job.update({'exit_code':code,'end_utc':timestamp(),'status':'completed' if code==0 else 'failed'})
                with open(out/'registry.jsonl','a') as reg: reg.write(json.dumps(job)+'\n')
                failures += code != 0; save(); print(json.dumps(job),flush=True)
    record.update({'status':'completed' if failures==0 else 'completed_with_failures','exit_code':int(failures>0),'end_utc':timestamp()})
    save()
    sys.exit(int(failures>0))


if __name__ == '__main__': main()
