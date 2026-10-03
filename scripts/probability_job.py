"""Budgeted, source-indexed serial job wrapper. Includes failed wall time."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.run_observed_probability import ROOT,RUN,SOURCE,POLICY,sha,read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    jobs=RUN/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    output=jobs/a.id;output.mkdir(exist_ok=False)
    lock=RUN/'active.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.monotonic()
    receipt=dict(id=a.id,command=command,pid=os.getpid(),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=os.environ.get('CODE_COMMIT'),gpu_uuid=os.environ.get('RESEARCH_GPU_UUID'),status='starting',
        policy_sha256=sha(POLICY),cwd=str(SOURCE),source_sha256={str(f.relative_to(SOURCE)):sha(f)
            for name in ('routeset','scripts','configs','tests') for f in (SOURCE/name).rglob('*') if f.is_file() and f.suffix in ('.py','.json','.sh')})
    try:
        if receipt['gpu_uuid']!=read(POLICY)['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('GPU identity env mismatch')
        gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if gpu!=receipt['gpu_uuid']:raise ValueError('Physical GPU UUID mismatch')
        spent=sum(read(f).get('elapsed_seconds',0) for f in jobs.glob('*/receipt.json'))
        allowance=read(POLICY)['gpu_wall_budget_seconds']-spent
        if allowance<=0:raise RuntimeError('Cumulative experiment budget exhausted')
        receipt.update(previous_spent_seconds=spent,remaining_seconds=allowance,status='running')
        write(output/'receipt.json',receipt)
        with (output/'stdout.log').open('w') as log:
            child=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt['child_pid']=child.pid;write(output/'receipt.json',receipt)
            try:code=child.wait(timeout=allowance)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGTERM);child.wait();code=124
        receipt.update(exit_code=code,status='completed' if code==0 else 'failed')
    except BaseException as e:
        receipt.update(status='failed',exit_code=1,error=repr(e));code=1
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(output/'receipt.json',receipt);lock.unlink(missing_ok=True)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'}),flush=True)
    sys.exit(code)


if __name__=='__main__':main()
