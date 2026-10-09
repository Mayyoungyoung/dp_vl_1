"""Isolated uncapped ledger; immutable commands/source and failure receipts."""
import argparse,datetime,os,subprocess,time
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    run=ROOT/'runs/feasible_space_v1';out=run/'jobs'/a.id;out.mkdir(parents=True,exist_ok=False)
    for family in ('realized_coverage_v1','verified_set_v1','observed_probability_v1','mode_geometry_v1'):
        if (ROOT/'runs'/family/'active.lock').exists():raise RuntimeError('Prior research queue active: '+family)
    lock=run/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    policy=SOURCE/'configs/feasible_space_v1.json';tic=time.monotonic()
    receipt=dict(id=a.id,command=command,pid=os.getpid(),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=os.environ.get('CODE_COMMIT'),policy_sha256=sha(policy),cwd=str(SOURCE),status='starting',
        source_sha256={str(f.relative_to(SOURCE)):sha(f) for name in ('routeset','scripts','configs','tests','research_realized_coverage_v1','research_feasible_space_v1') for f in (SOURCE/name).rglob('*') if f.is_file() and f.suffix in ('.py','.sh','.json')})
    try:
        gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert gpu==read(policy)['gpu_uuid']==os.environ['RESEARCH_GPU_UUID'] and os.environ['CUDA_VISIBLE_DEVICES']=='1'
        receipt.update(gpu_uuid=gpu,status='running',previous_spent_seconds=sum(read(f).get('elapsed_seconds',0) for f in (run/'jobs').glob('*/receipt.json')));write(out/'receipt.json',receipt)
        with (out/'stdout.log').open('w') as log:
            child=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt['child_pid']=child.pid;write(out/'receipt.json',receipt);code=child.wait()
        receipt.update(exit_code=code,status='completed' if code==0 else 'failed')
    except BaseException as e:code=1;receipt.update(exit_code=code,status='failed',error=repr(e))
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-tic,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(out/'receipt.json',receipt);lock.unlink()
    print({k:v for k,v in receipt.items() if k!='source_sha256'},flush=True);raise SystemExit(code)

if __name__=='__main__':main()
