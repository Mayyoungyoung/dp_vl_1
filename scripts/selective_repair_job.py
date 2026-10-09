"""Isolated receipt/lock wrapper; commands run only in immutable exports."""
import argparse,datetime,os,subprocess,time
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    run=ROOT/'runs/selective_repair_v1';out=run/'jobs'/a.id;out.mkdir(parents=True,exist_ok=False)
    for family in ('feasible_space_v1','realized_coverage_v1','mode_geometry_v1'):
        if (ROOT/'runs'/family/'active.lock').exists():raise RuntimeError('Research queue active: '+family)
    lock=run/'active.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    tic=time.monotonic();receipt=dict(id=a.id,command=command,cwd=str(SOURCE),source_commit=os.environ.get('CODE_COMMIT'),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='starting',
        source_sha256={str(f.relative_to(SOURCE)):sha(f) for name in ('routeset','scripts','configs','research_selective_repair_v1','research_realized_coverage_v1','research_feasible_space_v1') for f in (SOURCE/name).rglob('*') if f.is_file() and f.suffix in ('.py','.sh','.json')},policy_sha256=sha(SOURCE/'configs/selective_repair_v1.json'))
    try:
        uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert uuid==read(SOURCE/'configs/selective_repair_v1.json')['gpu_uuid']==os.environ['RESEARCH_GPU_UUID'] and os.environ['CUDA_VISIBLE_DEVICES']=='1'
        receipt.update(status='running',gpu_uuid=uuid);write(out/'receipt.json',receipt)
        with (out/'stdout.log').open('w') as log:
            child=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
            receipt['child_pid']=child.pid;write(out/'receipt.json',receipt);code=child.wait()
        receipt.update(exit_code=code,status='completed' if code==0 else 'failed')
    except BaseException as error:code=1;receipt.update(exit_code=code,status='failed',error=repr(error))
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-tic,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(out/'receipt.json',receipt);lock.unlink()
    print({k:v for k,v in receipt.items() if k!='source_sha256'},flush=True);raise SystemExit(code)
if __name__=='__main__':main()
