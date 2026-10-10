"""One-thread CPU2 pilot execution; same fixed controller, separate display."""
import datetime,os,subprocess,sys,time,traceback
from pathlib import Path
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,read,write,sha

def run():
    name='body_event_DEV_execution_suite_seed0_v1';out=RUN/'jobs'/name;out.mkdir(parents=True,exist_ok=False)
    lock=RUN/'body_aux_cpu.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    receipt=dict(command=sys.argv,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE.name,
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for directory in ('research_selective_repair_v1','scripts') for p in (SOURCE/directory).rglob('*') if p.suffix in ('.py','.sh')},
        affinity=sorted(os.sched_getaffinity(0)),threads=1,status='running',scope='Reused DEV two-family actual returned4 pilot; native timing/RNG not proven controlled; no formal causal CI')
    write(out/'receipt.json',receipt);tic=time.monotonic();code=1
    try:
        assert receipt['affinity']==[2] and os.environ['SELECTIVE_BODY_CPU_SET']=='2'
        assert read(RUN/'jobs/body_event_DEV_screen_seed0_v1/receipt.json')['status']=='completed'
        for family in ('feasible_space_v1','realized_coverage_v1','mode_geometry_v1'):assert not (ROOT/'runs'/family/'active.lock').exists()
        main=RUN/'active.lock'
        if main.exists():
            args=(Path('/proc')/main.read_text().strip()/'cmdline').read_bytes().decode().split('\0');job=args[args.index('--id')+1]
            active=read(RUN/'jobs'/job/'receipt.json');assert job.startswith('body_execution_train_') and active['status']=='running'
            assert 'execution' in active['command'] and any('render_selective_body_v1.sh' in v for v in active['command'])
            receipt['overlapping_one_thread_CPU_teacher']=job
        command=['bash',str(SOURCE/'scripts/render_selective_body_v1.sh'),'suite','--name',name,'--event']
        with (out/'stdout.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);receipt['child_pid']=child.pid;receipt['actual_command']=command;write(out/'receipt.json',receipt);code=child.wait()
    except BaseException as error:receipt['error']=repr(error);receipt['traceback']=traceback.format_exc()
    finally:
        receipt.update(status='completed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tic,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(out/'receipt.json',receipt);lock.unlink()
    print({k:v for k,v in receipt.items() if k!='source_sha256'},flush=True);raise SystemExit(code)

if __name__=='__main__':run()
