"""One-thread TRAIN-only pilot while our CPU-only physical collection runs."""
import datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,read,write,sha

def run():
    out=RUN/'jobs/body_prefix_dense_pilot_v1';out.mkdir(parents=True,exist_ok=False)
    auxiliary=RUN/'prefix_pilot_gpu.lock';fd=os.open(auxiliary,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    receipt=dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE.name,
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for directory in ('research_selective_repair_v1','routeset','scripts') for p in (SOURCE/directory).rglob('*.py')},
        command=sys.argv,affinity=sorted(os.sched_getaffinity(0)),threads=1,memory_fraction=.35,
        scope='384completed TRAIN feedback only; state/risk learning diagnostic,not DEV or allgoal result',status='running')
    write(out/'receipt.json',receipt);tic=time.monotonic();code=1
    try:
        assert receipt['affinity']==[3] and os.environ['CUDA_VISIBLE_DEVICES']=='1'
        uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert uuid=='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab';receipt['gpu_uuid']=uuid
        for other in ('feasible_space_v1','realized_coverage_v1','mode_geometry_v1'):
            assert not (ROOT/'runs'/other/'active.lock').exists()
        # The only permitted overlapping study job is our software-rendered,
        # one-thread CPU physical teacher. Never overlap another GPU learner.
        lock=RUN/'active.lock'
        if lock.exists():
            active=[p for p in (RUN/'jobs').glob('*/receipt.json') if read(p).get('status')=='running']
            assert len(active)==1 and active[0].parent.name.startswith('body_execution_train_')
            command=read(active[0])['command'];assert 'execution' in command and any('render_selective_body_v1.sh' in v for v in command)
            receipt['overlapping_CPU_teacher']=active[0].parent.name
        from research_selective_repair_v1.body_prefix_forecast import fit
        from research_selective_repair_v1.body_binary_forecast import fit as binary_fit
        for kind in ('recurrent','nonrecurrent'):
            fit('body_prefix_pilot_%s_seed0_v1'%kind,RUN/'body_prefix_pilot_fk_v1_data/samples.npz',kind=kind,threads=1)
        binary_fit('body_binary_pilot_seed0_v1',RUN/'body_feedback_data_v1/samples.npz',threads=1)
        code=0
    except BaseException as error:
        receipt['error']=repr(error);receipt['traceback']=traceback.format_exc();print(receipt['traceback'],flush=True)
    finally:
        receipt.update(status='completed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tic,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'receipt.json',receipt);auxiliary.unlink()
    print({k:v for k,v in receipt.items() if k!='source_sha256'},flush=True);raise SystemExit(code)

if __name__=='__main__':run()
