"""One-thread TRAIN-only pilot while our CPU-only physical collection runs."""
import argparse,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,read,write,sha

def run(name,version='v1',conditioning=0,object='state',all_goal_half=False):
    out=RUN/'jobs'/name;out.mkdir(parents=True,exist_ok=False)
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
            pid=int(lock.read_text());args=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
            job=args[args.index('--id')+1];active=read(RUN/'jobs'/job/'receipt.json')
            assert job.startswith('body_execution_train_') and active['status']=='running'
            command=active['command'];assert 'execution' in command and any('render_selective_body_v1.sh' in v for v in command)
            receipt['overlapping_CPU_teacher']=job
        if object=='event':
            assert conditioning==0
            from research_selective_repair_v1.body_event_forecast import fit
        else:from research_selective_repair_v1.body_prefix_forecast import fit
        from research_selective_repair_v1.body_binary_forecast import fit as binary_fit
        dataset=RUN/'body_prefix_pilot_fk_v1_data/samples.npz';binary_dataset=RUN/'body_feedback_data_v1/samples.npz'
        if all_goal_half:
            assert object=='event' and version!='v1'
            from research_selective_repair_v1.body_feedback_data import build as feedback_build
            from research_selective_repair_v1.body_prefix_data import build as prefix_build
            feedback_build('body_halfgoal_feedback_v1',expanded=True,family_limit=8)
            binary_dataset=RUN/'body_halfgoal_feedback_v1/samples.npz'
            prefix_build('body_halfgoal_prefix_v1',binary_dataset,RUN/'public_panda_canonical_v1.npz',expanded=True,family_limit=8)
            dataset=RUN/'body_halfgoal_prefix_v1/samples.npz'
            receipt['scope']='1152completed TRAIN feedback,all3goals,6fit/2held families; no DEV or16family formal claim'
            write(out/'receipt.json',receipt)
        if object=='event':
            from research_selective_repair_v1.check_prefix_recovery import compare
            import torch
            full=name+'_recovery_full';partial=name+'_recovery_split'
            fit(full,dataset,seed=206015,steps=120,threads=1)
            fit(partial,dataset,seed=206015,steps=120,stop_after=60,threads=1)
            fit(partial,dataset,seed=206015,steps=120,resume=True,threads=1)
            compare(torch.load(RUN/full/'last.pt',map_location='cpu',weights_only=False),torch.load(RUN/partial/'last.pt',map_location='cpu',weights_only=False))
            receipt['exact_event_checkpoint_optimizer_RNG_stream_recovery']=True
        for kind in ('recurrent','nonrecurrent'):
            kwargs={} if object=='event' else dict(conditioning=conditioning)
            prefix='event' if object=='event' else 'prefix'
            fit('body_%s_pilot_%s_seed0_%s'%(prefix,kind,version),dataset,kind=kind,threads=1,**kwargs)
        binary_name='body_binary_pilot_seed0_'+(version if all_goal_half else 'v1');binary=RUN/binary_name
        if (binary/'last.pt').exists():
            settings=read(binary/'SUMMARY.json')['settings']
            assert settings['seed']==0 and settings['steps']==2400 and settings['threads']==1 and settings['dataset_sha256']==sha(binary_dataset)
            receipt['reused_exact_binary_checkpoint_sha256']=sha(binary/'last.pt')
        else:binary_fit(binary_name,binary_dataset,threads=1)
        code=0
    except BaseException as error:
        receipt['error']=repr(error);receipt['traceback']=traceback.format_exc();print(receipt['traceback'],flush=True)
    finally:
        receipt.update(status='completed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tic,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'receipt.json',receipt);auxiliary.unlink()
    print({k:v for k,v in receipt.items() if k!='source_sha256'},flush=True);raise SystemExit(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--version',default='v1');p.add_argument('--conditioning',type=int,choices=[0,4],default=0);p.add_argument('--object',choices=['state','event'],default='state');p.add_argument('--all-goal-half',action='store_true');run(**vars(p.parse_args()))
