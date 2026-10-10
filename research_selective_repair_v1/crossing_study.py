"""Frozen same-feedback direct measure/categorical study and TRAIN diagnostics."""
import argparse,datetime,os,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,read,write,sha

def diagnostics(folder,dataset):
    from research_selective_repair_v1.body_crossing_measure import compose
    from research_selective_repair_v1.event_decomposition import metrics
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    file=RUN/folder/'TRAIN_diagnostic_predictions.npz'
    with np.load(file) as z:
        assert np.array_equal(z['ids'],d['ids']);held=z['heldout'];p={k:z[k] for k in z.files if k not in ('ids','slots','options','families','heldout')}
    base=read(RUN/'constraints_seed0_v1/config.json')['post_base'];prob=np.empty((len(d['ids']),17),np.float32)
    for ident in sorted(set(d['ids'])):
        ix=np.flatnonzero(d['ids']==ident);posts=d['completed'][ix[0]];height=np.clip(posts[:,2]-base,.025,.4)
        cfg=dict(row_x=[float(posts[j:j+2,0].mean()) for j in (0,2)],post_y=[posts[j:j+2,1].tolist() for j in (0,2)],post_heights=[float(height[j:j+2].max()) for j in (0,2)],post_base_z=base,tip_clearance_m=.02)
        prob[ix]=compose({k:v[ix] for k,v in p.items()},d['paths'][ix],cfg)[0]
    report=dict(fit=metrics(prob,d['labels'],~held),held_TRAIN=metrics(prob,d['labels'],held),scope='TRAIN-held families only,full composition and same-feedback category readout',locked_access=False)
    write(RUN/folder/'COMPOSED_DIAGNOSTICS.json',report);print(dict(folder=folder,**report),flush=True)

def run(name,full=False,screen=False):
    out=RUN/'jobs'/name;out.mkdir(parents=True,exist_ok=False)
    lock=RUN/'prefix_pilot_gpu.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    receipt=dict(command=sys.argv,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE.name,
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for directory in ('research_selective_repair_v1','routeset','scripts','tests') for p in (SOURCE/directory).rglob('*') if p.suffix in ('.py','.sh')},
        affinity=sorted(os.sched_getaffinity(0)),threads=1,memory_fraction=.35,status='running',scope='TRAIN only direct crossing measure and same-feedback auxiliary categorical control')
    write(out/'receipt.json',receipt);tic=time.monotonic();code=1
    try:
        assert receipt['affinity']==[3] and os.environ['CUDA_VISIBLE_DEVICES']=='1'
        receipt['gpu_uuid']=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert receipt['gpu_uuid']=='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab'
        for other in ('feasible_space_v1','realized_coverage_v1','mode_geometry_v1'):assert not(ROOT/'runs'/other/'active.lock').exists()
        main=RUN/'active.lock'
        if main.exists():
            args=(Path('/proc')/main.read_text().strip()/'cmdline').read_bytes().decode().split('\0');job=args[args.index('--id')+1]
            active=read(RUN/'jobs'/job/'receipt.json');assert job.startswith('body_execution_train_') and active['status']=='running'
            assert 'execution' in active['command'] and any('render_selective_body_v1.sh' in v for v in active['command'])
            receipt['overlapping_one_thread_CPU_teacher']=job
        import torch,unittest
        torch.set_num_threads(1)
        suite=unittest.defaultTestLoader.discover(str(SOURCE/'tests'),pattern='test_body_prefix*_v1.py')
        result=unittest.TextTestRunner(verbosity=1).run(suite);receipt['unit_tests']=result.testsRun
        assert result.wasSuccessful();assert not torch.cuda.is_initialized();receipt['CPU_tests_before_CUDA']=True
        from research_selective_repair_v1.body_crossing_measure import fit
        from research_selective_repair_v1.check_prefix_recovery import compare
        dataset=RUN/('body_events_data_v1' if full else 'body_halfgoal_events_v2')/'samples.npz'
        manifest=read(dataset.parent/'MANIFEST.json');assert manifest['rows']==(2304 if full else 1152) and manifest['no_DEV_feedback'] and manifest['samples_sha256']==sha(dataset)
        receipt['dataset_sha256']=sha(dataset);receipt['dataset_rows']=manifest['rows'];write(out/'receipt.json',receipt)
        if not screen:
            a=name+'_recovery_full';b=name+'_recovery_split'
            fit(a,dataset,seed=206016,steps=120)
            fit(b,dataset,seed=206016,steps=120,stop_after=60)
            fit(b,dataset,seed=206016,steps=120,resume=True)
            compare(torch.load(RUN/a/'last.pt',map_location='cpu',weights_only=False),torch.load(RUN/b/'last.pt',map_location='cpu',weights_only=False));receipt['exact_checkpoint_optimizer_RNG_recovery']=True
            for kind in ('recurrent','nonrecurrent'):
                for readout in ('analytic','categorical_aux'):
                    for seed in ((0,1,2) if full else (0,)):
                        folder='body_crossing_%s_%s_seed%d_%s_v1'%(kind,readout,seed,'full' if full else 'pilot')
                        fit(folder,dataset,kind=kind,readout=readout,seed=seed);diagnostics(folder,dataset)
        else:
            assert not full,'Formal study screens require explicit matched-head protocol'
            from research_selective_repair_v1.body_screen import screen as generate
            arms=[('actual','actual','recurrent','analytic'),('actual_nonrecurrent','actual','nonrecurrent','analytic'),('categorical','actual','recurrent','categorical_aux'),('categorical_nonrecurrent','actual','nonrecurrent','categorical_aux'),('planned','planned','recurrent','analytic')]
            for label,kind,architecture,readout in arms:
                folder='body_crossing_%s_%s_seed0_pilot_v1'%(architecture,readout)
                generate('body_crossing_%s_TRAIN_screen_seed0_v1'%label,kind,RUN/folder/'last.pt',word_safe=True,forecast='crossing',role='TRAIN',continuous=True,all_goals=True,family_start=6,families=2,dataset_scope='half')
        code=0
    except BaseException as error:receipt['error']=repr(error);receipt['traceback']=traceback.format_exc();print(receipt['traceback'],flush=True)
    finally:
        receipt.update(status='completed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tic,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(out/'receipt.json',receipt);lock.unlink()
    print({k:v for k,v in receipt.items() if k!='source_sha256'},flush=True);raise SystemExit(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--full',action='store_true');p.add_argument('--screen',action='store_true');run(**vars(p.parse_args()))
