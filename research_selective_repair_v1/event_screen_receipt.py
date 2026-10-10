"""Frozen pilot screening while only our CPU TRAIN teacher overlaps GPU work."""
import argparse,datetime,os,sys,time,traceback,subprocess
from pathlib import Path
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,read,write,sha

def run(name,api=False,amplitude=False):
    out=RUN/'jobs'/name;out.mkdir(parents=True,exist_ok=False)
    lock=RUN/'prefix_pilot_gpu.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    receipt=dict(command=sys.argv,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE.name,
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for directory in ('research_selective_repair_v1','routeset','scripts') for p in (SOURCE/directory).rglob('*.py')},
        affinity=sorted(os.sched_getaffinity(0)),threads=1,memory_fraction=.35,status='running',scope='Frozen event pilot DEV_MODEL target0; geometric screening,not actual execution')
    write(out/'receipt.json',receipt);tic=time.monotonic();code=1
    try:
        assert receipt['affinity']==[3] and os.environ['CUDA_VISIBLE_DEVICES']=='1'
        assert subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab'
        for family in ('feasible_space_v1','realized_coverage_v1','mode_geometry_v1'):assert not (ROOT/'runs'/family/'active.lock').exists()
        main=RUN/'active.lock'
        if main.exists():
            args=(Path('/proc')/main.read_text().strip()/'cmdline').read_bytes().decode().split('\0');job=args[args.index('--id')+1]
            active=read(RUN/'jobs'/job/'receipt.json');assert job.startswith('body_execution_train_') and active['status']=='running'
            assert 'execution' in active['command'] and any('render_selective_body_v1.sh' in v for v in active['command'])
            receipt['overlapping_CPU_teacher']=job
        import torch
        torch.set_num_threads(1)
        if api:
            from research_selective_repair_v1.body_deployment import bind,feedback
            decoder='body_event_halfgoal_api_decoder_v1'
            bind(decoder,RUN/'body_event_pilot_recurrent_seed0_v2/last.pt',RUN/'constraints_seed0_v1/last.pt',RUN/'calibrated_prototype_v1/PROTOTYPES.json',forecast='event',word_safe=True)
            feedback('body_event_halfgoal_API_feedback_v1',RUN/decoder/'last.pt',ROOT/'data/selective_repair_interventions_v1',limit=1)
            receipt['scope']='Bounded one TRAIN observation integrated RGB-D API feedback;33query sets,8finals each; not formal heads or actual execution'
            code=0
            return
        from research_selective_repair_v1.body_screen import screen
        if amplitude:
            rnn=RUN/'body_event_pilot_recurrent_seed0_v2/last.pt';mlp=RUN/'body_event_pilot_nonrecurrent_seed0_v2/last.pt';binary=RUN/'body_binary_pilot_seed0_v2/last.pt'
            for label,kind,forecast,checkpoint in [('actual','actual','event',rnn),('planned','planned','event',rnn),('actual_nonrecurrent','actual','event',mlp),('binary_planned','planned','binary',binary)]:
                screen('body_amplitude_%s_TRAIN_screen_seed0_v1'%label,kind,checkpoint,word_safe=True,forecast=forecast,role='TRAIN',continuous=True,all_goals=True,family_start=6,families=2,dataset_scope='half')
            receipt['scope']='Bounded32step amplitude interpolation on two event-held TRAIN families,all3goals;4matched controls; not DEV or actual execution'
            code=0
            return
        rnn=RUN/'body_event_pilot_recurrent_seed0_v1/last.pt';mlp=RUN/'body_event_pilot_nonrecurrent_seed0_v1/last.pt';binary=RUN/'body_binary_pilot_seed0_v1/last.pt'
        arms=[('actual','actual','event',rnn),('coordinate','coordinate','event',rnn),('actual_nonrecurrent','actual','event',mlp),
            ('planned','planned','event',rnn),('success','success','event',rnn),('binary_planned','planned','binary',binary),('binary_success','success','binary',binary)]
        for label,kind,forecast,checkpoint in arms:
            screen('body_event_%s_DEV_screen_seed0_v1'%label,kind,checkpoint,word_safe=True,forecast=forecast)
        code=0
    except BaseException as error:
        receipt['error']=repr(error);receipt['traceback']=traceback.format_exc();print(receipt['traceback'],flush=True)
    finally:
        receipt.update(status='completed' if code==0 else 'failed',exit_code=code,elapsed_seconds=time.monotonic()-tic,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(out/'receipt.json',receipt);lock.unlink()
    print({k:v for k,v in receipt.items() if k!='source_sha256'},flush=True);raise SystemExit(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--api',action='store_true');p.add_argument('--amplitude',action='store_true');run(**vars(p.parse_args()))
