"""One-thread CPU-only implementation check with explicit source receipts."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,unittest
from pathlib import Path

def run(name,build_prefix=False):
    source=Path(__file__).resolve().parents[1];root=Path('/home/wzy/dpvlm/route_set_v1')
    out=root/'runs/selective_repair_v1/technical'/name;out.mkdir(parents=True,exist_ok=False)
    receipt=dict(source_commit=source.name,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=sys.argv,affinity=sorted(os.sched_getaffinity(0)),threads=1,
        scope='Short CPU-only structural tests,not learner fitting or controller evaluation',status='running')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));tic=time.monotonic()
    assert receipt['affinity']==[3] and os.environ['CUDA_VISIBLE_DEVICES']=='1'
    uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid=='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab'
    import torch
    torch.set_num_threads(1)
    # Tests instantiate CPU models only; no CUDA tensor or context is needed.
    assert not torch.cuda.is_initialized()
    if build_prefix:
        from research_selective_repair_v1.body_prefix_data import build
        r=root/'runs/selective_repair_v1'
        build(name+'_data',r/'body_feedback_data_v1/samples.npz',r/'public_panda_canonical_v1.npz')
        result=None
    else:
        suite=unittest.TestSuite()
        for pattern in ('test_body_prefix*_v1.py','test_public_kinematics_v1.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(source/'tests'),pattern=pattern))
        with (out/'stdout.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    imported={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        p=Path(file)
        try:key=str(p.relative_to(source))
        except ValueError:continue
        if p.is_file():imported[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    passed=result is None or result.wasSuccessful()
    receipt.update(status='completed' if passed else 'failed',tests=0 if result is None else result.testsRun,
        failures=0 if result is None else len(result.failures),errors=0 if result is None else len(result.errors),source_sha256=imported,
        torch_version=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-tic,
        end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),locked_access=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt,flush=True)
    raise SystemExit(0 if passed else 1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--build-prefix',action='store_true');run(**vars(p.parse_args()))
