"""Post-queue artifact-only closure. Refuses unfinished jobs; no experiment launch."""
import argparse,datetime,json,os,sys,time
from research_feasible_space_v1.prepare import RUN
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha

def main(name):
    tic=time.monotonic();out=RUN/name
    if out.exists():raise FileExistsError(out)
    assert not (RUN/'active.lock').exists(),'Research job still active'
    receipts={str(f.relative_to(RUN)):read(f) for f in sorted((RUN/'jobs').glob('*/receipt.json'))}
    assert all(v['status'] in ('completed','failed') for v in receipts.values())
    files={str(f.relative_to(RUN)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(RUN.rglob('*')) if f.is_file() and f.name!='active.lock'}
    sources={}
    for r in receipts.values():
        commit=r['source_commit'];folder=ROOT/'research_v2/releases'/commit
        if commit in sources:continue
        for n,h in r['source_sha256'].items():assert sha(folder/n)==h,(commit,n)
        archive=ROOT/'research_v2/incoming'/(commit+'.tar');sources[commit]=dict(archive_sha256=sha(archive),source_files=r['source_sha256'])
    fresh=ROOT/'data/feasible_space_generalization_v1'
    data_files={str(f.relative_to(fresh)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(fresh.rglob('*')) if f.is_file()} if fresh.exists() else {}
    report=dict(files=files,data_files=data_files,data_root=str(fresh),sources=sources,receipt_count=len(receipts),completed=sum(r['status']=='completed' for r in receipts.values()),failed=sum(r['status']=='failed' for r in receipts.values()),
        command_seconds=sum(r['elapsed_seconds'] for r in receipts.values()),locked_access=False,experiment_queue_terminal=True,
        artifact_only_command=sys.argv,artifact_only_elapsed_seconds=time.monotonic()-tic,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(out,report);print(json.dumps({k:v for k,v in report.items() if k not in ('files','sources')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='ARTIFACT_INDEX.json');main(**vars(p.parse_args()))
