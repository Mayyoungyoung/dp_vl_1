import argparse
import json
from research_realized_coverage_v1.core import RUN,read,write

def main(output):
    rows={}
    for f in sorted(RUN.glob('*/eval_adaptive/metrics.json')):
        d=read(f);rows[f.parent.parent.name]={k:d[k] for k in ('raw','selected','condition_hit','cost')}
    jobs=[read(f) for f in (RUN/'jobs').glob('*/receipt.json')]
    report=dict(models=rows,jobs=len(jobs),spent_seconds=sum(d.get('elapsed_seconds',0) for d in jobs),
        active=[d['id'] for d in jobs if d['status'] in ('starting','running')],failed=[d['id'] for d in jobs if d['status']=='failed'])
    write(RUN/output,report)
    for k,d in rows.items():print(k,*(round(d[group][key],5) for group,key in [('raw','distinct'),('raw','valid_fraction'),('selected','distinct'),('selected','valid_fraction')]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='PROGRESS.json');a=p.parse_args();main(a.output)
