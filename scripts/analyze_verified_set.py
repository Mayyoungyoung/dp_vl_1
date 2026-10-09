"""Paired family statistics for completed verified-set runs, DEV only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT, read, write, sha
from scripts.research_v3_frequency import SUPPORT

RUN = ROOT/'runs/verified_set_v1'
BUDGET_RUN = RUN
OLD = ROOT/'runs/research_v3_v1'


def contrast(a, b, key, eligible=None):
    aa, bb = {r['id']:r for r in a}, {r['id']:r for r in b}
    assert aa.keys() == bb.keys()
    groups = {}
    for ident in sorted(aa):
        if eligible is not None and ident not in eligible: continue
        x,y = aa[ident],bb[ident]; assert x['family'] == y['family']
        groups.setdefault(x['family'],[]).append(x['raw'][key]-y['raw'][key])
    values=np.array([np.mean(v) for _,v in sorted(groups.items())])
    if not len(values): raise ValueError('No paired layout families')
    rng=np.random.default_rng(904100)
    bs=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
    return dict(difference=float(values.mean()), family_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),
                families=len(values),requests=sum(map(len,groups.values())))


def analyze(names, output):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    with np.load(SUPPORT/'support.npz') as s:
        eligible={str(s['ids'][i]) for i in range(len(s['ids']))
                  if s['splits'][i]=='DEV_MODEL' and len(set(s['modes'][i,s['mask'][i]]))<=8}
    rows={}; summaries={}; hashes={}; training={}
    for name in ['parent']+names:
        base=OLD/'safety_mean' if name=='parent' else RUN/name
        ev=base/'evaluation_matched_q_v1'
        rows[name]=read(ev/'rows.json'); summaries[name]=read(ev/'metrics.json')
        assert len(rows[name])==288
        for f in ('rows.json','metrics.json','pool.npz'): hashes[str(ev/f)]=sha(ev/f)
        assert sha(base/'last.pt')==summaries[name]['generator_sha256']
        if name!='parent':
            training[name]=read(base/'summary.json')
            assert training[name]['steps']==1200
    assert len({v['scorer_bundle_sha256'] for v in summaries.values()})==1
    by_seed={}
    for name,summary in training.items():
        seed=name.rsplit('_seed',1)[1]
        by_seed.setdefault(seed,[]).append(summary['sampler']['index_chain_sha256'])
    assert all(len(set(v))==1 for v in by_seed.values()),'Actual paired input streams differ'
    assert len({s['initial_sha256'] for s in training.values()})==1
    differences={}
    for name in names:
        controls=['parent']+[n for n in names if n!=name and n.rsplit('_seed',1)[-1]==name.rsplit('_seed',1)[-1]]
        differences[name]={}
        for control in controls:
            differences[name][control]={scope:{k:contrast(rows[name],rows[control],k,keep)
                for k in ('distinct','valid_fraction','recall')}
                for scope,keep in [('all_dev',None),('original_known_modes_le8',eligible)]}
    jobs=[read(f) for f in (BUDGET_RUN/'jobs').glob('*/receipt.json')]
    result=dict(names=names,metrics={k:dict(raw=v['raw'],selected=v['selected'],brier=v['reliability'].get('brier'))
        for k,v in summaries.items()},contrasts=differences,source_hashes=hashes,
        same_initial_parameters=True,paired_actual_sampler_streams=True,
        budget_closed_seconds=sum(r.get('elapsed_seconds',0) for r in jobs),
        job_failures=[r['id'] for r in jobs if r.get('status')=='failed'],
        source_scope='Development-only paired continuation seeds; not independent backbone pretraining',locked_access=False)
    gates={}
    for name in names:
        if not name.startswith(('project_seed','set_point_seed','set_project_seed')):continue
        controls=['parent']+[n for n in names if n!=name and n.startswith(('ordinary_seed','gate_seed','replay_','set_point_seed'))
                               and n.rsplit('_seed',1)[1]==name.rsplit('_seed',1)[1]]
        gates[name]={c:bool(differences[name][c]['all_dev']['distinct']['difference']>=.3
            and differences[name][c]['all_dev']['valid_fraction']['difference']>=-.01) for c in controls}
    result['engineering_gates']=gates
    write(out/'RESULTS.json',result)
    with (out/'SUMMARY.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','valid8','distinct8','known_recall8','selected_distinct4'])
        for n,v in summaries.items():writer.writerow([n,v['raw']['valid_fraction'],v['raw']['distinct'],v['raw']['recall'],v['selected']['distinct']])
    report=['# Verified-set development results','',
        'Same initial model and actual per-seed observation streams verified. Fixed final1200; all288 DEV requests retained.',
        'TRAIN population: '+read(RUN/'prepared/manifest.json')['eligibility']+'.',
        'No TEST_LOCKED access, no robot execution claim, no automatic novelty claim.','',
        '| Arm | Valid@8 | Distinct@8 | Known recall@8 | Selected distinct@4 |','|---|---:|---:|---:|---:|']
    for n,v in summaries.items():
        report.append('|%s|%.4f|%.4f|%.4f|%.4f|'%(n,v['raw']['valid_fraction'],v['raw']['distinct'],v['raw']['recall'],v['selected']['distinct']))
    report+=['','Engineering gates: `'+json.dumps(gates)+'`.','',
             'Family intervals, actual payload/checkpoint hashes, budget and failure records: RESULTS.json.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(result['metrics']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--population',choices=['known_le8','all'],default='known_le8')
    a=p.parse_args()
    if a.population=='all': RUN=RUN/'all_population_v2'
    analyze(a.names,a.output)
