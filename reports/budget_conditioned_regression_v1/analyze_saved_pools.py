"""Read the frozen 24 DEV pools only. No Torch, model, planning or repair."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import csv
import datetime
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SYNC=ROOT/'runs/synced_budget_conditioned_regression_v1'
RUN=HERE/'run/seed0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def checker():
    package=types.ModuleType('frozen_budget');package.__path__=[];sys.modules[package.__name__]=package
    for name in ('geometry','multigate'):
        path=HERE/'source/routeset'/(name+'.py.txt')
        spec=importlib.util.spec_from_loader('frozen_budget.'+name,SourceFileLoader('frozen_budget.'+name,str(path)))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return sys.modules['frozen_budget.multigate']


def main():
    start=time.perf_counter(); cpu=time.process_time()
    if os.name=='nt':
        import ctypes
        dll=ctypes.WinDLL('kernel32',use_last_error=True)
        dll.GetCurrentProcess.restype=ctypes.c_void_p
        dll.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        if not dll.SetProcessAffinityMask(dll.GetCurrentProcess(),2):raise OSError('local CPU1 affinity failed')
    else:os.sched_setaffinity(0,{1})
    index=read(HERE/'REMOTE_ARTIFACT_INDEX.json');summary=read(RUN/'summary.json')
    for row in index['files']:
        if 'local_path' in row:assert sha(ROOT/row['local_path'])==row['sha256']
    remote={row['path']:row for row in index['files']}
    for name,expected in summary['artifact_sha256'].items():assert remote['seed0/'+name]['sha256']==expected
    for name,expected in summary['identity']['source_sha256'].items():
        assert sha(HERE/'source'/(name+'.txt'))==expected
    data_path=SYNC/'development.npz';assert sha(data_path)==summary['identity']['dataset_sha256']
    with np.load(data_path,allow_pickle=False) as z:data={key:z[key] for key in z.files}
    assert set(data['splits'])=={'TRAIN','DEV_MODEL'}
    selected=np.flatnonzero(data['splits']=='DEV_MODEL');assert len(selected)==128
    d={key:value[selected] for key,value in data.items()}
    check=checker();R=d['path_mask'].sum(1)
    reference_sets=[set(m[k].tolist()) for m,k in zip(d['modes'],d['path_mask'])]
    assert [len(s) for s in reference_sets]==R.tolist()
    assert all(check.path_validity(p[m],s)['valid'].all() for p,m,s in zip(d['paths'],d['path_mask'],d['scenes']))
    all_pools=[];final={};rechecked_slots=0
    for step in range(500,3001,500):
        for k in (1,2,4,8):
            relative=Path('selection')/('step%04d'%step)/('k%d'%k); folder=RUN/relative
            receipt=read(folder/'receipt.json');saved=read(folder/'per_scene.json');metrics=read(folder/'metrics.json')
            pool=SYNC/'seed0'/relative/'predictions.npz'
            assert sha(pool)==receipt['files_sha256']['predictions.npz']
            with np.load(pool,allow_pickle=False) as z:
                pred=z['paths'];assert np.array_equal(z['parent_ids'],d['parent_ids']) and np.array_equal(z['scene_ids'],d['scene_ids'])
            assert pred.shape==(128,k,24,3)
            actual=check.route_metrics(pred,d['scenes'],d['modes'],d['path_mask']);per=actual.pop('per_scene')
            for key in ('valid','modes','unique_count','collision'):assert np.array_equal(per[key],saved[key]),(step,k,key)
            for key,value in actual.items():
                if value is not None:assert np.isclose(value,metrics[key],rtol=1e-12,atol=1e-12),(step,k,key)
            rows=[]
            for i,parent in enumerate(d['parent_ids']):
                flags=check.path_validity(pred[i],d['scenes'][i]);raw_modes=check.route_modes(pred[i],d['scenes'][i])
                known=set(per['modes'][i][per['modes'][i]>=0].tolist());assert not (known-reference_sets[i])
                unique=len(known);valid=int(per['valid'][i].sum());unknown=int((per['valid'][i]&(per['modes'][i]<0)).sum())
                duplicate=valid-unknown-unique;invalid=k-valid;cap=min(int(R[i]),k);gap=cap-unique
                assert gap>=0
                rows.append(dict(parent_id=str(parent),scene_id=str(d['scene_ids'][i]),step=step,k=k,
                    R_known_positive_types=int(R[i]),reference_types=sorted(reference_sets[i]),
                    oracle_budgeted_known_type_capacity=cap,valid_candidates=valid,invalid_candidates=invalid,
                    known_unique=unique,valid_unknown=unknown,known_duplicates=duplicate,
                    minimum_duplicate_slots_if_all_k_valid_and_R_saturated=max(k-int(R[i]),0),
                    missing_known_types=sorted(reference_sets[i]-known),budgeted_type_gap=gap,
                    invalid_only_oracle_replacement_max_gain=min(gap,invalid),
                    duplicate_only_oracle_replacement_max_gain=min(gap,duplicate),
                    note='replacement gains overlap; existence of stored positives is oracle evidence, not a learned repair',
                    candidate_valid=per['valid'][i].tolist(),candidate_valid_modes=per['modes'][i].tolist(),
                    candidate_raw_modes=raw_modes.tolist(),collision=flags['collision'].tolist(),
                    endpoint_error=flags['endpoint_error'].tolist(),length_ratio=flags['length_ratio'].tolist(),
                    in_bounds=flags['in_bounds'].tolist(),finite=flags['finite'].tolist()))
            aggregate=dict(step=step,k=k,metrics=metrics,known_capacity_sum=sum(r['oracle_budgeted_known_type_capacity'] for r in rows),
                known_unique_sum=sum(r['known_unique'] for r in rows),gap_sum=sum(r['budgeted_type_gap'] for r in rows),
                gap_parents=sum(r['budgeted_type_gap']>0 for r in rows),duplicate_slots=sum(r['known_duplicates'] for r in rows),
                unknown_slots=sum(r['valid_unknown'] for r in rows),invalid_slots=sum(r['invalid_candidates'] for r in rows),
                prediction_sha256=sha(pool),receipt_sha256=sha(folder/'receipt.json'))
            all_pools.append(aggregate);rechecked_slots+=128*k
            if step==3000:final[k]=dict(rows=rows,paths=pred,aggregate=aggregate)
    assert rechecked_slots==11520
    assert summary['final']['best']['step']==summary['final']['last']['step']==3000
    assert summary['final']['best']['pools']==summary['final']['last']['pools']
    k8=final[8]['rows']; byR=[]
    for n in sorted(set(R.tolist())):
        subset=[r for r in k8 if r['R_known_positive_types']==n]
        byR.append(dict(R=n,parents=len(subset),capacity=sum(r['oracle_budgeted_known_type_capacity'] for r in subset),
            unique=sum(r['known_unique'] for r in subset),gap=sum(r['budgeted_type_gap'] for r in subset),
            invalid=sum(r['invalid_candidates'] for r in subset),duplicates=sum(r['known_duplicates'] for r in subset)))
    ledger=[json.loads(line) for line in (RUN/'calls.jsonl').read_text().splitlines()]
    for kind,values in summary['actual_budget'].items():
        entries=[r for r in ledger if r['kind']==kind]
        assert len(entries)==values['events'] and sum(r['requests'] for r in entries)==values['requests'] and sum(r['slots'] for r in entries)==values['slots']
    sessions=[read(p) for p in sorted((RUN/'sessions').glob('*.json'))]
    outer={stage:read(HERE/'run'/(stage+'.status.json')) for stage in ('tests','metadata','pause2','resume')}
    outer_seconds={stage:(datetime.datetime.fromisoformat(v['end_utc'])-datetime.datetime.fromisoformat(v['start_utc'])).total_seconds() for stage,v in outer.items()}
    result=dict(protocol='budget_conditioned_saved_pool_analysis_v1',source_commit=summary['identity']['source_commit'],
        actual_model_calls=0,optimizer_steps=0,generated_candidates=0,search_calls=0,repaired_candidates=0,
        source_summary_sha256=sha(RUN/'summary.json'),development_data_sha256=sha(data_path),
        roles_read=['DEV_MODEL'],source_dataset_roles=['TRAIN','DEV_MODEL'],train_pool_saved=False,
        dev_parents=128,reference_positive_paths_checked=int(R.sum()),original_selection_pools_checked=24,
        original_selection_slots_rechecked=rechecked_slots,best_and_last_share_pool=True,
        final_by_k={str(k):v['aggregate'] for k,v in final.items()},all_selection_pools=all_pools,
        k8_by_R=byR,k8_gap_parent_records=[r for r in k8 if r['budgeted_type_gap']>0],
        counts=dict(k8_capacity_total=sum(min(int(n),8) for n in R),k8_unique_total=sum(r['known_unique'] for r in k8),
                    k8_full_capacity_parents=sum(r['budgeted_type_gap']==0 for r in k8),
                    k8_duplicate_slots=sum(r['known_duplicates'] for r in k8),
                    k8_minimum_duplicates_at_full_valid_saturation=sum(max(8-int(n),0) for n in R)),
        costs=dict(train_selection_seconds=summary['training_and_selection_elapsed_s'],
            inner_session_seconds=sum(v['session_elapsed_s'] for v in sessions),outer_stage_seconds=outer_seconds,
            outer_training_gpu_reserved_hours=(outer_seconds['pause2']+outer_seconds['resume'])/3600,
            inner_session_gpu_reserved_hours=sum(v['session_gpu_reserved_hours'] for v in sessions),
            scope='inner sessions and outer stages are nested alternatives, never add; tests/metadata separate',
            peak_allocated_bytes=summary['peak_memory_allocated_bytes'],peak_reserved_bytes=summary['peak_memory_reserved_bytes']),
        epistemic_scope='known declared opening-pair capacity only; continuous solution total unknown; oracle replacement is not model performance')
    write(HERE/'PER_PARENT_FINAL_POOLS.json',{str(k):v['rows'] for k,v in final.items()})
    # Complete128-parent view rather than selecting only failing examples.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(17,8),sharex=True)
    x=np.arange(128);axes[0].plot(x,np.minimum(R,8),label='known min(R,8)',color='black',lw=1)
    axes[0].scatter(x,[r['known_unique'] for r in k8],s=11,label='K8 predicted known unique');axes[0].set_ylabel('Types');axes[0].legend()
    axes[1].bar(x,[r['known_duplicates'] for r in k8],label='valid same-type duplicates',color='steelblue')
    axes[1].plot(x,np.maximum(8-R,0),label='minimum if K8 all-valid and saturated',color='orange',lw=1);axes[1].legend();axes[1].set_ylabel('Slots')
    axes[2].bar(x,[r['budgeted_type_gap'] for r in k8],label='known capacity gap',color='darkred')
    axes[2].scatter(x,[r['invalid_candidates'] for r in k8],label='invalid slots',s=10,color='black');axes[2].legend();axes[2].set_ylabel('Slots');axes[2].set_xlabel('Original DEV_MODEL parent index (all128)')
    fig.tight_layout();fig.savefig(HERE/'ALL128_K8_CAPACITY.png',dpi=160);plt.close(fig)
    failures=[i for i,r in enumerate(k8) if r['budgeted_type_gap']>0]
    fig,axes=plt.subplots(1,len(failures),figsize=(14,5),squeeze=False)
    from matplotlib.patches import Rectangle
    for ax,i in zip(axes.flat,failures):
        lower,upper=check.wall_boxes(d['scenes'][i])
        for lo,hi in zip(lower,upper):ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],facecolor='.8'))
        for j,path in enumerate(final[8]['paths'][i]):
            valid=k8[i]['candidate_valid'][j];ax.plot(path[:,0],path[:,1],lw=1.2,color='tab:blue' if valid else 'tab:red',alpha=.75)
            ax.text(path[12,0],path[12,1],str(j),fontsize=7)
        for path,mode,keep in zip(d['paths'][i],d['modes'][i],d['path_mask'][i]):
            if keep and int(mode) in k8[i]['missing_known_types']:ax.plot(path[:,0],path[:,1],'--',color='green',alpha=.35,lw=.7)
        ax.set(xlim=(-1,1),ylim=(-1,1),title='%s: R=%d, U=%d, invalid=%d'%(k8[i]['parent_id'],R[i],k8[i]['known_unique'],k8[i]['invalid_candidates']),xlabel='x',ylabel='y')
    fig.suptitle('Original K8 pools: blue valid / red invalid / dashed stored missing positives (no repair)')
    fig.tight_layout();fig.savefig(HERE/'K8_TWO_CAPACITY_GAPS.png',dpi=160);plt.close(fig)
    result['local_analysis_wall_seconds']=time.perf_counter()-start;result['local_analysis_cpu_seconds']=time.process_time()-cpu
    result['analysis_source_sha256']=sha(__file__);write(HERE/'ANALYSIS.json',result)
    print(json.dumps({key:result[key] for key in ('counts','costs','local_analysis_wall_seconds','local_analysis_cpu_seconds')},indent=2))


if __name__=='__main__':main()
