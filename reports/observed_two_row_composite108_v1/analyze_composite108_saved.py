"""Zero-forward local saved-pool comparison with the original constant64 control."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def verify_family(root,new=False):
    receipt=read(root/'peak_seed0'/('composite_training_receipt.json' if new else 'two_row_driver_receipt.json'))
    summary=root/'peak_seed0/summary.json'
    key='summary_sha256' if new else 'actual_training_summary_sha256'
    if sha(summary)!=receipt[key]:raise ValueError('Training summary seal differs')
    for relative,record in receipt['prediction_artifacts'].items():
        if sha(root/'peak_seed0'/relative)!=record['sha256']:raise ValueError('Training pool SHA differs: '+relative)
    diagnostic=read(root/'fixed_last_train/diagnostic_receipt.json')
    for name,value in diagnostic['artifact_sha256'].items():
        if Path(name).name!=name or sha(root/'fixed_last_train'/name)!=value:raise ValueError('Fixed-last pool SHA differs')
    return receipt,diagnostic


def load_pool(folder):
    rows=read(folder/'per_scene.json');by_id={r['scene_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Repeated condition')
    with np.load(folder/'predictions.npz',allow_pickle=False) as a:
        arrays={k:a[k].copy() for k in ('paths','gripper_open','scene_ids','parent_ids')}
    ids=list(map(str,arrays['scene_ids']))
    if set(ids)!=set(by_id) or len(ids)!=len(rows) or arrays['paths'].shape!=(len(rows),4,24,3):raise ValueError('Saved pool identity/budget differs')
    rows=[by_id[i] for i in ids]
    for pos,row in enumerate(rows):
        cs=row['tip_candidates'];t=row['tip_evaluation']
        if len(cs)!=4 or [c['candidate'] for c in cs]!=list(range(4)):raise ValueError('Candidate denominator changed')
        for k,c in enumerate(cs):
            if bool(np.isfinite(arrays['paths'][pos,k]).all())!=c['finite_xyz']:raise ValueError('Saved finite-path flag mismatch')
            if bool(np.isfinite(arrays['gripper_open'][pos,k]).all())!=c['finite_event_values']:raise ValueError('Saved finite-event flag mismatch')
            valid=all(c[key] for key in ('finite_xyz','finite_event_values','semantic_goal_correct','starts_at_current_state','tip_segments_clear','event_state_sequence_correct'))
            if valid!=c['TipValid']:raise ValueError('Saved tip rule differs')
        valid=[c for c in cs if c['TipValid']];known=[tuple(c['declared_passage_type']) for c in valid if c['declared_passage_type'] is not None]
        expected=dict(TipValidAtK=len(valid)/4,AnyTipValidAtK=float(bool(valid)),
            UniqueClassifiedTipValidAtK=len(set(known)),DuplicateClassifiedTipValidCount=len(known)-len(set(known)),
            UnknownTypeTipValidCount=sum(c['declared_passage_type'] is None for c in valid))
        if any(t[k]!=value for k,value in expected.items()):raise ValueError('Saved classified/unknown count differs')
    return rows,arrays


def aggregate(rows):
    candidates=[c for row in rows for c in row['tip_candidates']];n=len(rows)
    if not n:raise ValueError('Empty requested comparison pool')
    cells=Counter(('semantic_correct' if c['semantic_goal_correct'] else 'wrong_goal')+'__'+
                  ('tip_clear' if c['tip_segments_clear'] else 'collision_or_invalid') for c in candidates)
    metric=lambda name:mean([r['tip_evaluation'].get(name) for r in rows])
    result=dict(conditions=n,parents=len({r['parent_id'] for r in rows}),candidate_slots=4*n,
        tip_valid_candidates=sum(c['TipValid'] for c in candidates),semantic_correct_candidates=sum(c['semantic_goal_correct'] for c in candidates),
        tip_clear_candidates=sum(c['tip_segments_clear'] for c in candidates),
        any_tip_valid_conditions=sum(r['tip_evaluation']['AnyTipValidAtK'] for r in rows),
        classified_unique_total=sum(r['tip_evaluation']['UniqueClassifiedTipValidAtK'] for r in rows),
        valid_unknown_total=sum(r['tip_evaluation']['UnknownTypeTipValidCount'] for r in rows),
        classified_duplicate_total=sum(r['tip_evaluation']['DuplicateClassifiedTipValidCount'] for r in rows),
        TipValidAtK=metric('TipValidAtK'),AnyTipValidAtK=metric('AnyTipValidAtK'),
        UniqueClassifiedTipValidAtK=metric('UniqueClassifiedTipValidAtK'),
        semantic_goal_accuracy=metric('semantic_goal_accuracy'),TipClearAtK=metric('TipClearAtK'),
        KnownReferenceTypeCoverageAtK=metric('KnownReferenceTypeCoverageAtK'),
        KnownReferenceTypeCoverage_denominator=sum(r['tip_evaluation']['KnownReferenceTypeCoverageAtK'] is not None for r in rows),
        candidate_matched_ADE_m=mean([r['candidate_matched_ADE_m'] for r in rows]),
        reference_matched_ADE_m=mean([r['reference_matched_ADE_m'] for r in rows]),
        matched_ADE_conditions=sum(r['candidate_matched_ADE_m'] is not None for r in rows),
        saved_positive_reference_count=sum(r['reference_count'] for r in rows),
        no_positive_reference_conditions=sum(r['reference_count']==0 for r in rows),
        semantic_collision_four_cells=dict(cells))
    result['classified_valid_type_counts']=dict(Counter('/'.join(c['declared_passage_type']) for c in candidates
        if c['TipValid'] and c['declared_passage_type'] is not None))
    return result


def paired(old,new):
    a={r['scene_id']:r for r in old};b={r['scene_id']:r for r in new}
    if set(a)!=set(b):raise ValueError('Paired conditions differ')
    keys=('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','semantic_goal_accuracy','TipClearAtK')
    per_condition=[]
    for ident in sorted(a):
        if a[ident]['reference_count']!=b[ident]['reference_count']:raise ValueError('Shared positive reference count changed')
        per_condition.append(dict(scene_id=ident,parent_id=a[ident]['parent_id'],
            delta={k:b[ident]['tip_evaluation'][k]-a[ident]['tip_evaluation'][k] for k in keys}))
    parents=[]
    for parent in sorted({r['parent_id'] for r in per_condition}):
        items=[r for r in per_condition if r['parent_id']==parent]
        parents.append(dict(parent_id=parent,conditions=len(items),mean_delta={k:mean([r['delta'][k] for r in items]) for k in keys}))
    return dict(old=aggregate(old),new=aggregate(new),per_condition=per_condition,per_parent=parents,
        condition_wins_ties_losses={k:dict(wins=sum(r['delta'][k]>0 for r in per_condition),
            ties=sum(r['delta'][k]==0 for r in per_condition),losses=sum(r['delta'][k]<0 for r in per_condition)) for k in keys})


def analyze(old,new):
    verify_family(old);verify_family(new,True)
    old_summary=read(old/'peak_seed0/summary.json');new_summary=read(new/'peak_seed0/summary.json')
    old_cfg=read(old/'peak_seed0/config.json');new_cfg=read(new/'peak_seed0/config.json')
    same_fields=('steps','batch_size','candidates','horizon','width','depth','point_width','lr','seed','eval_every',
        'pooling','pixel_stride','event_scale','anchor_mode','endpoint_mode','endpoint_residual_bound','grounding_weight',
        'grounding_sigma','grounding_target','sampling_mode','refinement_mode','objective','checkpoint_selection')
    if any(old_cfg[k]!=new_cfg[k] for k in same_fields):raise ValueError('Ordinary control configuration differs')
    for summary in (old_summary,new_summary):
        if (summary['last_step'],summary['trajectory_exposures'],summary['sample_stream_audit']['observation_draws'])!=(12000,1536000,384000):raise ValueError('Actual matched budget differs')
    for key in ('initial_model_sha256','initial_torch_cpu_rng_sha256','initial_sampler_state_sha256'):
        if old_summary['sample_stream_audit'][key]!=new_summary['sample_stream_audit'][key]:raise ValueError('Actual initialization differs')
    histories={name:read(root/'peak_seed0/history.json') for name,root in [('old64',old),('composite96',new)]}
    if any([r['step'] for r in h]!=list(range(250,12001,250)) for h in histories.values()):raise ValueError('48 original selection opportunities required')
    pools={}
    for key,relative in [('best_dev','peak_seed0/dev_model'),('last_dev','peak_seed0/last_dev_model'),
                         ('best_train','peak_seed0/train'),('last_train','fixed_last_train')]:
        pools[key]={name:load_pool(root/relative) for name,root in [('old64',old),('composite96',new)]}
    comparisons={key:paired(pools[key]['old64'][0],pools[key]['composite96'][0]) for key in ('best_dev','last_dev')}
    train={}
    for key in ('best_train','last_train'):
        original=pools[key]['old64'][0];original_ids={r['scene_id'] for r in original}
        all_new=pools[key]['composite96'][0];shared=[r for r in all_new if r['scene_id'] in original_ids]
        added=[r for r in all_new if r['scene_id'] not in original_ids]
        if (len(original),len(shared),len(added))!=(189,189,96):raise ValueError('Actual common189/new96 TRAIN groups differ')
        if any(not r['parent_id'].startswith('two_row_reach_400') for r in added):raise ValueError('Unexpected additional TRAIN parent')
        train[key]=dict(common_old189=paired(original,shared),added_new96=aggregate(added),all285=aggregate(all_new))
    result=dict(protocol='composite108_saved_pool_pairing_v1',control=str(old),new=str(new),
        new_forward_requests=0,new_qwen_encodings=0,new_route_searches=0,raw_label_reads=0,
        same_configuration={k:new_cfg[k] for k in same_fields},
        initialization={k:new_summary['sample_stream_audit'][k] for k in new_summary['sample_stream_audit'] if k.startswith('initial_')},
        dev=comparisons,train=train,
        costs={name:dict(best_step=s['best_step'],last_step=s['last_step'],elapsed_seconds=s['elapsed_s'],
            gpu_hours_reserved=s['gpu_hours_reserved'],data_load_preprocess_seconds=s['data_load_preprocess_s'],
            peak_cuda_memory_mb=s['peak_cuda_memory_mb'],parameters=s['parameters'],training_candidate_path_states=s['trajectory_exposures'],
            observation_draws=s['sample_stream_audit']['observation_draws'],actual_index_chain_sha256=s['sample_stream_audit']['index_chain_sha256'])
            for name,s in [('old64',old_summary),('composite96',new_summary)]},
        exposure=dict(old_train_eligible=old_cfg['train_examples'],new_train_eligible=new_cfg['train_examples'],
            old_average_draws_per_eligible_input=384000/old_cfg['train_examples'],new_average_draws_per_eligible_input=384000/new_cfg['train_examples'],
            matched_total_budget=True,matched_per_parent_or_per_input_budget=False),
        interpretation='Single seed data-size control on reused DEV. Saved checker decisions are recomputed arithmetically, not new geometric evaluation. Unknown valid routes remain valid; no new method claim.')
    return result,histories,pools


def plots(out,histories,pools):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'old64':'#69717c','composite96':'#c33a28'}
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for ax,key in zip(axes,('TipValidAtK','semantic_goal_accuracy','UniqueClassifiedTipValidAtK')):
        for name,h in histories.items():ax.plot([r['step'] for r in h],[r['dev_model'][key] for r in h],label=name,color=colors[name])
        ax.set(title=key,xlabel='Training step');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Same 12000 steps / 48 DEV selections; reused development pool')
    fig.tight_layout();fig.savefig(out/'history_comparison.png',dpi=160);plt.close(fig)
    # Every one of the 12 DEV parents, all 3 targets, both selected-best and final.
    # No geometry or reference labels are opened for these saved-prediction plots.
    parent_ids=sorted({r['parent_id'] for r in pools['best_dev']['old64'][0]})
    for parent in parent_ids:
        fig,axes=plt.subplots(6,2,figsize=(10,19));legend_done=False
        for stage_index,stage in enumerate(('best_dev','last_dev')):
            for name,style in [('old64','--'),('composite96','-')]:
                rows,arrays=pools[stage][name];positions={str(i):n for n,i in enumerate(arrays['scene_ids'])};lookup={r['scene_id']:r for r in rows}
                for target in range(3):
                    ident=parent+'_target%d'%target;pos=positions[ident];row=lookup[ident]
                    for column,(a,b,label) in enumerate([(0,1,'XY'),(0,2,'XZ')]):
                        ax=axes[stage_index*3+target,column]
                        for k,path in enumerate(arrays['paths'][pos]):
                            ax.plot(path[:,a],path[:,b],style,color=colors[name],alpha=.68,lw=1.1,
                                    label=name if k==0 else None)
                            good=row['tip_candidates'][k]['TipValid']
                            ax.scatter(path[-1,a],path[-1,b],s=18,marker='o' if good else 'x',color=colors[name])
                        ax.set(title=f'{stage} target{target} {label}',xlabel='X (m)',ylabel=label[1]+' (m)');ax.grid(alpha=.2)
                        if stage_index==0 and target==0 and column==0:ax.legend(fontsize=8)
        fig.suptitle(parent+' — all K4 saved predictions\nEndpoints: circle=tip-valid, cross=failed. No raw geometry/reference overlay.',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/(parent+'.png'),dpi=125);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,required=True);parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh saved-pool analysis output required')
    result,histories,pools=analyze(args.old,args.new);args.output.mkdir(parents=True)
    write(args.output/'SAVED_POOL_COMPARISON.json',result)
    plots(args.output,histories,pools)
    write(args.output/'artifact_index.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in args.output.iterdir() if p.is_file()})
    print(json.dumps(dict(output=str(args.output),new_forward_requests=0,dev_parents=12,common_train_inputs=189,new_train_inputs=96)))
