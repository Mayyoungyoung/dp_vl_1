"""Recompute fixed saved pools and diagnose duplicate/coverage opportunity.

No generation, route repair, reference filtering, or new data-role access.
Cross-goal consistency is descriptive, not evidence of an update mechanism.
"""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import time

import numpy as np

from scripts.export_two_row_observations import verify_export, digest, write_json
from scripts.evaluate_observed_two_row import scene_metrics

PROTOCOL='two_row_saved_ordinary_diagnosis_v1'
SCALAR_FIELDS=('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK',
    'UnknownTypeTipValidCount','DuplicateClassifiedTipValidCount','KnownReferenceTypeCoverageAtK',
    'semantic_goal_accuracy','TipClearAtK','EventSequenceCorrectAtK','mean_path_length_m','endpoint_error_m')


def modes(values):
    return {tuple(value) for value in values if value is not None}


def opportunity(candidates, reference_types):
    known=modes(reference_types)
    valid=[tuple(r['declared_passage_type']) for r in candidates if r['classified_tip_valid']]
    unique=set(valid);missing=known-unique;duplicates=len(valid)-len(unique)
    return dict(known_reference_types=sorted(known),valid_predicted_types=sorted(unique),
        missing_known_types=sorted(missing),valid_classified_duplicate_slots=duplicates,
        duplicate_replacement_count_upper_bound=min(duplicates,len(missing)),
        unknown_valid_slots=sum(r['TipValid'] and not r['classified_tip_valid'] for r in candidates),
        nonexclusive_failure_counts={name:sum(not row[name] for row in candidates) for name in (
            'finite_xyz','finite_event_values','semantic_goal_correct','starts_at_current_state',
            'tip_segments_clear','event_state_sequence_correct')},
        interpretation='Count bound for replacing duplicate valid slots using known positives; not achievable gain or exhaustive solution count.')


def cross_goal_consistency(rows):
    parents=defaultdict(list)
    for row in rows:parents[row['parent_id']].append(row)
    pairs=[]
    for parent,conditions in sorted(parents.items()):
        for old,new in itertools.permutations(sorted(conditions,key=lambda r:r['id']),2):
            a,b=old['opportunity'],new['opportunity']
            common=modes(a['known_reference_types'])&modes(b['known_reference_types'])
            old_common=modes(a['valid_predicted_types'])&common
            retained=old_common&modes(b['valid_predicted_types'])
            pairs.append(dict(parent_id=parent,old_id=old['id'],new_id=new['id'],known_common_types=sorted(common),
                old_valid_common_types=sorted(old_common),retained_types=sorted(retained),
                independently_missing_types=sorted(old_common-retained),
                fraction=len(retained)/len(old_common) if old_common else None))
    eligible=[r for r in pairs if r['fraction'] is not None]
    return dict(pairs=pairs,directed_observed_pairs=len(pairs),eligible_pairs=len(eligible),
        mean_known_common_type_consistency=float(np.mean([r['fraction'] for r in eligible])) if eligible else None,
        interpretation='Independently generated pools under two target languages. No update was executed; missing references are not negatives.')


def aggregate(rows,requested_inputs):
    result={name:float(np.mean([r['metrics'][name] for r in rows if r['metrics'][name] is not None]))
        if any(r['metrics'][name] is not None for r in rows) else None for name in SCALAR_FIELDS}
    result.update(observed_inputs=len(rows),requested_inputs=requested_inputs,
        unobserved_requested_inputs=requested_inputs-len(rows),
        generated_candidate_slots=sum(r['metrics']['candidates'] for r in rows),
        duplicate_replacement_count_upper_bound_mean=float(np.mean([r['opportunity']['duplicate_replacement_count_upper_bound'] for r in rows])) if rows else None,
        conditions_with_duplicate_and_missing_known=sum(r['opportunity']['duplicate_replacement_count_upper_bound']>0 for r in rows),
        known_reference_cardinality_histogram=dict(Counter(len(r['opportunity']['known_reference_types']) for r in rows)),
        full_robot_validity=None,missing_input_policy='Not generated; excluded observed-input metrics and retained requested denominator separately.')
    return result


def read_pool(path,labels,expected,checkpoint_digest,recorded_digest):
    if digest(path)!=recorded_digest:raise ValueError('Saved prediction hash differs from training summary')
    with np.load(path,allow_pickle=False) as a:
        ids=a['scene_ids'].astype(str);parents=a['parent_ids'].astype(str)
        paths,events=a['paths'].copy(),a['gripper_open'].copy()
    if (set(ids)!=expected or len(set(ids))!=len(ids) or paths.shape!=(len(ids),4,24,3)
            or events.shape!=(len(ids),4,24) or len(parents)!=len(ids)):
        raise ValueError('One exact registered split with four H24 slots required')
    if any(parent!=labels[identifier]['parent_id'] for identifier,parent in zip(ids,parents)):
        raise ValueError('Saved prediction parent identity mismatch')
    return ids,paths,events,dict(prediction_sha256=recorded_digest,checkpoint_sha256=checkpoint_digest)


def analyze(data_root,run_root,output):
    data_root,run_root,output=map(Path,(data_root,run_root,output))
    if output.exists():raise FileExistsError('Fresh analysis output required')
    started=time.perf_counter();manifest,gate=verify_export(data_root)
    labels={r['id']:r for r in [json.loads(line) for line in (data_root/'supervision.jsonl').read_text().splitlines()]}
    summary=json.loads((run_root/'summary.json').read_text());config=json.loads((run_root/'config.json').read_text())
    receipt_path=run_root/'two_row_driver_receipt.json';receipt=json.loads(receipt_path.read_text())
    if (config.get('two_row_export_sha256')!=digest(data_root/'export_manifest.json')
            or config.get('two_row_driver_protocol')!='ordinary_two_row_frozen_qwen_saturation_v1'):
        raise ValueError('Training must use this frozen ordinary two-row export')
    if (receipt['actual_training_summary_sha256']!=digest(run_root/'summary.json')
            or receipt['export_sha256']!=digest(data_root/'export_manifest.json')):
        raise ValueError('Training receipt does not bind this result/export')
    hashes={str(data_root/'export_manifest.json'):digest(data_root/'export_manifest.json'),
        str(run_root/'summary.json'):digest(run_root/'summary.json'),str(run_root/'config.json'):digest(run_root/'config.json'),
        str(receipt_path):digest(receipt_path)}
    for relative,entry in receipt['prediction_artifacts'].items():
        path=run_root/relative
        if path.resolve()!=Path(entry['path']).resolve() or digest(path)!=entry['sha256']:
            raise ValueError('Frozen prediction artifact differs from driver receipt')
        hashes[str(path)]=entry['sha256']
    def checked(path):
        path=Path(path);actual=digest(path)
        if manifest['source_files_sha256'].get(str(path))!=actual:raise ValueError('Label/reference source changed')
        hashes[str(path)]=actual;return path
    cache={}
    def label_data(identifier):
        if identifier in cache:return cache[identifier]
        label=labels[identifier]
        with np.load(checked(label['observation']),allow_pickle=False) as a:
            current={key:a[key].copy() for key in ('gripper_pose','gripper_open')}
        with np.load(checked(label['verification_only']),allow_pickle=False) as a:
            geometry={key:a[key].copy() for key in ('obstacle_centers','obstacle_halfsizes')}
        cfg=json.loads(checked(label['route_config']).read_text());refs=[]
        for name in label['routes']:
            with np.load(checked(name),allow_pickle=False) as a:refs.append(a['gripper_pose'][:,:3].copy())
        cache[identifier]=(current,geometry,cfg,refs);return cache[identifier]
    groups={};pools={}
    for name,split,weight,pred_key,metric_key in (
            ('best_dev','DEV_MODEL','best.pt','prediction_sha256','metrics'),
            ('last_dev','DEV_MODEL','last.pt','last_prediction_sha256','last_metrics'),
            ('best_train','TRAIN','best.pt',None,'train_metrics')):
        folder={'best_dev':'dev_model','last_dev':'last_dev_model','best_train':'train'}[name]
        file=run_root/folder/'predictions.npz';ckey='best_checkpoint_sha256' if weight=='best.pt' else 'last_checkpoint_sha256'
        if digest(run_root/weight)!=summary[ckey]:raise ValueError('Checkpoint differs from summary')
        expected={identifier for identifier,row in labels.items() if row['split']==split}
        original=receipt['prediction_artifacts'][folder+'/predictions.npz']['sha256']
        if pred_key is not None and original!=summary[pred_key]:raise ValueError('Summary and receipt prediction hashes differ')
        ids,paths,events,provenance=read_pool(file,labels,expected,summary[ckey],original)
        hashes[str(file)]=original;hashes[str(run_root/weight)]=summary[ckey];rows=[]
        for pos,identifier in enumerate(ids):
            label=labels[identifier];current,geometry,cfg,refs=label_data(identifier)
            metrics,candidates=scene_metrics(paths[pos],events[pos],current,geometry,label['semantic_targets'],label['route_types'],cfg)
            rows.append(dict(id=identifier,parent_id=label['parent_id'],split=split,metrics=metrics,candidates=candidates,
                opportunity=opportunity(candidates,label['route_types'])))
        combined=aggregate(rows,manifest['selection']['requested_parents'][split]*3)
        for key in SCALAR_FIELDS:
            if key not in summary[metric_key]:continue
            expected_value=summary[metric_key][key];actual=combined[key]
            if (actual is None)!=(expected_value is None) or actual is not None and not np.isclose(actual,expected_value,rtol=1e-9,atol=1e-9):
                raise ValueError('Independent recomputation differs: '+name+'/'+key)
        groups[name]=dict(split=split,metrics=combined,rows=rows,cross_goal=cross_goal_consistency(rows),provenance=provenance)
        pools[name]={identifier:paths[pos] for pos,identifier in enumerate(ids)}
    output.mkdir(parents=True)
    plots=plot_all_dev(output,groups,pools,labels,cache)
    result=dict(protocol=PROTOCOL,groups=groups,plots=plots,source_files_sha256=hashes,
        live_mechanical_gate=gate,analyzer_sha256=digest(__file__),elapsed_seconds=time.perf_counter()-started,
        new_model_requests=0,new_candidate_states=0,geometry_repair=False,source_model_or_data_changed=False,
        limitation='Observed tip validity only. Cross-goal consistency is not an executed update or evidence of causal mechanism benefit.')
    if any(digest(path)!=value for path,value in hashes.items()):raise ValueError('Analyzed source changed')
    write_json(output/'analysis.json',result)
    write_json(output/'artifact_index.json',{p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps({name:value['metrics'] for name,value in groups.items()}));return result


def plot_all_dev(output,groups,pools,labels,cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colors=['#0072B2','#D55E00','#009E73','#CC79A7'];files=[]
    ids=sorted(pools['best_dev']);parents=sorted({labels[i]['parent_id'] for i in ids})
    lookup={name:{r['id']:r for r in groups[name]['rows']} for name in ('best_dev','last_dev')}
    for parent in parents:
        conditions=[i for i in ids if labels[i]['parent_id']==parent]
        all_points=[]
        for identifier in conditions:
            current,geometry,cfg,refs=cache[identifier]
            all_points.extend(refs+[np.asarray(cfg['goal_xyz']),np.asarray(current['gripper_pose'][:3])[None]])
            for name in ('best_dev','last_dev'):all_points.append(pools[name][identifier].reshape(-1,3))
        finite=np.concatenate(all_points);finite=finite[np.isfinite(finite).all(axis=1)]
        low,high=finite.min(axis=0)-.04,finite.max(axis=0)+.04
        fig,axes=plt.subplots(len(conditions),4,figsize=(17,3.4*len(conditions)),squeeze=False)
        for row,identifier in enumerate(conditions):
            label=labels[identifier];current,geometry,cfg,refs=cache[identifier]
            for column,(name,ordinate) in enumerate(itertools.product(('best_dev','last_dev'),(1,2))):
                ax=axes[row,column]
                for c,h in zip(geometry['obstacle_centers'],geometry['obstacle_halfsizes']):
                    ax.add_patch(Rectangle((c[0]-h[0],c[ordinate]-h[ordinate]),2*h[0],2*h[ordinate],color='.55',alpha=.4))
                for ref in refs:ax.plot(ref[:,0],ref[:,ordinate],color='.7',lw=.6,alpha=.45)
                paths=pools[name][identifier]
                for k,path in enumerate(paths):
                    if np.isfinite(path).all():ax.plot(path[:,0],path[:,ordinate],color=colors[k],lw=1.4)
                goals=np.asarray(cfg['goal_xyz']);target=goals[label['semantic_targets']['target_index']]
                ax.scatter(goals[:,0],goals[:,ordinate],s=15,c='black');ax.scatter(target[0],target[ordinate],marker='*',s=100,c='red',zorder=5)
                metric=lookup[name][identifier]['metrics']
                ax.set_title(f"{identifier.rsplit('_',1)[-1]} | {name} | V={metric['TipValidAtK']:.2f}, U={metric['UniqueClassifiedTipValidAtK']}",fontsize=9)
                ax.set_xlim(low[0],high[0]);ax.set_ylim(low[ordinate],high[ordinate]);ax.grid(alpha=.15)
                ax.set_xlabel('x (m)');ax.set_ylabel(('y','z')[ordinate-1]+' (m)')
        fig.suptitle(parent+' | all observed targets, all four candidates | gray: all accepted raw references',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.97));path=output/(parent+'_all_predictions.png');fig.savefig(path,dpi=120);plt.close(fig)
        files.append(path.name)
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','run','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();analyze(args.data,args.run,args.output)
