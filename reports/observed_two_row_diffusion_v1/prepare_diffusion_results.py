"""Local, zero-forward derivative report/figures and unapplied MAIN candidates."""
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMILY = Path(__file__).resolve().parent
BINARIES = ROOT/'runs/synced_observed_two_row_diffusion_v1'
ORDINARY = ROOT/'reports/observed_two_row_composite108_v1'
PARENTS = ['two_row_reach_%d'%i for i in range(283264,283276)]
IDS = [p+'_target%d'%t for p in PARENTS for t in range(3)]
ARMS = ('independent','set')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify_sources():
    index=read(FAMILY/'REMOTE_ARTIFACT_INDEX.json'); checked=[]
    for row in index['files']:
        if row['disposition']=='remote_only': continue
        root=BINARIES if row['disposition']=='ignored_local_npz' else FAMILY
        path=root/row['local_path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Original archive bytes differ: '+str(path))
        checked.append(dict(path=str(path.relative_to(ROOT)),sha256=row['sha256'],bytes=row['bytes']))
    if len(checked)!=848: raise ValueError('Expected all528 originals and320 ignored NPZ')
    return checked


def pool(arm,stage):
    if arm=='ordinary':
        rel='dev_model' if stage=='best' else 'last_dev_model'
        meta=ORDINARY/'peak_seed0'/rel; array_path=meta/'predictions.npz'
        receipt=read(ORDINARY/'peak_seed0/composite_training_receipt.json')
        for name in ('predictions.npz','per_scene.json'):
            if sha(meta/name)!=receipt['prediction_artifacts'][rel+'/'+name]['sha256']: raise ValueError('Ordinary pool seal differs')
    else:
        step=6000 if stage=='best' else 12000
        rel=arm+'/dev/step%05d'%step
        meta=FAMILY/rel; array_path=BINARIES/rel/'predictions.npz'
    rows=read(meta/'per_scene.json'); by_id={r['scene_id']:r for r in rows}
    with np.load(array_path,allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in ('paths','gripper_open','scene_ids','parent_ids')}
    if list(arrays['scene_ids'].astype(str))!=IDS or set(by_id)!=set(IDS) or arrays['paths'].shape!=(36,4,24,3):
        raise ValueError('All36 same oldDEV/K4 required')
    for pos,identifier in enumerate(IDS):
        row=by_id[identifier]
        if row['parent_id']!=identifier.rsplit('_target',1)[0] or len(row['tip_candidates'])!=4: raise ValueError('Parent/candidate budget mismatch')
        for k,decision in enumerate(row['tip_candidates']):
            if decision['candidate']!=k or bool(np.isfinite(arrays['paths'][pos,k]).all())!=decision['finite_xyz']: raise ValueError('Slot identity/finite decision differs')
        valid=[c for c in row['tip_candidates'] if c['TipValid']]
        if row['tip_evaluation']['TipValidAtK']!=len(valid)/4: raise ValueError('Saved validity count mismatch')
    return dict(rows=by_id,arrays=arrays,array_sha256=sha(array_path),metadata_sha256=sha(meta/'per_scene.json'))


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image,ImageDraw
    out=FAMILY/'visualizations'; out.mkdir(exist_ok=False)
    pools={(arm,stage):pool(arm,stage) for arm in ('ordinary',)+ARMS for stage in ('best','last')}
    colors=['#24689a','#cf6e19','#42904d','#a74697']; image_records=[]
    labels=[('ordinary','best'),('independent','best'),('set','best'),('ordinary','last'),('independent','last'),('set','last')]
    for parent in PARENTS:
        ids=[parent+'_target%d'%t for t in range(3)]; positions=[IDS.index(i) for i in ids]
        combined=np.concatenate([v['arrays']['paths'][positions].reshape(-1,3) for v in pools.values()])
        finite=combined[np.isfinite(combined).all(1)]
        lo,hi=finite.min(0),finite.max(0); pad=np.maximum((hi-lo)*.05,.025); lo-=pad;hi+=pad
        fig,axes=plt.subplots(6,6,figsize=(22,18))
        for row_index,(arm,stage) in enumerate(labels):
            item=pools[arm,stage]
            for target,(identifier,pos) in enumerate(zip(ids,positions)):
                decisions=item['rows'][identifier]['tip_candidates']
                stats=item['rows'][identifier]['tip_evaluation']
                for projection,coord in enumerate((1,2)):
                    ax=axes[row_index,target*2+projection]
                    for k,path in enumerate(item['arrays']['paths'][pos]):
                        if not np.isfinite(path).all():
                            ax.text(.02,.05+.08*k,'Nonfinite slot %d'%k,transform=ax.transAxes,color=colors[k],fontsize=7);continue
                        valid=decisions[k]['TipValid']
                        ax.plot(path[:,0],path[:,coord],color=colors[k],ls='-' if valid else '--',lw=1.05,alpha=.85)
                        ax.scatter(path[0,0],path[0,coord],marker='s',s=8,color='black',zorder=4)
                        ax.scatter(path[-1,0],path[-1,coord],marker='o' if valid else 'x',s=22,color=colors[k],zorder=5)
                    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[coord],hi[coord]);ax.grid(alpha=.17)
                    ax.tick_params(labelsize=6)
                    ax.set_title('%s %s | target%d %s\nTip %d/4; known %d; unknown %d; dup %d'%(arm,stage,target,'XY' if coord==1 else 'XZ',
                        round(stats['TipValidAtK']*4),stats['UniqueClassifiedTipValidAtK'],stats['UnknownTypeTipValidCount'],stats['DuplicateClassifiedTipValidCount']),fontsize=7)
                    ax.set_xlabel('X (m)',fontsize=7);ax.set_ylabel(('Y' if coord==1 else 'Z')+' (m)',fontsize=7)
        legend=[Line2D([0],[0],color=colors[k],label='candidate%d'%k) for k in range(4)]
        fig.legend(handles=legend,loc='lower center',ncol=4,fontsize=9)
        fig.suptitle(parent+' | all3 targets, ordinary / independent / set, original best6000 / fixed last12000\n'
            'Repeat0 only, exact K4 each. Solid + circle = saved TipValid; dashed + cross = failed; square = start.\n'
            'Same per-parent XYZ limits across arms/stages. No filtering, checker rerun, raw geometry, or reference overlay.',fontsize=12)
        fig.tight_layout(rect=(0,.028,1,.945));path=out/(parent+'.png');fig.savefig(path,dpi=125);plt.close(fig)
        image_records.append(dict(parent_id=parent,path=str(path.relative_to(FAMILY)),sha256=sha(path),conditions=3,
            submitted_paths=3*2*3*4,projections=2,raw_data_opened=False))
    for page in range(3):
        canvas=Image.new('RGB',(1500,1320),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,6),'All parents page %d/3; thumbnails only, use full figures for path inspection'%(page+1),fill='black')
        for slot,record in enumerate(image_records[page*4:page*4+4]):
            image=Image.open(FAMILY/record['path']).convert('RGB');image.thumbnail((745,635))
            x=(slot%2)*750;y=25+(slot//2)*645;canvas.paste(image,(x,y));draw.text((x+6,y+620),record['parent_id'],fill='black')
        path=out/('contact_sheet_%d.png'%(page+1));canvas.save(path)
        image_records.append(dict(path=str(path.relative_to(FAMILY)),sha256=sha(path),scope='four consecutive parents thumbnail page'))
    write(out/'figure_receipt.json',dict(script_sha256=sha(__file__),pools={a+'/'+s:{k:v for k,v in p.items() if k.endswith('sha256')} for (a,s),p in pools.items()},
        images=image_records,new_forward_calls=0,new_checker_calls=0,raw_data_reads=0,selection='All12 oldDEV parents, all3 goals, all4 slots, repeat0.'))
    return image_records


def main_candidates(analysis):
    out=ROOT/'.bootstrap/diffusion_main_candidate_v1';out.mkdir(exist_ok=False)
    json_path,csv_path=ROOT/'reports/MAIN_RESULTS.json',ROOT/'reports/MAIN_RESULTS.csv'
    original_json_sha,original_csv_sha=sha(json_path),sha(csv_path)
    old=read(json_path)
    with csv_path.open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream);old_fields=reader.fieldnames;old_csv=list(reader)
    if len(old)!=277 or len(old_csv)!=277: raise ValueError('Expected277 unchanged historical results')
    added=[]
    for arm in ARMS:
        summary=read(FAMILY/arm/'summary.json'); costs=analysis['costs'][arm]
        for stage in ('best','last'):
            for repeat in range(3):
                metrics=analysis['repeated_metrics'][arm][stage]['per_repeat'][str(repeat)]
                row={k:None for k in old_fields};charge_train=stage=='best' and repeat==0;charge_repeat=stage=='best' and repeat>0
                timing=analysis['cached_sampler_timing'][arm][stage][str(repeat)]['seconds']['cached_sampler_seconds']
                body=costs['training_body_seconds'] if charge_train else (costs['separate_stage_body_seconds']['repeat%d'%repeat] if charge_repeat else None)
                outer=costs['outer_seconds_by_stage']['train'] if charge_train else (costs['outer_seconds_by_stage']['repeat%d'%repeat] if charge_repeat else None)
                row.update(tier='observed_RGBD_language_two_row_derived',task='three_goal_two_row_narrow_ID',protocol='observed_two_row_tip_eval_v1',
                    method='ordinary_Qwen_RGBD_%s_x0_diffusion12000_%s'%(arm,stage),seed=0,split='DEV_MODEL',K=4,
                    run_id='observed_two_row_diffusion_v1_fixed_receipt/%s/%s_repeat%d'%(arm,stage,repeat),
                    TipValidAtK=metrics['TipValidAtK'],AnyTipValidAtK=metrics['AnyTipValidAtK'],UniqueClassifiedTipValidAtK=metrics['UniqueClassifiedTipValidAtK'],
                    KnownReferenceTypeCoverageAtK=metrics['KnownReferenceTypeCoverageAtK'],TipClearAtK=metrics['TipClearAtK'],
                    UnknownTypeTipValidCount=metrics['UnknownTypeTipValidCount'],DuplicateClassifiedTipValidCount=metrics['DuplicateClassifiedTipValidCount'],
                    semantic_goal_accuracy=metrics['semantic_goal_accuracy'],candidate_ADE_m=metrics['candidate_matched_ADE_m'],reference_ADE_m=metrics['reference_matched_ADE_m'],
                    event_sequence_accuracy=metrics['EventSequenceCorrectAtK'],reference_evaluation_examples=36,semantic_evaluation_examples=36,
                    training_exposures=1536000 if charge_train else 0,training_threads=1,selected_step=6000 if stage=='best' else 12000,
                    grounding_weight=.02,forward_passes=40,elapsed_s=body,gpu_hours=body/3600 if body is not None else 0,
                    anchor_mode='straight_through_peak',checkpoint_selection_protocol='Original48 reused DEV scores UniqueClassifiedTipValid + .05*TipValid; earliest tie; repeats do not reselect',
                    sampling_repeats=1,budget_status='This row is one K4 sampling repetition, not a new training seed or merged K12 pool. All144 slots retained.',
                    generated_complete_path_states=4,path_updates=40,complete_path_state_exposures=1536000 if charge_train else 0,
                    metric_aggregation='all36 oldDEV instructions /12 parents /144 K4 slots; single sampling repeat',
                    training_run_id='observed_two_row_diffusion_v1_fixed_receipt/'+arm,training_gpu_hours=costs['training_body_seconds']/3600 if charge_train else 0,
                    candidate_slots_requested_total=144,candidate_slots_charged_total=144,format_or_budget_failure_slots_total=0,
                    code_commit=analysis['training_source'],source='reports/observed_two_row_diffusion_v1/analysis_run/analysis/analysis.json',
                    shared_encoding_id='observation_two_row_composite108_v1/qwen_cache',new_qwen_encoding_requests=0,
                    peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes'],registered_train_parents=96,observed_train_parents=95,
                    actual_train_inputs=285,unavailable_train_inputs=3,dev_checkpoint_selection_opportunities=48,
                    job_outer_wall_seconds=outer,job_outer_gpu_reserved_hours=outer/3600 if outer is not None else 0,
                    all_arm_job_outer_gpu_hours=costs['outer_gpu_hours_reserved'] if charge_train else 0,
                    trainable_parameter_count=analysis['initialization'][arm]['audit']['active_parameters'],positive_train_reference_count=1663,
                    fixed_last_train_diagnostic_forward_requests=285 if charge_train else 0,
                    fixed_last_train_diagnostic_job_seconds=costs['outer_seconds_by_stage']['fixed-last-train'] if charge_train else 0,
                    fixed_last_train_diagnostic_body_seconds=costs['separate_stage_body_seconds']['fixed-last-train'] if charge_train else 0,
                    fixed_last_train_diagnostic_path_states=1140 if charge_train else 0,
                    additional_observation_draws=384000 if charge_train else 0,optimizer_updates=12000 if charge_train else 0,
                    dev_evaluation_path_states=6912 if charge_train else (288 if charge_repeat else 0),
                    selected_checkpoint_sha256=summary['best_checkpoint_sha256' if stage=='best' else 'last_checkpoint_sha256'],
                    sampling_repeat_index=repeat,sampling_training_seed_count=1,denoising_calls_per_request=40,geometry_encoder_calls_per_request=1,
                    cached_sampler_median_ms=timing['median']*1000,cached_sampler_p95_ms=timing['p95']*1000,
                    sampled_positive_target_slots=1536000 if charge_train else 0,
                    all_positive_reference_accesses=analysis['stream']['all_positive_pool_visits'] if charge_train else 0,
                    teacher_diagnostic_outer_seconds=costs['outer_seconds_by_stage']['denoising-diagnostic'] if charge_train else 0,
                    teacher_diagnostic_denoiser_calls=30 if charge_train else 0,
                    training_geometry_forwards=12000 if charge_train else 0,training_denoiser_forwards=12000 if charge_train else 0,
                    dev_geometry_forwards=1728 if charge_train else (72 if charge_repeat else 0),dev_denoiser_forwards=69120 if charge_train else (2880 if charge_repeat else 0),
                    cost_scope='Fresh ordinary diffusion baseline; both arms same384000 observation draws/1536000 sampled positive targets/48 selections. Shared27 initialization tensors, not identical head architecture or active parameter count. Diffusion samples4 positives while ordinary matching reads all positives; not equal FLOPs/time. Costs on best-repeat0 charge training once; best-repeat1/2 charge their whole best+last repeat stages; matching last rows charge zero. all_arm_job_outer_gpu_hours is a subtotal including diagnostics/repeats, not additive to stage costs. Body intervals nested in outer.40 denoise states include final state; K4 final candidates. Cached sampler timing excludes Qwen and complete online processing; online fields null. Tip-only task checks do not certify arm/execution.0f startup and analysis environment failure remain separately reported.')
                added.append(row)
    new=old+added; write(out/'MAIN_RESULTS.json',new);write(out/'ADDED_ROWS.json',added)
    fields=old_fields+[k for row in added for k in row if k not in old_fields]
    fields=list(dict.fromkeys(fields))
    with (out/'MAIN_RESULTS.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(old_csv)
        writer.writerows({k:('' if v is None else v) for k,v in row.items()} for row in added)
    if read(out/'MAIN_RESULTS.json')[:277] != old: raise ValueError('Historical JSON objects changed')
    with (out/'MAIN_RESULTS.csv').open(newline='',encoding='utf-8') as stream: rebuilt=list(csv.DictReader(stream))
    if any({k:r[k] for k in old_fields}!=old_csv[i] for i,r in enumerate(rebuilt[:277])): raise ValueError('Historical CSV cells changed')
    if sha(json_path)!=original_json_sha or sha(csv_path)!=original_csv_sha: raise ValueError('MAIN changed while candidate prepared')
    write(out/'PRESERVATION_RECEIPT.json',dict(old_rows=277,added_rows=12,new_rows=289,
        original_json_sha256=original_json_sha,original_csv_sha256=original_csv_sha,
        candidate_json_sha256=sha(out/'MAIN_RESULTS.json'),candidate_csv_sha256=sha(out/'MAIN_RESULTS.csv'),
        all_original_json_objects_unchanged=True,all_original_csv_cells_unchanged=True,global_main_written=False,
        new_columns=[k for k in fields if k not in old_fields],analysis_sha256=sha(FAMILY/'analysis_run/analysis/analysis.json'),script_sha256=sha(__file__)))


def quantitative_summary(analysis):
    result=dict(repeats=analysis['repeated_metrics'],teacher={a:analysis['teacher_diagnostic'][a]['by_t'] for a in ARMS},
        train={g:{a:p[side] for a,side in zip(ARMS,('left','right'))} for g,p in analysis['paired_fixed_last_train'].items()},
        costs=analysis['costs'],stream={k:v for k,v in analysis['stream'].items() if k not in ('identity','reference_frequencies','positive_reference_counts','scene_ids','parent_ids')},
        ordinary={stage:analysis['ordinary_control_dev']['independent'][stage]['0']['left'] for stage in ('best','last')},
        ordinary_train=analysis['ordinary_control_fixed_last_train']['independent']['all285']['left'],
        new_forward_calls=0,training_seeds_per_arm=1,pools_merged=False)
    write(FAMILY/'QUANTITATIVE_SUMMARY.json',result)


def main():
    checked=verify_sources(); analysis=read(FAMILY/'analysis_run/analysis/analysis.json')
    if not all(analysis['verification'][k] for k in ('all_ten_stages_completed','same_actual_training_stream','same_initial_shared_tensors','repeats_evaluated_separately')):
        raise ValueError('Sealed analysis did not verify paired lineage')
    images=figures(); quantitative_summary(analysis);main_candidates(analysis)
    write(FAMILY/'LOCAL_DERIVATION_RECEIPT.json',dict(original_files_verified=len(checked),original_file_records=checked,
        original_remote_only=16,images=images,analysis_sha256=sha(FAMILY/'analysis_run/analysis/analysis.json'),script_sha256=sha(__file__),
        new_forward_calls=0,new_checker_calls=0,raw_observations_opened=0,reserved_files_opened=0,global_main_written=False))


if __name__=='__main__': main()
