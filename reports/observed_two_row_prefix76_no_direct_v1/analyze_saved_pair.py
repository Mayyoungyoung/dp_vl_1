"""Compare sealed constant/no_direct pools and plot all DEV parents; zero inference.

The only raw geometric labels opened are the12 existing DEV visualization assets
already archived for the native baseline. No collector, model or search is run.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'routeset').is_dir())
NEW=ROOT/'reports/observed_two_row_prefix76_no_direct_v1'
OLD=ROOT/'reports/observed_two_row_prefix76_convergence_v1'
LAYOUT={'best_train':'peak_seed0/train','last_train':'fixed_last_train',
        'best_dev':'peak_seed0/dev_model','last_dev':'peak_seed0/last_dev_model'}
FIELDS=('examples','parents','candidate_matched_ADE_m','candidate_endpoint_error_m','semantic_goal_accuracy',
 'TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount',
 'DuplicateClassifiedTipValidCount','KnownReferenceTypeCoverageAtK','TipClearAtK','StartCorrectAtK','EventSequenceCorrectAtK')

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def pool(path):
 rows=read(path/'per_scene.json');metrics=read(path/'metrics.json');counts=Counter();types=Counter();endpoint_fail=[]
 with np.load(path/'predictions.npz',allow_pickle=False) as p:arrays={key:p[key].copy() for key in p.files}
 ids=list(map(str,arrays['scene_ids']));by_id={r['scene_id']:r for r in rows}
 assert len(set(ids))==len(ids)==len(rows) and set(ids)==set(by_id)
 assert arrays['paths'].shape==(len(ids),4,24,3) and arrays['gripper_open'].shape==(len(ids),4,24)
 for i,identifier in enumerate(ids):
  row=by_id[identifier];assert row['parent_id']==str(arrays['parent_ids'][i]) and len(row['tip_candidates'])==4
  for item in row['tip_candidates']:
   semantic=bool(item['semantic_goal_correct']);clear=bool(item['tip_segments_clear'])
   counts['candidate_slots']+=1;counts['semantic_fail']+=not semantic;counts['tip_collision']+=not clear
   counts['both_semantic_and_collision_fail']+=not semantic and not clear
   counts['semantic_fail_clear']+=not semantic and clear;counts['semantic_correct_collision']+=semantic and not clear
   counts['tip_valid']+=bool(item['TipValid']);counts['classified_tip_valid']+=bool(item['classified_tip_valid'])
   counts['tip_valid_unknown_type']+=bool(item['TipValid'] and not item['classified_tip_valid'])
   counts['start_fail']+=not bool(item['starts_at_current_state']);counts['event_fail']+=not bool(item['event_state_sequence_correct'])
   if not semantic:endpoint_fail.append(float(item['endpoint_error_m']))
   if item['classified_tip_valid']:types[json.dumps(item['declared_passage_type'])]+=1
 for key in ('TipValidAtK','semantic_goal_accuracy','TipClearAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount','DuplicateClassifiedTipValidCount'):
  assert np.isclose(np.mean([r['tip_evaluation'][key] for r in rows]),metrics[key],rtol=0,atol=1e-12),key
 n=counts['candidate_slots'];assert np.isclose(counts['tip_valid']/n,metrics['TipValidAtK'])
 failures=dict(count=len(endpoint_fail),min_m=min(endpoint_fail) if endpoint_fail else None,max_m=max(endpoint_fail) if endpoint_fail else None,
   over_3_to_4cm=sum(.03<x<=.04 for x in endpoint_fail),over_4_to_6cm=sum(.04<x<=.06 for x in endpoint_fail),over_6cm=sum(x>.06 for x in endpoint_fail))
 public=dict(metrics={k:metrics[k] for k in FIELDS},failure_counts=dict(counts),classified_types=dict(types),endpoint_failure_distances=failures,
   metrics_sha256=sha(path/'metrics.json'),per_scene_sha256=sha(path/'per_scene.json'),predictions_sha256=sha(path/'predictions.npz'),scene_ids=ids)
 return public,by_id,arrays

def compare_rows(left,right,left_np,right_np):
 li={str(x):i for i,x in enumerate(left_np['scene_ids'])};ri={str(x):i for i,x in enumerate(right_np['scene_ids'])}
 assert set(left)==set(right)==set(li)==set(ri)
 results=[]
 for identifier in sorted(left):
  a,b=left[identifier],right[identifier];am,bm=a['tip_evaluation'],b['tip_evaluation']
  results.append(dict(id=identifier,parent_id=a['parent_id'],target_index=int(identifier.rsplit('target',1)[1]),
   constant={key:am[key] for key in ('TipValidAtK','semantic_goal_accuracy','TipClearAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount','DuplicateClassifiedTipValidCount')},
   no_direct={key:bm[key] for key in ('TipValidAtK','semantic_goal_accuracy','TipClearAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount','DuplicateClassifiedTipValidCount')},
   anchor_shift_m=float(np.linalg.norm(left_np['learned_surface_anchor'][li[identifier]]-right_np['learned_surface_anchor'][ri[identifier]]))))
 fields=('TipValidAtK','semantic_goal_accuracy','TipClearAtK','UniqueClassifiedTipValidAtK')
 aggregate={key:dict(improved=sum(r['no_direct'][key]>r['constant'][key] for r in results),same=sum(r['no_direct'][key]==r['constant'][key] for r in results),worse=sum(r['no_direct'][key]<r['constant'][key] for r in results)) for key in fields}
 goals={str(g):{key:dict(constant=float(np.mean([r['constant'][key] for r in results if r['target_index']==g])),no_direct=float(np.mean([r['no_direct'][key] for r in results if r['target_index']==g]))) for key in fields} for g in range(3)}
 return dict(rows=results,condition_changes=aggregate,goal_index_breakdown=goals)

def plot_dev(pools):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from PIL import Image,ImageOps,ImageDraw
 out=NEW/'figures';out.mkdir(exist_ok=False)
 native=ROOT/'reports/observed_two_row_native_astar_v1'
 index=read(native/'SYNC_SHA256_INDEX.json');allowed={row['local_path']:row['sha256'] for row in index['entries']}
 def checked(path):
  key=path.relative_to(ROOT).as_posix();assert allowed[key]==sha(path);return path
 colors=['#0072B2','#D55E00','#009E73','#CC79A7'];manifest=[]
 ids=sorted(pools['no_direct']['best_dev'][1]);parents=sorted({pools['no_direct']['best_dev'][1][i]['parent_id'] for i in ids})
 assert parents==['two_row_reach_%d'%i for i in range(283264,283276)]
 for parent in parents:
  cfg=read(checked(native/'visualization_inputs'/parent/(parent+'.json')))
  geometry_path=checked(ROOT/'runs/observed_two_row_native_astar_v1/visualization_inputs'/parent/'verification_only.npz')
  with np.load(geometry_path) as g:centers=g['obstacle_centers'].copy();halves=g['obstacle_halfsizes'].copy()
  conds=[i for i in ids if pools['no_direct']['best_dev'][1][i]['parent_id']==parent]
  assert len(conds)==3
  points=[np.asarray(cfg['goal_xyz']),centers-halves,centers+halves];lookup={}
  for method in ('constant','no_direct'):
   for stage in ('best_dev','last_dev'):
    arrays=pools[method][stage][2];lookup[method,stage]={str(identifier):arrays['paths'][i] for i,identifier in enumerate(arrays['scene_ids'])}
    points.extend(lookup[method,stage][identifier] .reshape(-1,3) for identifier in conds)
  finite=np.concatenate(points);finite=finite[np.isfinite(finite).all(axis=1)];low=finite.min(axis=0)-.04;high=finite.max(axis=0)+.04
  fig,axes=plt.subplots(3,8,figsize=(26,10.2),squeeze=False)
  summaries=[]
  for row,identifier in enumerate(conds):
   target=int(identifier.rsplit('target',1)[1]);col=0
   for method,stage in (('constant','best_dev'),('no_direct','best_dev'),('constant','last_dev'),('no_direct','last_dev')):
    record=pools[method][stage][1][identifier];metric=record['tip_evaluation'];paths=lookup[method,stage][identifier]
    summaries.append(dict(id=identifier,method=method,stage=stage,tip_valid=metric['TipValidAtK'],unique=metric['UniqueClassifiedTipValidAtK'],unknown=metric['UnknownTypeTipValidCount']))
    for ordinate in (1,2):
     ax=axes[row,col];col+=1
     for c,h in zip(centers,halves):ax.add_patch(Rectangle((c[0]-h[0],c[ordinate]-h[ordinate]),2*h[0],2*h[ordinate],color='.6',alpha=.4))
     for k,path in enumerate(paths):ax.plot(path[:,0],path[:,ordinate],color=colors[k],lw=1.35)
     goals=np.asarray(cfg['goal_xyz']);ax.scatter(goals[:,0],goals[:,ordinate],s=12,c='black');ax.scatter(goals[target,0],goals[target,ordinate],marker='*',s=95,c='red',zorder=4)
     ax.set_title(f"T{target} {method} {stage[:4]}\nV{metric['TipValidAtK']:.2f} U{metric['UniqueClassifiedTipValidAtK']} ?{metric['UnknownTypeTipValidCount']}",fontsize=8)
     ax.set_xlim(low[0],high[0]);ax.set_ylim(low[ordinate],high[ordinate]);ax.grid(alpha=.15);ax.set_xlabel('x (m)',fontsize=8);ax.set_ylabel(('y','z')[ordinate-1]+' (m)',fontsize=8);ax.tick_params(labelsize=7)
  fig.suptitle(parent+' | constant versus no_direct, all3 targets/all4 candidates | no filtering or repair; boxes/targets: evaluation only',fontsize=12)
  fig.tight_layout(rect=(0,0,1,.97));path=out/(parent+'.png');fig.savefig(path,dpi=120);plt.close(fig)
  manifest.append(dict(parent_id=parent,file=path.relative_to(NEW).as_posix(),sha256=sha(path),conditions=summaries,route_config_sha256=sha(native/'visualization_inputs'/parent/(parent+'.json')),geometry_sha256=sha(geometry_path)))
 for page in range(3):
  selected=manifest[page*4:(page+1)*4];thumbs=[]
  for item in selected:
   with Image.open(NEW/item['file']) as im:thumbs.append(ImageOps.contain(im.convert('RGB'),(1920,760)))
  sheet=Image.new('RGB',(1920,sum(im.height for im in thumbs)+30),'white');y=25
  for im in thumbs:sheet.paste(im,(0,y));y+=im.height
  ImageDraw.Draw(sheet).text((10,5),'All12 DEV parents, ordered page%d/3; each row is one parent'%page,fill='black')
  sheet.save(out/('contact_sheet_%d.jpg'%page),quality=95)
 return manifest

def main():
 started=time.perf_counter()
 for root in (NEW,OLD):
  index=read(root/'REMOTE_ARTIFACT_INDEX.json')
  for name,row in index['files'].items():assert sha(root/name)==row['sha256'] and (root/name).stat().st_size==row['bytes']
 pools={method:{stage:pool(root/folder) for stage,folder in LAYOUT.items()} for method,root in (('constant',OLD),('no_direct',NEW))}
 paired={stage:compare_rows(pools['constant'][stage][1],pools['no_direct'][stage][1],pools['constant'][stage][2],pools['no_direct'][stage][2]) for stage in LAYOUT}
 summaries={method:read(root/'peak_seed0/summary.json') for method,root in (('constant',OLD),('no_direct',NEW))}
 status=read(NEW/'train12000.status.json');outer=(datetime.fromisoformat(status['end_utc'])-datetime.fromisoformat(status['start_utc'])).total_seconds()
 result=dict(protocol='same_budget_no_direct_constant_saved_pool_analysis_v1',pools={method:{stage:values[0] for stage,values in row.items()} for method,row in pools.items()},paired=paired,
  same_data_shared_initialization_sample_chain_and_final_rng=read(NEW/'paired_stream_receipt.json'),
  full_model_initialization_equal=False,initial_forward_equal_claim=False,parameter_reduction=549120,capacity_confound=True,
  budgets=dict(training_observation_draws_each=384000,training_candidate_states_each=1536000,dev_selections_each=48,actual_train_inputs=189,requested_train_inputs=192,missing_train_inputs=3,dev_inputs=36),
  best_steps={method:s['best_step'] for method,s in summaries.items()},fixed_last_step=12000,
  cost=dict(no_direct_outer_seconds=outer,no_direct_outer_reserved_gpu_hours=outer/3600,no_direct_driver_runtime=read(NEW/'runtime_events.jsonl'),
   constant_outer_seconds=559.381471,base_seconds={method:s['elapsed_s'] for method,s in summaries.items()},base_gpu_hours={method:s['gpu_hours_reserved'] for method,s in summaries.items()},
   no_direct_fixed_last_train_seconds=read(NEW/'fixed_last_train/diagnostic_receipt.json')['elapsed_seconds'],scope='Nested timings, never add base/driver/process'),
  caveats=['Saved metrics reaggregated; no new neural inference or route generation.','No full-robot validity. Known type references incomplete; unknown remains valid when original checker passed.','No novel method or fixed online latency benefit. Same seed is one paired ordinary structural/capacity control, not multisource confirmation.'],new_forward_requests=0,new_optimizer_updates=0,reserved_raw_opened=False)
 write(NEW/'SAVED_POOL_COMPARISON.json',result)
 figures=plot_dev(pools);write(NEW/'FIGURE_MANIFEST.json',dict(parents=figures,all12=True,all36_conditions=True,all4_candidates_each=True,comparison_models=4,source_script_sha256=sha(Path(__file__)),new_forward_requests=0,reference_curves_not_loaded=True,elapsed_seconds=time.perf_counter()-started))
 print(json.dumps(dict(best_steps=result['best_steps'],cost=result['cost'],pools={method:{stage:value[0]['metrics'] for stage,value in row.items()} for method,row in pools.items()})))

if __name__=='__main__':main()
