"""Compare saved ordinary pools only; no training, forward or source-label reads."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'routeset').is_dir() and (p/'reports').is_dir())
NEW=ROOT/'reports/observed_two_row_prefix44_convergence_v1'
OLD=ROOT/'reports/observed_two_row_prefix44_v1'
FIELDS=('examples','parents','candidate_matched_ADE_m','candidate_endpoint_error_m','semantic_goal_accuracy',
 'TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount',
 'DuplicateClassifiedTipValidCount','KnownReferenceTypeCoverageAtK','TipClearAtK','StartCorrectAtK','EventSequenceCorrectAtK')

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pool(path):
 rows=read(path/'per_scene.json');metrics=read(path/'metrics.json');counts=Counter();types=Counter()
 for row in rows:
  for item in row['tip_candidates']:
   semantic=bool(item['semantic_goal_correct']);clear=bool(item['tip_segments_clear'])
   counts['candidate_slots']+=1
   counts['semantic_fail']+=not semantic;counts['tip_collision']+=not clear
   counts['both_semantic_and_collision_fail']+=not semantic and not clear
   counts['semantic_fail_clear']+=not semantic and clear
   counts['semantic_correct_collision']+=semantic and not clear
   counts['semantic_correct_clear']+=semantic and clear
   counts['tip_valid']+=bool(item['TipValid']);counts['classified_tip_valid']+=bool(item['classified_tip_valid'])
   counts['tip_valid_unknown_type']+=bool(item['TipValid'] and not item['classified_tip_valid'])
   counts['start_fail']+=not bool(item['starts_at_current_state'])
   counts['event_fail']+=not bool(item['event_state_sequence_correct'])
   if item['classified_tip_valid']:types[json.dumps(item['declared_passage_type'],sort_keys=True)]+=1
 return dict(metrics={k:metrics[k] for k in FIELDS},failure_counts=dict(counts),classified_types=dict(types),
  scene_ids=sorted(r['scene_id'] for r in rows),metrics_sha256=sha(path/'metrics.json'),per_scene_sha256=sha(path/'per_scene.json'))

def main():
 newsummary=read(NEW/'peak_seed0/summary.json');oldsummary=read(OLD/'training/peak_seed0/summary.json')
 layout={'best_train':('peak_seed0/train','training/peak_seed0/train'),
  'best_dev':('peak_seed0/dev_model','training/peak_seed0/dev_model'),
  'last_dev':('peak_seed0/last_dev_model','training/peak_seed0/last_dev_model'),
  'last_train':('fixed_last_train','last_train/analysis/last_train')}
 pools={}
 for key,(newpath,oldpath) in layout.items():
  a=pool(NEW/newpath);b=pool(OLD/oldpath)
  assert a['scene_ids']==b['scene_ids'],key+' changed evaluation identity'
  assert a['failure_counts']['candidate_slots']==4*a['metrics']['examples']
  pools[key]={'original1500':b,'convergence6000':a}
 statuses={s:read(NEW/(s+'.status.json')) for s in ('stage1500','finish')}
 process_seconds={s:(datetime.fromisoformat(r['end_utc'])-datetime.fromisoformat(r['start_utc'])).total_seconds() for s,r in statuses.items()}
 receipt=read(NEW/'convergence_result_receipt.json');events=[json.loads(s) for s in (NEW/'runtime_events.jsonl').read_text().splitlines()]
 result=dict(protocol='saved_pool_convergence_comparison_v1',same_data_candidate_information_recipe=True,
  comparison_has_different_optimization_and_dev_selection_budget=True,original_steps=1500,new_steps=6000,
  original_best_step=oldsummary['best_step'],new_best_step=newsummary['best_step'],
  training_inputs=93,registered_training_inputs=96,unavailable_training_inputs=3,dev_inputs=36,candidates_per_request=4,
  pools=pools,stage1500_exact=read(NEW/'stage1500_audit.json'),
  process_seconds=process_seconds,total_process_seconds=sum(process_seconds.values()),
  total_reserved_gpu_hours=sum(process_seconds.values())/3600,driver_events=events,
  fixed_last_train_diagnostic=receipt['fixed_last_train_diagnostic'],
  original_last_train_saturation_loss=read(OLD/'last_train/analysis/report.json')['stages']['last_train']['original_saturation_loss'],
  new_last_train_saturation_loss=None,new_loss_reason='This extra fixed-last diagnostic saves paths and evaluation metrics; no new matching-loss forward was requested.',
  old_final_logged_total_loss=read(OLD/'training/peak_seed0/history.json')[-1]['loss'],
  new_final_logged_total_loss=read(NEW/'peak_seed0/history.json')[-1]['loss'],
  cost_receipt_sha256=sha(NEW/'convergence_result_receipt.json'),old_summary_sha256=sha(OLD/'training/peak_seed0/summary.json'),
  new_summary_sha256=sha(NEW/'peak_seed0/summary.json'),
  geometry_scope='Tip-only box/semantic checks, not robot-arm execution validity; classified reference sets remain incomplete.',
  new_forward_requests=0,new_optimizer_updates=0,raw_source_labels_opened=False)
 (NEW/'SAVED_POOL_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
 compact={k:{method:{f:v['metrics'][f] for f in ('semantic_goal_accuracy','TipValidAtK','UniqueClassifiedTipValidAtK','KnownReferenceTypeCoverageAtK','candidate_matched_ADE_m')} for method,v in row.items()} for k,row in pools.items()}
 print(json.dumps(dict(original_best_step=result['original_best_step'],new_best_step=result['new_best_step'],pools=compact,cost_seconds=process_seconds)))

if __name__=='__main__':main()
