"""Archived TRAIN64 quality only; no new data/model/planner execution."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
from PIL import Image,ImageDraw
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file());P=ROOT/'reports/observed_two_row_extension64_quality_v1';OLD=ROOT/'reports/observed_two_row_extension288_quality_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(P/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
a=read(P/'analysis_run/train64/analysis.json');slots=read(P/'analysis_run/train64/all_requested_slots.json')
oldchecks=[]
for i in range(32):
 pid='two_row_reach_'+str(400000+i)
 for f in (OLD/'analysis_run/train32'/pid).glob('*.png'):
  new=P/'analysis_run/train64'/pid/f.name
  oldchecks.append(dict(parent=pid,file=f.name,old_sha256=sha(f),new_sha256=sha(new),equal=f.read_bytes()==new.read_bytes()))
assert len(oldchecks)==128 and all(x['equal'] for x in oldchecks)
def group(lo,hi):
 ss=[x for x in slots if lo<=int(x['parent_id'].split('_')[-1])-400000<hi]
 cc=[x for x in a['conditions'] if lo<=int(x['parent_id'].split('_')[-1])-400000<hi]
 accepted=[x for x in ss if x['status']=='accepted'];failed=[x for x in ss if x['status']=='failed']
 lengths=np.array([x['recomputed_fields']['length_m'] for x in accepted]);ends=np.array([x['recomputed_fields']['endpoint_error_m'] for x in accepted])
 colors=collections.Counter()
 for i in range(lo,hi):
  f=P/'corpus/parents'/('two_row_reach_'+str(400000+i))/'observations.jsonl'
  for line in f.read_text().splitlines():
   row=json.loads(line);colors[row['instruction']]+=1
 unknown=[x for x in accepted if x['actual_accepted_type'] is None]
 return dict(parents=hi-lo,conditions=len(cc),slots=len(ss),accepted=len(accepted),failed=len(failed),
  accepted_rate=len(accepted)/len(ss),known=len(accepted)-len(unknown),unknown=len(unknown),
  R_gt4=sum(x['has_more_than_K4_known_types'] for x in cc),R_histogram=dict(collections.Counter(x['distinct_known_types'] for x in cc)),
  zero_known_positive_conditions=[dict(id=x['id'],positive_references=x['valid_references']) for x in cc if x['distinct_known_types']==0],
  min_refs=min(x['valid_references'] for x in cc),max_refs=max(x['valid_references'] for x in cc),
  length_m=dict(min=float(lengths.min()),median=float(np.median(lengths)),mean=float(lengths.mean()),p90=float(np.percentile(lengths,90)),p95=float(np.percentile(lengths,95)),max=float(lengths.max()),gt2=int(sum(lengths>2)),gt3=int(sum(lengths>3)),gt4=int(sum(lengths>4))),
  longest=sorted([dict(id=x['input_id'],attempt=x['attempt'],length=x['recomputed_fields']['length_m'],trace_sha256=x['trace_sha256']) for x in accepted],key=lambda x:x['length'],reverse=True)[:5],
  endpoint_error_m=dict(median=float(np.median(ends)),max=float(ends.max())),
  failure_errors=dict(collections.Counter(x['error'] for x in failed)),
  failure_predicates_nonexclusive=dict(robot_collision=sum(x['collision_pair'] is not None for x in failed),raw_tip=sum(x.get('recomputed_fields',{}).get('tip_polyline_clear') is False for x in failed),h24_tip=sum(x.get('recomputed_fields',{}).get('tip_polyline_24_clear') is False for x in failed),type_changed=sum(x.get('recomputed_fields',{}).get('actual_route_type')!=x.get('recomputed_fields',{}).get('h24_route_type') for x in failed),endpoint_gt3cm=sum(x.get('recomputed_fields',{}).get('endpoint_error_m',0)>.03 for x in failed)),
  strict_restore_passes=sum(x['strict_restore_passed'] for x in ss),event_transition_counts=dict(collections.Counter(x.get('event_transitions') for x in ss)),
  explicit_get_path_calls=sum(x['planning_calls'] for x in ss),planning_seconds=sum(x['planning_seconds'] for x in ss),simulation_seconds=sum(x['simulation_seconds'] for x in ss),
  worker_seconds=sum(x['closure']['worker_elapsed_seconds'] for x in a['parents'] if lo<=x['index']<hi),
  instruction_histogram=dict(sorted(colors.items())))
groups={name:group(lo,hi) for name,lo,hi in [('old32',0,32),('added32',32,64),('all64',0,64)]}
write('TRAIN64_SUPPLEMENT.json',dict(groups=groups,old32_images_byte_checks=oldchecks,
 old32_visual_qa_reference='../observed_two_row_extension288_quality_v1/condition_quality_and_visual_qa.json',
 analysis_sha256=sha(P/'analysis_run/train64/analysis.json'),new_forward=0,new_search=0,new_simulation=0))
(P/'qa_views').mkdir(exist_ok=True)
front=Image.new('RGB',(8*244,4*250),'white');draw=ImageDraw.Draw(front)
for j,i in enumerate(range(32,64)):
 pid='two_row_reach_'+str(400000+i);x=(j%8)*244;y=(j//8)*250
 front.paste(Image.open(P/'analysis_run/train64'/pid/'front.png').convert('RGB'),(x+10,y+20));draw.text((x+10,y+4),pid,fill='black')
front.save(P/'qa_views/new32_fronts_original_resolution.png')
for page in range(8):
 sheet=Image.new('RGB',(2049,4*349),'white');draw=ImageDraw.Draw(sheet)
 for row in range(4):
  i=32+page*4+row;pid='two_row_reach_'+str(400000+i)
  for target in range(3):
   path=P/'analysis_run/train64'/pid/f'target{target}_all9.png';im=Image.open(path).convert('RGB');im.thumbnail((683,325))
   x=683*target;y=349*row;sheet.paste(im,(x,y+24));draw.text((x+8,y+5),f'{pid} target{target} ALL 9 SLOTS',fill='black')
 sheet.save(P/'qa_views'/f'new32_all96_page{page+1}.png')
print(json.dumps(groups,ensure_ascii=False,indent=2))
