from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

root=Path('reports/observed_two_row_formal_train16_v1')
a=json.loads((root/'analysis/analysis.json').read_text())
rows=json.loads((root/'analysis/all_432_slots.json').read_text())
accepted=[r for r in rows if r['status']=='accepted']
failed=[r for r in rows if r['status']=='failed' and r.get('error')=="RuntimeError('unchanged endpoint/raw/H24 clearance or type acceptance failed')"]
patterns=Counter()
for r in failed:
    f=r['recomputed_fields'];reasons=[]
    if f['endpoint_error_m']>.03:reasons.append('endpoint')
    if not f['tip_polyline_clear']:reasons.append('raw_tip')
    if not f['tip_polyline_24_clear']:reasons.append('H24_tip')
    if f['actual_route_type']!=f['h24_route_type']:reasons.append('type_change')
    patterns['+'.join(reasons)]+=1
unknown=[r for r in accepted if r['actual_accepted_type'] is None]
relations=Counter()
for r in unknown:
    for relation in {c['passage_relation'] for c in r['actual_row_crossings']}:
        relations[relation]+=1
length=np.asarray([r['recomputed_fields']['length_m'] for r in accepted])
supp=dict(source_analysis_sha256=hashlib.sha256((root/'analysis/analysis.json').read_bytes()).hexdigest(),
    source_slots_sha256=hashlib.sha256((root/'analysis/all_432_slots.json').read_bytes()).hexdigest(),
    analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    accepted_known_over_references=sum(c['known_over_references'] for c in a['conditions']),
    accepted_known_lateral_references=sum(r['actual_accepted_type'] is not None and 'over' not in r['actual_accepted_type'] for r in accepted),
    acceptance_rejection_count=len(failed),acceptance_failure_combination_counts=dict(patterns),
    unknown_actual_crossing_relations_overlapping=dict(relations),
    accepted_length_quantiles_m={str(q):float(np.quantile(length,q)) for q in (0,.25,.5,.75,.9,.95,1)},
    accepted_length_over_2m=int(sum(length>2)),accepted_length_over_3m=int(sum(length>3)),
    registered_acceptance_unchanged=True,references_filtered=0,dev_or_higher_raw_read=False)
(root/'supplemental_statistics.json').write_text(json.dumps(supp,indent=2))
images=[root/'analysis'/c['parent_id']/('target%d_all9.png'%c['target']) for c in a['conditions']]
assert len(images)==48 and all(p.exists() for p in images)
folder=root/'contact_sheets';folder.mkdir(exist_ok=False)
for page in range(8):
    canvas=Image.new('RGB',(2730,2010),'white');draw=ImageDraw.Draw(canvas)
    for cell,path in enumerate(images[page*6:(page+1)*6]):
        with Image.open(path) as im:thumb=ImageOps.contain(im.convert('RGB'),(1365,650))
        x=(cell%2)*1365;y=30+(cell//2)*660
        draw.text((x+8,y-18),path.parent.name+'/'+path.name,fill='black')
        canvas.paste(thumb,(x,y))
    draw.text((8,4),'Fixed TRAIN all-slot contact sheet %d/8; all 48 target figures, no selection'%(page+1),fill='black')
    canvas.save(folder/('page_%02d.png'%(page+1)))
print(json.dumps(supp,indent=2))
