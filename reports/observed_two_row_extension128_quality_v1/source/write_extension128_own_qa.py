import json,hashlib,datetime
from pathlib import Path
r=Path('F:/dpvlm/reports/observed_two_row_extension128_quality_v1')
inventory=json.loads((r/'VISUAL_QA_PENDING.json').read_text(encoding='utf-8'))
notes={
64:'Target2 slot0 is an accepted unknown large high arc; complete failed and accepted slots remain visible.',
65:'Target2 slot8 is an accepted over/positive_y large lateral loop extending far left; other shorter routes remain visible.',
66:'Target2 slots1/4/5 are red failures beside accepted unknown high arcs; no omitted panels.',
67:'Target1 only slots2/8 are accepted and both unknown; the seven failed slots remain visible.',
68:'Several target1/2 red partial trajectories stop before the goal; all requested panels remain.',
69:'Target0 first five slots are failed; target2 accepted routes include high arcs/loops.',
70:'Target1 slot0 and target2 slot6 have large failed loops; target2 slots3/8 are accepted unknown. Some original axis/title text overlaps under wide extents.',
71:'Target0/1 slot4 accepted over/middle; target2 retains many failures and later accepted unknown loops.',
72:'Target2 slot5 accepted unknown has a broad high arc; target0 slot8 large failed loop is retained.',
73:'Target1 slot8 accepted over/positive_y has a very large leftward circle; some original subplot text overlaps under common wide axes.',
74:'Target2 slots3/7 large failed loops reach high z; target1 slot3 accepted unknown is also a high loop.',
75:'Target1 slots4 and7 both accepted over/middle but are geometrically very different (short local lift versus large lateral loop). The classified duplicate is not a duplicate image or identical trajectory.',
76:'Target0 first three slots failed; accepted target2 slot6 has repeated high lobes. All27 slots remain visible.',
77:'Target2 accepted unknown low and lifted variants coexist with failed slots0/3/5.',
78:'Target0 exactly four accepted slots4/5/6/7 are unknown; target1 slot7 accepted over/middle is a large circle.',
79:'Target1 slot0 accepted unknown climbs high; slot2 broad loop failed; target2 red partial failures retained.',
80:'Target2 failed partial trajectories0/1/3/4 coexist with valid over/positive_y and unknown variants.',
81:'Target1 only slot6 failed; target2 many failed partial paths remain plotted with separate goal stars.',
82:'Target0 all nine are accepted including six unknown variants; target2 failure slots1/2/8 remain.',
83:'Target2 all nine accepted with varied unknown curves; target1 slot2 is a large failed loop, not omitted.',
84:'Target2 failed slots3/8 terminate short of goal while accepted unknown loop slot5 remains.',
85:'Target1 slot2 accepted unknown contains a large multi-loop high excursion; target0 slot4 is a short failed partial.',
86:'Target0 only slots3/7/8 accepted; target2 slot1 accepted unknown loops high around the scene.',
87:'Target0 slot6 is the accepted unknown 6.6218m maximum-length large loop; adjacent slot7 a similar large failed loop. Target1 only slots1/5/8 accepted. Original wide-axis text overlaps slightly.',
88:'Target0 slot3 tangled loop failed; target1 accepted unknown slot3 high loop remains and all failure slots retained.',
89:'Target0 only slots1/3/7 accepted; target2 slot5 is a partial failed trajectory beyond its apparent lateral approach.',
90:'Target0/1 accepted unknown high excursions and failed partials coexist; target2 slot7 failed tangled curve retained.',
91:'Target1 slot6 and target0 slot4 large loops failed; target2 accepted unknown slot5 high loop retained.',
92:'Target2 slots0/4/6 are accepted unknown beside five red failures; target0 known middle and positive routes distinguishable.',
93:'Target2 slot1 huge red loop failed; target0 slot6 green high arc accepted unknown. All27 slots visible.',
94:'Target2 all nine accepted with known over/negative_y and over/positive_y as well as unknown; target0/1 numerous failures retained.',
95:'Target0 first three failed; target2 accepted unknown and known routes coexist with failed slots3/6.'}
rows=[]
for item in inventory['parent_sheets']:
    n=int(item['parent'].rsplit('_',1)[1])-400000
    if n not in notes:continue
    p=r/item['path']; assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    rows.append(dict(index=n,parent=item['parent'],path=item['path'],sha256=item['sha256'],viewed=True,
      tool='view_image(detail=original)',all_three_target_plots_inspected=True,all_27_slot_xy_xz_pairs_present=True,
      front_inspected=True,missing_or_corrupt_panels_observed=False,observations=notes[n]))
assert len(rows)==32
fronts=[]
for page in (1,2):
    p=r/'qa_views'/f'new64_fronts_original_page{page}.png'
    fronts.append(dict(path=p.relative_to(r).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),viewed=True,
      tool='view_image(detail=original)',parent_indices=list(range(64+(page-1)*16,64+page*16)),
      observations='All16 labeled fronts present at original224px. Canonical arm, four posts, and three colored targets visible; no blank/corrupt frame observed. This is visual inspection, not a physical-state or collision proof.'))
out=dict(protocol='extension128_actual_visual_review_constraint_update_v1',reviewer='/root/constraint_update',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='completed_for_assigned_scope',
 assigned_indices=list(range(64,96)),parents=rows,front_pages=fronts,unique_route_plots=96,unique_requested_slots=864,
 all_assigned_images_actually_viewed=True,other_indices_not_claimed=list(range(96,128)),
 general_observation='Accepted/failed/unknown paths and partial failures remain visible. Long accepted loops are retained. Some common-axis original plots have small axis/title overlaps; full-resolution originals are archived.',
 claims_not_supported=['continuous whole-arm safety','complete solution count','model learnability','method advantage'],
 new_forward=0,new_search=0,new_simulation=0,new_reserved_raw_reads=0)
p=r/'VISUAL_QA_CONSTRAINT_UPDATE_64_95.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),parents=len(rows),front_pages=len(fronts))))
