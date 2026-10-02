"""Supplemental read-only counts from already audited TRAIN instructions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/wzy/dpvlm/route_set_v1')
RUN=ROOT/'runs/observed_two_row_formal_train16_analysis_v1'
audit=json.loads((RUN/'analysis/analysis.json').read_text())
corpus=Path(audit['source_corpus']);rows=[];hashes={}
for index in range(16):
    parent='two_row_reach_%d'%(283200+index)
    path=corpus/'parents'/'TRAIN'/parent/'observations.jsonl'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==audit['source_files_sha256'][parent]['observations.jsonl']
    hashes[str(path)]=digest
    for row in [json.loads(line) for line in path.read_text().splitlines()]:
        assert row['parent_id']==parent and row['split']=='TRAIN'
        prefix='Move the gripper to touch the ';suffix=' sphere while avoiding the gray posts.'
        assert row['instruction'].startswith(prefix) and row['instruction'].endswith(suffix)
        rows.append(dict(id=row['id'],parent_id=parent,instruction=row['instruction'],
            color=row['instruction'][len(prefix):-len(suffix)]))
assert len(rows)==48 and len({r['id'] for r in rows})==48
registration_path=corpus/'registration.json'
assert hashlib.sha256(registration_path.read_bytes()).hexdigest()==audit['source_registration_sha256']
palette=json.loads(registration_path.read_text())['specification']['color_table']
names=[row['name'] for row in palette]
counts=Counter(r['color'] for r in rows)
assert set(counts)<=set(names)
result=dict(protocol='fixed_first16_train_actual_instruction_color_counts_v1',source_analysis_sha256=hashlib.sha256((RUN/'analysis/analysis.json').read_bytes()).hexdigest(),
    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_files_sha256=hashes,
    requested_conditions=48,recorded_conditions=len(rows),official_palette_size=len(names),distinct_train_colors=len(counts),
    color_condition_counts={name:counts[name] for name in sorted(names)},missing_train_colors=sorted(set(names)-set(counts)),conditions=rows,
    dev_or_higher_instructions_read=False,selection_changed=False,
    limitation='Counts describe recorded TRAIN language only, not generalization or grounding quality. No rare-color resampling.')
output=RUN/'train_instruction_color_audit.json'
assert not output.exists()
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('distinct_train_colors','official_palette_size','color_condition_counts','missing_train_colors')}))
