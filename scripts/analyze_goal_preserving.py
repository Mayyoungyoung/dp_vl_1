"""Existing metrics only, on sealed candidate pools; no model selection sweep."""
import json
from pathlib import Path
import numpy as np
from scripts.analyze_segment_clearance import metrics


def analyze(root):
    baseline = Path('reports/segment_clearance_v1/evaluation')
    output = {}
    fields = ['CandidateValidRate','CollisionFailureRate','GoalFailureRate','AnyValid8','GeometricModeCount8','TwoDistinctValid4','ReferenceModeCoverage8','SelectedValid1']
    table = ['| Split / arm / seed | Valid % | Collision % | Target fail % | Any@8 % | Modes@8 | Two@4 % | Ref@8 % | qTop1 % |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    cases = {}
    for split in ('old_dev','dev32'):
        output[split] = {}; cases[split] = {}
        all_rows = {}
        for arm in ('A','B','C','D','E'):
            scores = []
            for seed in range(3):
                file = (baseline if arm in ('A','B') else root/'evaluation')/split/('%s_seed%d'%(arm,seed))/'rows.json'
                if not file.exists(): continue
                rows = json.loads(file.read_text());all_rows[arm, seed] = rows
                value = {k:float(np.mean([metrics(r)[k] for r in rows])) for k in metrics(rows[0])}
                scores.append(value)
                output[split]['%s_seed%d'%(arm,seed)] = value
                cells = ['%s / %s / %d'%(split,arm,seed)]+['%.3f'%(value[k]*(1 if k=='GeometricModeCount8' else 100)) for k in fields]
                table.append('| '+' | '.join(cells)+' |')
            if len(scores)==3:
                value = {k:float(np.mean([v[k] for v in scores])) for k in scores[0]}
                output[split][arm+'_mean3'] = value
                cells = [split+' / '+arm+' / mean3']+['%.3f'%(value[k]*(1 if k=='GeometricModeCount8' else 100)) for k in fields]
                table.append('| '+' | '.join(cells)+' |')
        for (arm, seed), rows in all_rows.items():
            if arm in ('A','B'): continue
            b = {r['id']:r for r in all_rows['B',seed]}
            delta = []
            for r in rows:
                base = b[r['id']]
                value = dict(id=r['id'],parent=r['parent'],before=metrics(base),after=metrics(r),
                             all_endpoint_fail_before=all(not c['semantic_goal_correct'] for c in base['candidates']),
                             all_endpoint_fail_after=all(not c['semantic_goal_correct'] for c in r['candidates']))
                delta.append(value)
            cases[split]['%s_seed%d'%(arm,seed)] = dict(
                all_endpoint_fail_before=sum(d['all_endpoint_fail_before'] for d in delta),
                all_endpoint_fail_after=sum(d['all_endpoint_fail_after'] for d in delta),
                fixed_cases=[d for d in delta if d['id'].endswith(('283268_target1','400269_target0'))],
                largest_valid_loss=min(delta,key=lambda d:d['after']['CandidateValidRate']-d['before']['CandidateValidRate']),
                largest_target_loss=max(delta,key=lambda d:d['after']['GoalFailureRate']-d['before']['GoalFailureRate']))
    (root/'RESULTS.json').write_text(json.dumps(output,indent=2)+'\n')
    (root/'ENDPOINT_CASES.json').write_text(json.dumps(cases,indent=2)+'\n')
    (root/'CORE_TABLE.md').write_text('\n'.join(table)+'\n')
    print('\n'.join(table))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root)
