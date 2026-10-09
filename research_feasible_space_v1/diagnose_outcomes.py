"""Independent saved-pool check: region safety is neither task nor mode safety."""
import argparse,json
from collections import Counter
import numpy as np
from research_feasible_space_v1.prepare import RUN
from research_realized_coverage_v1.core import DATA,lines,read,write
from scripts.evaluate_paired_modes import references,check_candidates
from scripts.research_v3_audit import mode

def main(name,names):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'};refs={k:references(r) for k,r in labels.items()};reports={};details={}
    for n in names:
        folder=RUN/n/'eval_adaptive';rows=read(folder/'rows.json');counts=Counter();cases=[]
        with np.load(folder/'pool.npz') as z:pool={k:z[k] for k in z.files}
        index={str(k):i for i,k in enumerate(pool['ids'])}
        for row in rows:
            i=index[row['id']];ref=refs[row['id']];p=pool['paths'][i];centers=pool['centers'][i];e=pool['events'][i]
            _,check=check_candidates(p,e,ref['label'],ref['current'],ref['truth'],ref['config']);cs=row['corridor_certified_under_eval_truth'];inside=row['inside_predicted_cells']
            for j,c in enumerate(check):
                cm=mode(centers[j],ref['config']);pm=mode(p[j],ref['config']);counts['routes']+=1
                counts['certified_and_inside']+=int(cs[j] and inside[j]);counts['certified_inside_geometry_failure']+=int(cs[j] and inside[j] and not c['tip_segments_clear'])
                counts['wrong_center_raw_mode']+=int(cm!=row['assigned_modes'][j]);counts['output_raw_mode_changed_from_center']+=int(cm!=pm)
                counts['certified_inside_endpoint_failure']+=int(cs[j] and inside[j] and not c['semantic_goal_correct'])
                if not c['TipValid'] or pm!=row['assigned_modes'][j]:cases.append(dict(id=row['id'],slot=j,certified=cs[j],inside=inside[j],center_word=cm,output_word=pm,requested=row['assigned_modes'][j],
                    goal_correct=c['semantic_goal_correct'],segments_clear=c['tip_segments_clear'],event_correct=c['event_state_sequence_correct']))
        # A failing analytic safety implication must not be silently reported as success.
        assert counts['certified_inside_geometry_failure']==0,dict(counts)
        reports[n]=dict(counts);details[n]=cases
    report=dict(models=reports,certificate_implication_checked=True,scope='SavedDEVpredictions; oracle only independent checking',locked_access=False)
    write(out/'RESULTS.json',report);write(out/'FAILURE_CASES.json',details);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--names',nargs='+',required=True);main(**vars(p.parse_args()))
