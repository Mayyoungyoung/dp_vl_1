"""Saved-prediction portability diagnostic; no inference policy or model update."""
import argparse,json
from collections import Counter
import numpy as np
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.geometry import node_radii_numpy
from research_realized_coverage_v1.core import DATA,old,read,write,lines,sha
from scripts.evaluate_paired_modes import references,check_candidates
from scripts.research_v3_audit import mode


def main(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    witness_file=old.RUN/'strong_counterfactual_v1/rows.json'
    assert sha(witness_file)=='515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084'
    edges=read(witness_file)['safety_mean']
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    refs={k:references(r) for k,r in labels.items()};reports={};records={};hashes={}
    kinds=['copied_absolute','transferred_relative','destination_center','destination_native']
    for seed in range(3):
        folder=RUN/('refreshed_tapered_bounded_seed%d'%seed)/'eval_adaptive'
        with np.load(folder/'pool.npz') as z:pool={k:z[k] for k in z.files}
        rows={r['id']:r for r in read(folder/'rows.json')};index={str(k):i for i,k in enumerate(pool['ids'])}
        hashes[str(folder/'pool.npz')]=sha(folder/'pool.npz');counts=Counter();details=[]
        for edge in edges:
            src,dst=edge['source_id'],edge['destination_id'];si,di=index[src],index[dst];ref=refs[dst]
            for j,word in enumerate(rows[src]['assigned_modes']):
                counts['source_slots']+=1
                matches=[k for k,w in enumerate(rows[dst]['assigned_modes']) if w==word]
                if not matches:counts['not_requested_at_destination']+=1;continue
                if not rows[src]['valid'][j] or rows[src]['words'][j]!=word:counts['source_not_valid_same_mode']+=1;continue
                k=matches[0];counts['eligible_matched_valid_source']+=1
                sp=pool['paths'][si,j];sc=pool['centers'][si,j];sr=node_radii_numpy(pool['radii'][si,j])
                dc=pool['centers'][di,k];dr=node_radii_numpy(pool['radii'][di,k]);fraction=np.divide(sp-sc,sr[:,None],out=np.zeros_like(sp),where=sr[:,None]>1e-12)
                assert np.max(np.abs(fraction))<=1.0001
                fraction=np.clip(fraction,-1,1);fraction[[0,-1]]=0
                transferred=dc+dr[:,None]*fraction
                candidates=np.array([sp,transferred,dc,pool['paths'][di,k]])
                events=np.repeat(pool['events'][di,k][None],4,axis=0)
                _,checked=check_candidates(candidates,events,ref['label'],ref['current'],ref['truth'],ref['config'])
                success=[bool(c['TipValid'] and mode(p,ref['config'])==word) for c,p in zip(checked,candidates)]
                for kind,ok in zip(kinds,success):counts[kind+'_valid_same_mode']+=int(ok)
                if not success[0]:
                    counts['old_coordinates_invalid_same_mode']+=1
                    for kind,ok in zip(kinds,success):counts[kind+'_repair']+=int(ok)
                details.append(dict(source_id=src,destination_id=dst,family=edge['family'],mode=word,source_slot=j,destination_slot=k,
                                    outcomes=dict(zip(kinds,success)),destination_corridor_certified=rows[dst]['corridor_certified_under_eval_truth'][k]))
        den=counts['eligible_matched_valid_source'];repair_den=counts['old_coordinates_invalid_same_mode']
        reports[str(seed)]=dict(counts=dict(counts),success_fraction={k:counts[k+'_valid_same_mode']/den for k in kinds},
                               repair_fraction={k:counts[k+'_repair']/repair_den if repair_den else None for k in kinds})
        records[str(seed)]=details
    report=dict(seed_results=reports,pool_hashes=hashes,witness_sha256=sha(witness_file),
                scope='Matched valid source/requested destination slots from saved3boundedcontinuations; not fixed2169/270denominators and not a deployment8route policy',
                construction='Hold bounded source relative coordinates fixed; map through observed destination predicted cells; all events from destination prediction',
                inference_decodes_added=0,model_updates=0,locked_access=False)
    write(out/'RESULTS.json',report);write(out/'rows.json',records);print(json.dumps({k:v for k,v in report.items() if k!='pool_hashes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);main(**vars(p.parse_args()))
