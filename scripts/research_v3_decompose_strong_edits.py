"""Read sealed local edit checks; separate endpoint failure from route capacity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(workspace,output,reference_overlap=False):
    base=Path(workspace);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    snapshots={'safety_mean':('research_v3_linear_closure_20261009','evaluation_matched_q_v1'),
        'safety_worst':('research_v3_checkpoint_20261009','evaluation_fixed_q_v2'),
        'margin_mean':('research_v3_linear_closure_20261009','evaluation_matched_q_v1'),
        'optimizer_restored':('research_v3_anchor_optimizer_closure_20261009','evaluation_matched_q_v1'),
        'frequency_set_canonical':('research_v3_canonical_closure_20261009','evaluation_fixed_q_v2')}
    hashes={}
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def read(p):
        hashes[str(p.relative_to(base))]=sha(p)
        return json.loads(p.read_text())
    root=base/'runs/research_v3_strong_counterfactual_closure_20261009/runs/research_v3_v1/strong_counterfactual_v1'
    results=read(root/'RESULTS.json');rows=read(root/'rows.json');allrows={};summary={}
    known={}
    if reference_overlap:
        support=base/'runs/research_v3_checkpoint_20261009/runs/research_v3_v1/frequency_support_v1'
        receipt=read(support/'receipt.json');assert sha(support/'support.npz')==receipt['support_sha256']
        hashes[str((support/'support.npz').relative_to(base))]=receipt['support_sha256']
        with np.load(support/'support.npz') as z:
            modes,mask=z['modes'],z['mask']
            known={str(k):set(modes[j,mask[j]]) for j,k in enumerate(z['ids'])}
    for name,(snapshot,evaluation) in snapshots.items():
        folder=base/'runs'/snapshot/'runs/research_v3_v1'/name/evaluation
        assert sha(folder/'pool.npz')==results['pool_sha256'][name]
        hashes[str((folder/'pool.npz').relative_to(base))]=sha(folder/'pool.npz')
        dest={r['id']:r for r in read(folder/'rows.json')};enriched=[]
        for row in rows[name]:
            r=dest[row['destination_id']];checks=r['candidates'];valid=[bool(c['TipValid']) for c in checks]
            assert valid==row['destination_valid']==r['valid']
            words=[w for v,w in zip(valid,r['words']) if v and w is not None]
            duplicates=len(words)-len(set(words))
            correct_invalid=sum(c['semantic_goal_correct'] and not c['TipValid'] for c in checks)
            eligible=duplicates+correct_invalid
            lost=len(row['lost_witness_modes'])
            assert eligible<=row['destination_invalid_or_duplicate_slots']
            enriched.append(dict(family=row['family'],target=row['target'],origin=row['origin'],destination=row['destination'],
                destination_id=row['destination_id'],lost_modes=lost,slot_feasible=row['slot_feasible_lost_modes'],
                all_endpoints_wrong=all(not c['semantic_goal_correct'] for c in checks),
                correct_endpoint_invalid_slots=correct_invalid,duplicate_valid_slots=duplicates,
                correct_endpoint_slot_feasible=min(lost,eligible)))
            if reference_overlap:
                kl=len(set(row['lost_witness_modes'])&known[row['destination_id']]);ul=lost-kl
                enriched[-1].update(known_lost_modes=kl,unreferenced_lost_modes=ul,
                    known_correct_endpoint_slot_feasible=min(kl,eligible),unreferenced_correct_endpoint_slot_feasible=min(ul,eligible))
        allrows[name]=enriched;summary[name]={}
        for edge,registered in results['summary'][name].items():
            a,b=edge.split('_to_');rr=[r for r in enriched if r['origin']==a and r['destination']==b]
            assert len(rr)==96 and sum(r['slot_feasible'] for r in rr)==registered['slot_feasible_lost_modes']
            families=sorted({r['family'] for r in rr});assert len(families)==32
            rng=np.random.default_rng(610092);draws=rng.integers(32,size=(10000,32))
            intervals={}
            for key in ('slot_feasible','correct_endpoint_slot_feasible'):
                v=np.array([np.mean([r[key] for r in rr if r['family']==f]) for f in families])
                intervals[key]=dict(mean_per_pair=float(v.mean()),CI95=np.quantile(v[draws].mean(1),[.025,.975]).tolist())
            summary[name][edge]=dict(slot_feasible=sum(r['slot_feasible'] for r in rr),
                slot_feasible_on_all_wrong_endpoint_pairs=sum(r['slot_feasible'] for r in rr if r['all_endpoints_wrong']),
                slot_feasible_on_some_correct_endpoint_pairs=sum(r['slot_feasible'] for r in rr if not r['all_endpoints_wrong']),
                correct_endpoint_slot_feasible=sum(r['correct_endpoint_slot_feasible'] for r in rr),
                pairs_with_correct_endpoint_slot_feasible=sum(r['correct_endpoint_slot_feasible']>0 for r in rr),
                all_wrong_endpoint_pairs=sum(r['all_endpoints_wrong'] for r in rr),intervals=intervals)
            if reference_overlap:
                for key in ('known_lost_modes','unreferenced_lost_modes','known_correct_endpoint_slot_feasible','unreferenced_correct_endpoint_slot_feasible'):
                    summary[name][edge][key]=sum(r[key] for r in rr)
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('RESULTS.json',dict(summary=summary,source_sha256=hashes,
        scope='Read-only all5 models/all4 directions. Oracle constructive opportunities, descriptive label-defined groups, no trained model or q comparison. Fixed32 DEV families; seed0 model uncertainty not estimated.'))
    write('rows.json',allrows);print(json.dumps(summary['safety_mean'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);p.add_argument('--reference-overlap',action='store_true')
    a=p.parse_args();main(a.workspace,a.output,a.reference_overlap)
