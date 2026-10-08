"""Same concrete route across registered edits: distinguish route and mode loss.

Uses only already sealed DEV_MODEL predictions. All cross-scene labels come
from the existing full checker; no new candidates or q values are invented.
"""
import argparse
from collections import defaultdict
import numpy as np
from scripts.run_observed_probability import read,write,sha,lines
from scripts.paired_modes_data import DATA,RUN as OLD_RUN
from scripts.research_v3_frequency import RUN
from scripts.research_v3_audit import mode,plain
from scripts.evaluate_paired_modes import references,check_candidates
from scripts.research_v3_analyze_frequency import ARMS

EDGES=[('open','closed'),('open','shifted'),('closed','open'),('shifted','open')]


def survival(source_valid,cross_valid,cross_words,destination_valid,destination_words):
    still={w for v,c,w in zip(source_valid,cross_valid,cross_words) if v and c and w is not None}
    actual={w for v,w in zip(destination_valid,destination_words) if v and w is not None}
    return dict(surviving_source_routes=int(np.sum(np.array(source_valid)&np.array(cross_valid))),
        invalidated_source_routes=int(np.sum(np.array(source_valid)&~np.array(cross_valid))),
        surviving_witness_modes=len(still),retained_witness_modes=len(still&actual),
        lost_witness_modes=sorted(still-actual),surviving_modes=sorted(still))


def main(output):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    refs={k:references(v) for k,v in labels.items()}
    metadata={r['id']:r for r in lines(DATA/'export/metadata.jsonl') if r['id'] in labels}
    grouped=defaultdict(dict)
    for ident in labels:
        family=metadata[ident]['family_id'];variant=metadata[ident]['variant']
        grouped[(family,ident.rsplit('target',1)[1])][variant]=ident
    assert len(grouped)==96 and all(set(v)=={'open','closed','shifted'} for v in grouped.values())
    allrows={};summary={};hashes={}
    for arm in ['initial']+ARMS:
        if arm=='initial':
            pool=OLD_RUN/'evaluation/paired_dev/R1_seed0/pool.npz'
            oldrows=read(RUN/'audit_v4/candidate_rows.json')['0']
        else:
            folder=RUN/('frequency_'+arm)/'evaluation_fixed_q_v2'
            pool=folder/'pool.npz';oldrows=read(folder/'rows.json')
            assert sha(pool)==read(folder/'metrics.json')['pool_sha256']
        hashes[arm]=sha(pool)
        with np.load(pool) as z:data={k:z[k] for k in ('paths','events','ids','labels')}
        index={str(k):j for j,k in enumerate(data['ids'])};rmap={r['id']:r for r in oldrows};rows=[]
        for (family,target),variants in sorted(grouped.items()):
            for origin,destination in EDGES:
                a,b=variants[origin],variants[destination];ai,bi=index[a],index[b]
                sa,db=refs[a],refs[b]
                for k in ('gripper_pose','gripper_open'):
                    np.testing.assert_allclose(sa['current'][k],db['current'][k],atol=1e-6,rtol=0)
                np.testing.assert_allclose(sa['config']['goal_xyz'],db['config']['goal_xyz'],atol=1e-6,rtol=0)
                _,checks=check_candidates(data['paths'][ai],data['events'][ai],db['label'],db['current'],db['truth'],db['config'])
                cross=np.array([c['TipValid'] for c in checks])
                words=[mode(p,db['config']) if v else None for p,v in zip(data['paths'][ai],cross)]
                row=survival(data['labels'][ai],cross,words,data['labels'][bi],rmap[b]['words'])
                row.update(family=family,target=target,origin=origin,destination=destination,source_id=a,destination_id=b,
                    source_valid=data['labels'][ai].tolist(),cross_valid=cross.tolist(),destination_valid=data['labels'][bi].tolist(),
                    cross_words=words,cross_checks=checks,
                    concrete_adaptation_gain=float(data['labels'][bi].mean()-cross.mean()))
                rows.append(row)
        allrows[arm]=rows
        summary[arm]={}
        for a,b in EDGES:
            rr=[r for r in rows if r['origin']==a and r['destination']==b]
            opportunities=sum(r['surviving_witness_modes'] for r in rr);lost=sum(len(r['lost_witness_modes']) for r in rr)
            summary[arm][a+'_to_'+b]=dict(pairs=len(rr),surviving_mode_opportunities=opportunities,
                lost_surviving_modes=lost,lost_fraction=lost/opportunities if opportunities else None,
                pairs_losing_surviving_modes=sum(bool(r['lost_witness_modes']) for r in rr),
                invalidated_source_routes=sum(r['invalidated_source_routes'] for r in rr),
                adaptation_gain=float(np.mean([r['concrete_adaptation_gain'] for r in rr])))
    write(out/'rows.json',plain(allrows));write(out/'RESULTS.json',dict(summary=summary,pool_sha256=hashes,
        source_manifest_sha256=sha(DATA/'export/supervision.jsonl'),
        scope='32 DEV families,96 target pairs per direction; transferred valid paths witness possibility, failed paths do not prove mode absence; no rescoring or optimizer update',
        q_scope='No counterfactual score is reported: original q values cannot be reused as new-scene scores.'))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
