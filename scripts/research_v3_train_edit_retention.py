"""Registered TRAIN-only computations on sealed predictions; no model forward."""
import argparse
from collections import defaultdict
import numpy as np
from scripts.run_observed_probability import read,write,sha,lines
from scripts.paired_modes_data import DATA
from scripts.research_v3_frequency import RUN,SUPPORT
from scripts.research_v3_counterfactual import EDGES,survival
from scripts.research_v3_audit import mode,plain
from scripts.evaluate_paired_modes import references,check_candidates


def nearest_mean_point_distance(path,teachers):
    if len(teachers)==0:return None
    return float(np.linalg.norm(np.asarray(teachers)-path[None],axis=-1).mean(-1).min())


def main(output):
    out=RUN/output;out.mkdir(exist_ok=False)
    old=RUN/'posttrain_collision_v1';receipt=read(old/'RESULTS.json')
    assert sha(old/'predictions.npz')==receipt['prediction_sha256']=='9e3a136a578eb49977b0308555c6141a21cc2cfc6c6b39e1a1128d962febe301'
    assert receipt['checkpoint_sha256']==sha(RUN/'safety_mean/last.pt')=='ff2dfbc5463a38e9acb2af740e9b605cbfda5d1317cc146f5f2c4e65c2a7a60c'
    manifest=DATA/'export/supervision.jsonl'
    assert sha(manifest)=='dcca00cc4a7679c7fd6426b7e59801dcbe508744181bdf1cb7b8d0e48c7fa1a8'
    labels={r['id']:r for r in lines(manifest) if r['split']=='TRAIN'}
    metadata={r['id']:r for r in lines(DATA/'export/metadata.jsonl') if r['id'] in labels}
    with np.load(old/'predictions.npz') as z:data={k:z[k] for k in z.files}
    index={str(k):i for i,k in enumerate(data['ids'])};assert len(index)==len(labels)==1152 and set(index)==set(labels)
    source_hashes={str(manifest):sha(manifest),str(old/'predictions.npz'):sha(old/'predictions.npz'),str(old/'rows.json'):sha(old/'rows.json')}
    refs={};own={};groups=defaultdict(dict)
    saved={r['id']:r for r in read(old/'rows.json')}
    support_receipt=read(SUPPORT/'receipt.json');assert sha(SUPPORT/'support.npz')==support_receipt['support_sha256']
    # The immutable support file includes permitted DEV arrays; only TRAIN
    # indices are used below. No DEV statistics, threshold or losses are computed.
    with np.load(SUPPORT/'support.npz') as z:support={k:z[k] for k in ('paths','modes','mask','ids','splits')}
    sindex={str(k):j for j,k in enumerate(support['ids'])}
    for ident,label in labels.items():
        ref=references(label);refs[ident]=ref;i=index[ident]
        checks=check_candidates(data['paths'][i],data['events'][i],label,ref['current'],ref['truth'],ref['config'])[1]
        assert [c['TipValid'] for c in checks]==[c['TipValid'] for c in saved[ident]['candidates']]
        own[ident]=dict(checks=checks,valid=np.array([c['TipValid'] for c in checks]),
            words=[mode(p,ref['config']) if c['TipValid'] else None for p,c in zip(data['paths'][i],checks)])
        groups[(metadata[ident]['family_id'],ident.rsplit('target',1)[1])][metadata[ident]['variant']]=ident
        for file in [label['route_config'],label['verification_only']]+label['routes']:
            source_hashes[file]=sha(file);assert source_hashes[file]==support_receipt['input_sha256'][file]
        sj=sindex[ident];assert support['splits'][sj]=='TRAIN'
    assert len(groups)==384 and len({f for f,t in groups})==128
    assert all(set(g)=={'open','closed','shifted'} for g in groups.values())
    rows=[];distances=[]
    for (family,target),variants in sorted(groups.items()):
        for origin,destination in EDGES:
            a,b=variants[origin],variants[destination];sa,db=refs[a],refs[b];ai=index[a]
            for key in ('gripper_pose','gripper_open'):np.testing.assert_allclose(sa['current'][key],db['current'][key],atol=1e-6,rtol=0)
            np.testing.assert_allclose(sa['config']['goal_xyz'],db['config']['goal_xyz'],atol=1e-6,rtol=0)
            checks=check_candidates(data['paths'][ai],data['events'][ai],db['label'],db['current'],db['truth'],db['config'])[1]
            cross=np.array([c['TipValid'] for c in checks]);words=[mode(p,db['config']) if c['TipValid'] else None for p,c in zip(data['paths'][ai],checks)]
            row=survival(own[a]['valid'],cross,words,own[b]['valid'],own[b]['words'])
            classified=[w for v,w in zip(own[b]['valid'],own[b]['words']) if v and w is not None]
            free_correct=len(classified)-len(set(classified))+sum(c['semantic_goal_correct'] and not c['TipValid'] for c in own[b]['checks'])
            sj=sindex[b];mask=support['mask'][sj];teacher_paths=support['paths'][sj,mask];teacher_modes=support['modes'][sj,mask]
            known=set(teacher_modes);lost=set(row['lost_witness_modes']);kl=len(lost&known);ul=len(lost-known)
            row.update(family=family,target=target,origin=origin,destination=destination,source_id=a,destination_id=b,
                correct_endpoint_slot_feasible=min(len(lost),free_correct),correct_endpoint_slots=free_correct,
                all_endpoints_wrong=all(not c['semantic_goal_correct'] for c in own[b]['checks']),
                known_lost_modes=kl,unreferenced_lost_modes=ul,
                known_correct_endpoint_slot_feasible=min(kl,free_correct),unreferenced_correct_endpoint_slot_feasible=min(ul,free_correct),
                source_valid=own[a]['valid'].tolist(),cross_valid=cross.tolist(),destination_valid=own[b]['valid'].tolist(),cross_words=words)
            rows.append(row)
            original_paths=db['paths'][np.array(db['reference_valid'],bool)]
            original_modes=np.array([mode(p,db['config']) for p in original_paths])
            for j,(p,w) in enumerate(zip(data['paths'][ai],words)):
                if not (own[a]['valid'][j] and cross[j] and w is not None):continue
                distances.append(dict(family=family,target=target,origin=origin,destination=destination,source_id=a,destination_id=b,
                    candidate=j,mode=w,lost_mode=w in lost,eligible_lost_path=w in lost and free_correct>0,
                    nearest_original_mean_point_m=nearest_mean_point_distance(p,original_paths[original_modes==w]),
                    nearest_augmented_mean_point_m=nearest_mean_point_distance(p,teacher_paths[teacher_modes==w])))
    summary={}
    for a,b in EDGES:
        rr=[r for r in rows if r['origin']==a and r['destination']==b];dd=[d for d in distances if d['origin']==a and d['destination']==b]
        sums={k:sum(r[k] for r in rr) for k in ('surviving_witness_modes','slot_feasible_lost_modes','correct_endpoint_slot_feasible','known_lost_modes','unreferenced_lost_modes','known_correct_endpoint_slot_feasible','unreferenced_correct_endpoint_slot_feasible')}
        sums.update(pairs=len(rr),lost_modes=sum(len(r['lost_witness_modes']) for r in rr),
            pairs_with_correct_endpoint_opportunity=sum(r['correct_endpoint_slot_feasible']>0 for r in rr),
            slot_feasible_on_all_wrong_endpoint_pairs=sum(r['slot_feasible_lost_modes'] for r in rr if r['all_endpoints_wrong']))
        sums['distances']={}
        for group in ('all_survivors','eligible_lost_paths'):
            picked=dd if group=='all_survivors' else [d for d in dd if d['eligible_lost_path']]
            stats={}
            for key in ('nearest_original_mean_point_m','nearest_augmented_mean_point_m'):
                values=[d[key] for d in picked if d[key] is not None]
                stats[key]=dict(count=len(values),missing_mode=len(picked)-len(values),quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else [])
            sums['distances'][group]=stats
        summary[a+'_to_'+b]=sums
    write(out/'rows.json',plain(rows));write(out/'distances.json',plain(distances))
    write(out/'RESULTS.json',plain(dict(summary=summary,prediction_sha256=receipt['prediction_sha256'],support_sha256=support_receipt['support_sha256'],
        input_sha256=source_hashes,rows_sha256=sha(out/'rows.json'),distances_sha256=sha(out/'distances.json'),
        scope='Only1152 TRAIN predictions/checks/distances computed, no model/q forward or update. Support container includes permitted DEV arrays, unused in computation. Distances are all24-point mean Euclidean distances, not thresholds. Known/unreferenced capacities compete.')))
    print(plain(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
