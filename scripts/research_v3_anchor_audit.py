"""Observed anchor support and semantic localization, no optimizer updates."""
import argparse
import numpy as np
from scripts.run_observed_probability import read,write,sha,lines,torch_setup
from scripts.research_v3_frequency import RUN
from scripts.paired_modes_data import DATA,RUN as OLD_RUN
from scripts.evaluate_paired_modes import inputs_for


def main(output):
    torch=torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    obs=lines(DATA/'export/observations.jsonl');labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl')}
    train_families=sorted({r['parent_id'].rsplit('_',1)[0] for r in obs if r['split']=='TRAIN'})[:16]
    selected=[r for r in obs if r['split']=='DEV_MODEL' or (r['split']=='TRAIN' and r['parent_id'].rsplit('_',1)[0] in train_families)]
    assert len(selected)==432 and {r['split'] for r in selected}=={'TRAIN','DEV_MODEL'}
    paths={'initial':OLD_RUN/'R1_seed0/last.pt','mean':RUN/'safety_mean/last.pt','worst':RUN/'safety_worst/last.pt'}
    records={};summaries={};hashes={}
    for arm,file in paths.items():
        s=torch.load(file,map_location='cpu',weights_only=False)
        model=ProbabilisticGeometryRouteHead(**s['config']['head_options']).cuda();model.load_state_dict(s['model']);model.eval()
        rows=[];bound=float(model.endpoint_residual_bound)
        for row in selected:
            inp=inputs_for(row,labels[row['id']],DATA/'export/qwen_cache',torch,hashes)
            with torch.inference_mode():xyz,event,details=model(**inp)
            # Only now read target geometry for evaluation; not a forward input.
            cfg=read(labels[row['id']]['route_config']);target_index=int(row['id'].rsplit('target',1)[1])
            goal=np.array(cfg['goal_xyz'][target_index]);valid=inp['valid_mask'][0].cpu().numpy()
            points=inp['world_xyz'][0].cpu().numpy()[valid];attention=details['attention'][0].cpu().numpy()[valid]
            anchor=details['anchor_xyz'][0].cpu().numpy();endpoints=xyz[0,:,-1].cpu().numpy()
            assert np.max(np.abs(endpoints-anchor))<=bound+1e-6
            distances=np.linalg.norm(points-goal,axis=-1);positive=distances<=.03
            endpoint_errors=np.linalg.norm(endpoints-goal,axis=-1)
            reachable_distance=float(np.linalg.norm(np.maximum(np.abs(goal-anchor)-bound,0)))
            best_positive=float(attention[positive].max()) if positive.any() else None
            rows.append(dict(id=row['id'],split=row['split'],family=row['parent_id'].rsplit('_',1)[0],
                nearest_observed_goal_distance_m=float(distances.min()),observed_goal_support=bool(positive.any()),
                anchor_error_m=float(np.linalg.norm(anchor-goal)),anchor_box_minimum_goal_distance_m=reachable_distance,
                impossible_under_anchor_bound=reachable_distance>.03,all_endpoints_wrong=bool((endpoint_errors>.03).all()),
                endpoint_error_m=endpoint_errors.tolist(),positive_attention_mass=float(attention[positive].sum()),
                positive_max_attention_rank=int(np.sum(attention>best_positive))+1 if best_positive is not None else None,
                observed_points=len(points),anchor_xyz=anchor.tolist(),goal_xyz=goal.tolist(),residual_bound_m=bound))
        records[arm]=rows;summaries[arm]={}
        for role in ('TRAIN','DEV_MODEL'):
            rr=[r for r in rows if r['split']==role];fail=[r for r in rr if r['all_endpoints_wrong']]
            summaries[arm][role]=dict(requests=len(rr),all_endpoints_wrong=len(fail),
                observed_goal_support=sum(r['observed_goal_support'] for r in rr),
                impossible_anchor_box=sum(r['impossible_under_anchor_bound'] for r in rr),
                failed_with_observed_support=sum(r['observed_goal_support'] for r in fail),
                failed_impossible_anchor_box=sum(r['impossible_under_anchor_bound'] for r in fail),
                failed_positive_attention_mass_mean=float(np.mean([r['positive_attention_mass'] for r in fail])) if fail else None)
        del model,s
    write(out/'rows.json',records);write(out/'RESULTS.json',dict(summary=summaries,checkpoint_sha256={a:sha(p) for a,p in paths.items()},
        input_sha256=hashes,train_families=train_families,
        scope='Fixed first16 TRAIN families and all32 DEV_MODEL families, observation-only forward; target geometry is checked afterwards. No training or checkpoint selection.',
        interpretation='Observed support tests only proximity of visible points; the anchor-box test is a necessary endpoint representability bound, not an execution certificate.'))
    print(summaries,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
