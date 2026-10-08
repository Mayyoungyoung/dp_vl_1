"""Seal full TRAIN predictions, then audit residual geometric errors."""
import argparse
from collections import Counter
import numpy as np
from scripts.run_observed_probability import torch_setup,read,write,sha,lines
from scripts.research_v3_frequency import RUN
from scripts.paired_modes_data import DATA
from scripts.evaluate_paired_modes import inputs_for,check_candidates


def main(output,checkpoint_name='safety_mean'):
    torch=torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from routeset.segment_clearance import path_segment_clearances
    out=RUN/output;out.mkdir(exist_ok=False)
    folder=DATA/'export'
    rows=[r for r in lines(folder/'observations.jsonl') if r['split']=='TRAIN']
    labels={r['id']:r for r in lines(folder/'supervision.jsonl') if r['split']=='TRAIN'}
    assert len(rows)==1152 and len({r['parent_id'].rsplit('_',1)[0] for r in rows})==128
    assert checkpoint_name in ('safety_mean','margin_mean','margin_linear')
    checkpoint=RUN/checkpoint_name/'last.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda()
    model.load_state_dict(saved['model']);model.eval();torch.manual_seed(0)
    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),out/'rng_before.pt')
    hashes={};paths=[];events=[]
    for row in rows:
        inp=inputs_for(row,{'observation':labels[row['id']]['observation']},folder/'qwen_cache',torch,hashes)
        with torch.inference_mode():xyz,event,_=model(**inp)
        paths.append(xyz[0].cpu().numpy());events.append(event[0].cpu().numpy())
    paths=np.array(paths);events=np.array(events)
    np.savez_compressed(out/'predictions.npz',paths=paths,events=events,ids=[r['id'] for r in rows])
    prediction_hash=sha(out/'predictions.npz')
    del model,saved
    records=[];counts=Counter();deficits=[];bad_segments=[]
    for i,row in enumerate(rows):
        label=labels[row['id']];config=read(label['route_config'])
        with np.load(label['observation']) as z:current={k:z[k] for k in ('gripper_pose','gripper_open')}
        with np.load(label['verification_only']) as z:truth={k:z[k] for k in ('obstacle_centers','obstacle_halfsizes')}
        _,candidates=check_candidates(paths[i],events[i],label,current,truth,config)
        with torch.inference_mode():
            clearance=path_segment_clearances(torch.tensor(paths[i:i+1],device='cuda'),
                torch.tensor(truth['obstacle_centers'][None],dtype=torch.float32,device='cuda'),
                torch.tensor(truth['obstacle_halfsizes'][None],dtype=torch.float32,device='cuda'))[0].cpu().numpy()
        gap=np.maximum(.02-clearance,0)
        for j,c in enumerate(candidates):
            counts['candidates']+=1;counts['valid']+=int(c['TipValid'])
            for k in ('semantic_goal_correct','post_segments_clear','workspace_floor_correct'):counts[k]+=int(c[k])
            counts['post_collision_with_correct_goal']+=int(c['semantic_goal_correct'] and not c['post_segments_clear'])
            c['maximum_post_clearance_deficit_m']=float(gap[j].max())
            c['violating_post_segments']=int((gap[j]>0).sum())
            if not c['post_segments_clear']:
                deficits.append(float(gap[j].max()));bad_segments.append(c['violating_post_segments'])
        counts['requests']+=1;counts['any_valid']+=int(any(c['TipValid'] for c in candidates))
        counts['all_endpoints_wrong']+=int(not any(c['semantic_goal_correct'] for c in candidates))
        records.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],candidates=candidates))
        for key in ('route_config','verification_only'):hashes[label[key]]=sha(label[key])
    assert sha(out/'predictions.npz')==prediction_hash
    result=dict(counts=dict(counts),colliding_maximum_deficit_quantiles_m=np.quantile(deficits,[0,.25,.5,.75,1]).tolist() if deficits else [],
        colliding_violating_segment_counts=dict(Counter(bad_segments)),checkpoint_sha256=sha(checkpoint),
        prediction_sha256=prediction_hash,input_sha256=hashes,
        checkpoint_name=checkpoint_name,
        scope='All1152 original TRAIN requests, fixed named final model, no updates or scorer role access. Oracle geometry opened after all predictions sealed; diagnostic only.')
    write(out/'rows.json',records);write(out/'RESULTS.json',result)
    print({k:v for k,v in result.items() if k!='input_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint-name',default='safety_mean',choices=['safety_mean','margin_mean','margin_linear']);a=p.parse_args();main(a.output,a.checkpoint_name)
