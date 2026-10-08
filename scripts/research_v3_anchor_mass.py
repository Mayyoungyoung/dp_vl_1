"""TRAIN-only fixed-scale attention-mass audit; no changed route predictions."""
import argparse
import numpy as np
from scripts.run_observed_probability import read, write, sha, lines, torch_setup
from scripts.research_v3_frequency import RUN
from scripts.paired_modes_data import DATA
from scripts.evaluate_paired_modes import inputs_for


from routeset.attention_mass import mass_anchor


def main(output):
    torch=torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    rows=[r for r in lines(DATA/'export/observations.jsonl') if r['split']=='TRAIN']
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    assert len(rows)==1152 and len({r['parent_id'].rsplit('_',1)[0] for r in rows})==128
    file=RUN/'safety_mean/last.pt';saved=torch.load(file,map_location='cpu',weights_only=False)
    model=ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda()
    model.load_state_dict(saved['model']);model.eval();records=[];hashes={}
    for n,row in enumerate(rows):
        inp=inputs_for(row,labels[row['id']],DATA/'export/qwen_cache',torch,hashes)
        with torch.inference_mode():
            _,_,details=model(**inp)
            mask=inp['valid_mask'][0];points=inp['world_xyz'][0,mask];weights=details['attention'][0,mask]
            peak=details['anchor_xyz'][0]
            mass,value=mass_anchor(points,weights)
            soft=(points*weights[:,None]).sum(0)
            anchors={k:v.cpu().numpy() for k,v in [('peak',peak),('mass',mass),('soft',soft)]}
        # Geometry below is evaluation only, after all candidate anchors were computed.
        cfg=read(labels[row['id']]['route_config']);goal=np.array(cfg['goal_xyz'][int(row['id'].rsplit('target',1)[1])])
        item=dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],split='TRAIN',anchors={})
        for key,anchor in anchors.items():
            error=float(np.linalg.norm(anchor-goal))
            lower=float(np.linalg.norm(np.maximum(np.abs(anchor-goal)-model.endpoint_residual_bound,0)))
            item['anchors'][key]=dict(xyz=anchor.tolist(),error_m=error,within_3cm=error<=.03,impossible=lower>.03)
        records.append(item)
        if (n+1)%96==0:print(f'Completed {n+1}/{len(rows)} TRAIN requests',flush=True)
    summary={k:dict(within_3cm=sum(r['anchors'][k]['within_3cm'] for r in records),
                   impossible=sum(r['anchors'][k]['impossible'] for r in records),
                   mean_error_m=float(np.mean([r['anchors'][k]['error_m'] for r in records]))) for k in anchors}
    base=summary['peak'];trial=summary['mass']
    gate=base['impossible']>=4 and trial['impossible']<=.75*base['impossible'] and trial['within_3cm']>=base['within_3cm']
    write(out/'rows.json',records)
    write(out/'RESULTS.json',dict(summary=summary,gate=gate,checkpoint_sha256=sha(file),input_sha256=hashes,
        sigma_m=.025,points='all valid observed points, exact untruncated Gaussian kernel',
        scope='TRAIN only, one sealed mean checkpoint; soft expectation diagnostic only; no optimizer or route changes; gate is necessary before registering an intervention'))
    print(dict(summary=summary,gate=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
