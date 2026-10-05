"""TRAIN-only task invariance under actual intermediate-route substitutions."""
import numpy as np
from scripts.run_factored_q import RUN,OLD,load_pool,folder
from scripts.run_observed_probability import read,write,lines,torch_setup


def main(arm='conditional'):
    torch=torch_setup()
    from routeset.factored_q import FactorRouteScorer
    from routeset.observed_probability import ProbabilisticGeometryRouteHead,route_observation_features
    from scripts.evaluate_paired_modes import dataset,inputs_for
    datafolder,role=dataset('SCORE_TRAIN')
    rows={r['id']:r for r in lines(datafolder/'observations.jsonl') if r['split']==role}
    labels={r['id']:r for r in lines(datafolder/'supervision.jsonl') if r['split']==role}
    result={}
    for gs in range(3):
        pool,_=load_pool(gs,'SCORE_TRAIN');st=torch.load(OLD/('R1_seed%d'%gs)/'last.pt',map_location='cpu',weights_only=False)
        generator=ProbabilisticGeometryRouteHead(**st['config']['head_options']).cuda().eval();generator.load_state_dict(st['model'])
        heads=[]
        for ss in range(3):
            state=torch.load(folder(arm,gs,ss)/'best.pt',map_location='cpu',weights_only=False)
            m=FactorRouteScorer(arm).cuda().eval();m.load_state_dict(state['model'])
            heads.append((m,state['normalization']))
        deltas=[[] for _ in heads]
        # Fixed first24 TRAIN requests, regardless of predictions, no DEV selection.
        for i,ident in enumerate(pool['ids'][:24]):
            inp=inputs_for(rows[str(ident)],labels[str(ident)],datafolder/'qwen_cache',torch,{})
            paths=torch.tensor(pool['paths'][i:i+1],device='cuda');events=torch.tensor(pool['events'][i:i+1],device='cuda')
            changed=paths.clone();changed[:,:,1:-1]=paths.roll(1,1)[:,:,1:-1]
            torch.testing.assert_close(changed[:,:,[0,-1]],paths[:,:,[0,-1]],rtol=0,atol=0)
            with torch.inference_mode():
                geo=generator.geometry(**inp,return_point_features=True)
                ctx=generator.head.feature_encoder(inp['features'])+generator.head.state_encoder(inp['current'])+geo['context']
                args=(events,inp['current'],inp['world_xyz'],inp['rgb'],inp['valid_mask'],geo['point_features'],ctx,geo['anchor_xyz'])
                x,c=route_observation_features(paths,*args);xx,cc=route_observation_features(changed,*args)
                np.testing.assert_allclose(x[0].cpu().numpy(),pool['nodes'][i],rtol=1e-5,atol=1e-6)
                for j,(model,norm) in enumerate(heads):
                    def predict(a,b):
                        return model((a-torch.tensor(norm['nodes_mean'],device='cuda'))/torch.tensor(norm['nodes_std'],device='cuda'),
                                     (b-torch.tensor(norm['context_mean'],device='cuda'))/torch.tensor(norm['context_std'],device='cuda'))[...,0].sigmoid()
                    deltas[j].extend((predict(x,c)-predict(xx,cc)).abs().cpu().numpy().ravel().tolist())
        for ss,values in enumerate(deltas):
            result['g%d_s%d'%(gs,ss)]=dict(mean_abs_change=float(np.mean(values)),p95=float(np.quantile(values,.95)),max=float(np.max(values)),values=values)
    triggered=any(v['mean_abs_change']>.02 or v['p95']>.1 for v in result.values())
    filename='task_invariance_diagnosis.json' if arm=='conditional' else arm+'_invariance.json'
    write(RUN/filename,dict(results=result,revision_gate=triggered,arm=arm,modified_paths=576,scorer_candidate_evaluations=1728,
          gate='Any model mean task-probability change >.02 or p95 >.1; fixed24 SCORE_TRAIN requests and all9 models.',
          scope='Task checker label is exactly invariant because endpoints/events are held fixed. Changed paths are real recomputed observation features. This diagnoses task-score sensitivity, not proof it causes the DEV performance gap.'))
    print('TASK_INVARIANCE',triggered,{k:{j:v[j] for j in ('mean_abs_change','p95')} for k,v in result.items()},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',default='conditional');a=p.parse_args();main(a.arm)
