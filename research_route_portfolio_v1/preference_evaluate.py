"""Late symbolic preferences: allocate before decode versus select after decode."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha,lines,torch_setup
from research_route_portfolio_v1.evaluate import RUN,POLICY


def preferences(vocab):
    return dict(ground_only=[w for w in vocab if 'over' not in w.split('|')],
                over_any=[w for w in vocab if 'over' in w.split('|')],
                first_gap0=[w for w in vocab if w.split('|')[0]=='gap0'])


def evaluate(seed):
    torch=torch_setup()
    from routeset.mode_geometry import load_fixed_scored_mode_planner,VOCAB
    from research_route_portfolio_v1.planner import PreferenceRoutePlanner
    from scripts.mode_geometry_experiment import Q
    from scripts.evaluate_paired_modes import inputs_for,references,check_candidates
    from scripts.research_v3_audit import mode
    from scripts.analyze_paired_selection import select
    spec=read(POLICY)['datasets']['shift'];data=ROOT/spec['relative_path']
    assert sha(data/'export/observations.jsonl')==spec['observations_sha256']
    assert sha(data/'export/supervision.jsonl')==spec['supervision_sha256']
    rows=lines(data/'export/observations.jsonl');labels={r['id']:r for r in lines(data/'export/supervision.jsonl')}
    assert len(rows)==336 and all(r['split']=='DEV_MODEL' for r in rows)
    checkpoint=ROOT/f'runs/mode_geometry_v1/canonical_C_seed{seed}/last.pt'
    planner=PreferenceRoutePlanner(load_fixed_scored_mode_planner(checkpoint,Q,'cuda').requires_grad_(False))
    request=preferences(VOCAB)
    masks={k:torch.tensor([[w in words for w in VOCAB]],device='cuda') for k,words in request.items()}
    baseline=RUN/f'shift_C_seed{seed}'
    seal=read(baseline/'PREDICTION_SEAL.json')
    assert seal['generator_sha256']==sha(checkpoint) and sha(baseline/'pool.npz')==seal['predictions_sha256']
    with np.load(baseline/'pool.npz') as z:bp={k:z[k] for k in z.files}
    assert np.array_equal(bp['ids'],[r['id'] for r in rows])
    out=RUN/f'preference_C_seed{seed}';out.mkdir(parents=True,exist_ok=False)
    paths=[];events=[];qs=[];assigned=[];chosen=[];hashes={};calls=[]
    hook=planner.planner.generator.head.output.register_forward_hook(lambda module,args,result:calls.append(tuple(result.shape)))
    for row in rows:
        inp=inputs_for(row,labels[row['id']],data/'export/qwen_cache',torch,hashes)
        observed=planner.observe(**inp)
        pr=[];ev=[];qv=[];mv=[];sv=[]
        for key in request:
            result=planner.propose(observed,masks[key])
            for vals,target in [(result['paths'],pr),(result['events'],ev),(result['q'],qv),
                                (result['mode_ids'],mv),(result['selected_indices'],sv)]:target.append(vals[0].cpu().numpy())
        paths.append(pr);events.append(ev);qs.append(qv);assigned.append(mv);chosen.append(sv)
    hook.remove()
    assert len(calls)==336*3 and all(v[:2]==(1,8) for v in calls)
    np.savez_compressed(out/'pool.npz',paths=paths,events=events,q=qs,mode_ids=assigned,
                        selected_indices=chosen,ids=bp['ids'],preference_names=list(request))
    write(out/'PREDICTION_SEAL.json',dict(source_commit=os.environ.get('CODE_COMMIT'),generator_sha256=sha(checkpoint),
        scorer_sha256=sha(Q),baseline_pool_sha256=seal['predictions_sha256'],pool_sha256=sha(out/'pool.npz'),
        source_sha256={str(f.relative_to(SOURCE)):sha(f) for f in (SOURCE/'research_route_portfolio_v1').glob('*.py')},
        preferences=request,input_hashes=hashes,decoded_sets=len(calls),routes_per_preference=8,
        observation_encodings=336,returned_per_preference=4,locked_access=False))
    # Complete prediction seal exists before any ground-truth geometry check.
    baseline_rows=read(baseline/'rows.json');results=[]
    for i,row in enumerate(rows):
        ref=references(labels[row['id']])
        for j,(key,wanted) in enumerate(request.items()):
            _,cc=check_candidates(paths[i][j],events[i][j],ref['label'],ref['current'],ref['truth'],ref['config'])
            valid=[c['TipValid'] for c in cc]
            words=[mode(p,ref['config']) if v else None for p,v in zip(paths[i][j],valid)]
            nominal=[VOCAB[m] for m in bp['mode_ids'][i]]
            restricted=bp['q'][i].copy()
            restricted[[w not in wanted for w in nominal]]=-1.
            posthoc=select(bp['paths'][i],restricted,4)
            def metric(actual,ok,selection):
                values=[actual[t] for t in selection if ok[t] and actual[t] in wanted]
                return dict(distinct=len(set(values)),compliant_valid_fraction=len(values)/len(selection),
                            any_compliant_valid=int(bool(values)))
            old=baseline_rows[i]
            results.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],preference=key,
                proposed8=metric(words,valid,range(8)),proposed4=metric(words,valid,chosen[i][j]),
                baseline8=metric(old['words'],old['valid'],range(8)),baseline4=metric(old['words'],old['valid'],posthoc),
                proposed_selection=chosen[i][j].tolist(),baseline_selection=posthoc))
    report=dict(seed=seed,preferences={},scope='336 reused DEV requests; symbolic requester constraints, no learned preference-language grounding; same8route cost per request and original completeq',locked_access=False,robot_execution=False)
    for key in request:
        rr=[r for r in results if r['preference']==key]
        report['preferences'][key]={arm:{m:float(np.mean([r[arm][m] for r in rr])) for m in rr[0][arm]}
                                   for arm in ('proposed8','proposed4','baseline8','baseline4')}
    write(out/'rows.json',results);write(out/'RESULTS.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);evaluate(**vars(p.parse_args()))
