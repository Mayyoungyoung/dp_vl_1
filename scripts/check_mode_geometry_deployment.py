"""Real saved DEV replay, fixed scorer equality and observation-only API."""
import argparse
import numpy as np
from scripts.mode_geometry_experiment import RUN,Q,DATA,read,write,lines,sha,torch_setup


def main(name,output):
    torch=torch_setup()
    from routeset.mode_geometry import load_fixed_scored_mode_planner
    from scripts.evaluate_paired_modes import inputs_for
    from scripts.analyze_paired_selection import select
    folder=RUN/name/'eval_adaptive'
    with np.load(folder/'pool.npz') as z:pool={k:z[k] for k in z.files}
    ident=str(pool['ids'][0]);rows={r['id']:r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL'}
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    inp=inputs_for(rows[ident],labels[ident],DATA/'export/qwen_cache',torch,{})
    model=load_fixed_scored_mode_planner(RUN/name/'last.pt',Q,'cuda')
    with torch.inference_mode():value=model(**inp)
    errors={}
    for key in ('paths','events','q'):
        actual=value[key][0].cpu().numpy();expected=pool[key][0]
        np.testing.assert_array_equal(actual,expected);errors[key]=float(np.abs(actual-expected).max())
    np.testing.assert_array_equal(value['selected_indices'][0].cpu().numpy(),select(pool['paths'][0],pool['q'][0],4))
    assert value['selected_paths'].shape==(1,4,24,3)
    write(RUN/output,dict(max_absolute_errors=errors,selected_indices_exact=True,request=ident,
        model_sha256=sha(RUN/name/'last.pt'),scorer_sha256=sha(Q),input_keys=sorted(inp)))
    print(errors,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.name,a.output)
