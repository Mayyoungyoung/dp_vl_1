"""Prospective realization-space preservation support ablation.

Remove only nominal-word equality. Exact same fitted fields, probabilities,
three measured recipes, .01 predicted guards and original q. Truth post-seal.
"""
import argparse
from research_selective_repair_v1.io import RUN,read,write

def run(name):
    from research_selective_repair_v1.body_screen import screen
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    assert read(RUN/'body_nominal_guard_DEV_diagnostic_v1/SUMMARY.json')['records'][0]['changed_requests']==69
    write(out/'PROTOCOL.json',dict(hypothesis='Nominal word equality blocks repairs predicted to preserve realized mode mass; remove it before truth/new execution',
        unchanged='Full2304 TRAIN feedback,seed0 frozen MLP fields,three recipe support,.01 per-word expected coverage/.01 per-route failure guards,6561exact allocations,8final,original complete q4',
        geometry_gates='Existing body protocol unchanged; no retroactive acceptance',locked_access=False))
    arms=[('actual','actual','crossing','body_crossing_nonrecurrent_analytic_seed0_full_v1'),
          ('planned','planned','crossing','body_crossing_nonrecurrent_analytic_seed0_full_v1'),
          ('categorical','actual','crossing','body_crossing_nonrecurrent_categorical_aux_seed0_full_v1'),
          ('binary','planned','binary','body_binary_seed0_v1')]
    results=[]
    for label,kind,forecast,folder in arms:
        screen('body_realized_guard_%s_DEV_screen_seed0_v1'%label,kind,RUN/folder/'last.pt',
            protected=True,word_safe=False,forecast=forecast,all_goals=True,role='DEV_MODEL')
        results.append(dict(method=label,**read(RUN/('body_realized_guard_%s_DEV_screen_seed0_v1'%label)/'SUMMARY.json')))
    write(out/'SUMMARY.json',dict(records=results,scope='Support ablation/current observations and post-seal geometry; no native execution/matched heads/new layouts',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
