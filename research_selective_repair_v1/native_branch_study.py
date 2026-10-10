"""Matched native-state-feedback pilot, strong controls and exact resume check."""
import argparse
from research_selective_repair_v1.io import RUN,read,write

def run(name):
    import torch,unittest
    from research_selective_repair_v1.body_native_branch import fit
    from research_selective_repair_v1.body_crossing_measure import fit as route_fit
    from research_selective_repair_v1.body_binary_forecast import fit as binary_fit
    from research_selective_repair_v1.check_prefix_recovery import compare
    from research_selective_repair_v1.crossing_study import diagnostics
    from research_selective_repair_v1.mode_recipe_confusion import fit as empirical_fit
    from research_selective_repair_v1.body_screen import screen
    from research_selective_repair_v1.event_decomposition import metrics
    import numpy as np
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover('tests',pattern='test_body_prefix*_v1.py'))
    assert result.wasSuccessful()
    dataset=RUN/'body_native_branch_data_v1/samples.npz'
    a=name+'_recovery_full';b=name+'_recovery_split'
    fit(a,dataset,seed=206017,steps=120);fit(b,dataset,seed=206017,steps=120,stop_after=60);fit(b,dataset,seed=206017,steps=120,resume=True)
    compare(torch.load(RUN/a/'last.pt',map_location='cpu',weights_only=False),torch.load(RUN/b/'last.pt',map_location='cpu',weights_only=False))
    folders=[]
    for label,feedback in [('state',True),('aux',False)]:
        folder='body_native_%s_seed0_pilot_v1'%label;fit(folder,dataset,feedback=feedback);diagnostics(folder,dataset);folders.append(folder)
    route='body_native_route_seed0_pilot_v1';route_fit(route,dataset,kind='recurrent',threads=4);diagnostics(route,dataset)
    binary='body_native_binary_seed0_pilot_v1';binary_fit(binary,dataset,threads=4)
    table,nom=empirical_fit(dataset,'body_native_mode_recipe_confusion_v1')
    with np.load(dataset) as z:d={k:z[k] for k in ('labels','families','options')}
    with np.load(RUN/binary/'TRAIN_diagnostic_predictions.npz') as z:success=z['success'];held=z['heldout']
    from research_selective_repair_v1.mode_recipe_confusion import compose
    prob=compose(success,nom,d['options'],table)
    write(RUN/'body_native_mode_recipe_confusion_v1/DIAGNOSTICS.json',dict(fit=metrics(prob,d['labels'],~held),held_TRAIN=metrics(prob,d['labels'],held),locked_access=False))
    arms=[('state','actual','native',folders[0]),('aux','actual','native',folders[1]),('route','actual','crossing',route),
          ('planned','planned','native',folders[0]),('binary','planned','binary',binary),('confusion','actual','confusion',binary)]
    for label,kind,forecast,folder in arms:
        screen('body_native_%s_TRAIN_screen_seed0_v1'%label,kind,RUN/folder/'last.pt',word_safe=True,forecast=forecast,role='TRAIN',all_goals=True,
               family_start=14,families=2,dataset_scope='native_pilot',recipe_support='prefix')
    write(out/'SUMMARY.json',dict(status='completed',unit_tests=result.testsRun,exact_checkpoint_optimizer_RNG_recovery=True,
        fits=folders+[route,binary],screen_labels=[a[0] for a in arms],scope='TRAIN-held pilot; no DEV/fresh/native returned4/matched SuccessHead claim',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
