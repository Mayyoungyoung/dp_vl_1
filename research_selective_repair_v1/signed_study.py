"""Focused signed-support response fits; same2400rows/budget for every arm."""
import argparse
import torch
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha

def fit_study(name):
    from research_selective_repair_v1.body_crossing_measure import fit
    from research_selective_repair_v1.body_binary_forecast import fit as binary_fit
    from research_selective_repair_v1.check_prefix_recovery import compare
    from research_selective_repair_v1.crossing_study import diagnostics
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);dataset=RUN/'body_signed_events_pilot_v1/samples.npz'
    m=read(dataset.parent/'MANIFEST.json');assert m['rows']==2400 and m['samples_sha256']==sha(dataset) and m['no_DEV_feedback']
    a=name+'_recovery_full';b=name+'_recovery_split'
    fit(a,dataset,kind='nonrecurrent',seed=206017,steps=120,threads=4)
    fit(b,dataset,kind='nonrecurrent',seed=206017,steps=120,stop_after=60,threads=4)
    fit(b,dataset,kind='nonrecurrent',seed=206017,steps=120,resume=True,threads=4)
    compare(torch.load(RUN/a/'last.pt',map_location='cpu',weights_only=False),torch.load(RUN/b/'last.pt',map_location='cpu',weights_only=False))
    for readout in ('analytic','categorical_aux'):
        folder='body_signed_nonrecurrent_%s_seed0_pilot_v1'%readout
        fit(folder,dataset,kind='nonrecurrent',readout=readout,seed=0,threads=4);diagnostics(folder,dataset)
    binary_fit('body_signed_binary_seed0_pilot_v1',dataset,seed=0)
    write(out/'SUMMARY.json',dict(exact_checkpoint_optimizer_RNG_recovery=True,dataset_sha256=sha(dataset),fit_rows=2400,
        matching='Two MLP same-event-supervision readouts and binary control,2400updates each,seed0; no GRU necessity claim',
        scope='All16TRAINfamilies/all3old goals plus96target0 signed interventions; partial signed-support pilot,not full lowering-generalization evidence',locked_access=False))

def screens(name):
    from research_selective_repair_v1.body_screen import screen
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    arms=[('identity','identity','crossing',None),('lift','lift','crossing',None),('lower','lower','crossing',None),
        ('actual','actual','crossing','body_signed_nonrecurrent_analytic_seed0_pilot_v1'),
        ('planned','planned','crossing','body_signed_nonrecurrent_analytic_seed0_pilot_v1'),
        ('categorical','actual','crossing','body_signed_nonrecurrent_categorical_aux_seed0_pilot_v1'),
        ('binary','planned','binary','body_signed_binary_seed0_pilot_v1')]
    for label,kind,forecast,folder in arms:
        screen('body_signed_%s_TRAIN_screen_seed0_v1'%label,kind,RUN/folder/'last.pt' if folder else None,
            word_safe=True,forecast=forecast,all_goals=True,role='TRAIN',family_start=14,families=2,dataset_scope='signed_pilot',recipe_support='signed')
    write(out/'SUMMARY.json',dict(arms=arms,scope='36TRAIN-held requests/all3goals geometry; 12lowered goal0 requests supply same2400row fit. No DEV/native/acceptance claim',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--screen',action='store_true');a=p.parse_args();screens(a.name) if a.screen else fit_study(a.name)
