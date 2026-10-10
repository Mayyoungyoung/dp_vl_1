"""Frozen TRAIN probability factorization diagnostic; no additional fit/search."""
import argparse
import numpy as np
import torch
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_crossing_measure import compose
from research_selective_repair_v1.event_decomposition import metrics
from research_selective_repair_v1.mode_recipe_confusion import nominal

def run(name):
    torch.set_num_threads(4);out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    dataset=RUN/'body_native_branch_data_v1/samples.npz';m=read(dataset.parent/'MANIFEST.json')
    assert m['native_joint_labels'] and m['no_DEV_feedback'] and sha(dataset)==m['samples_sha256']
    with np.load(dataset) as z:d={k:z[k] for k in ('ids','paths','completed','labels')}
    with np.load(RUN/'body_native_binary_seed0_pilot_v1/TRAIN_diagnostic_predictions.npz') as z:binary=z['success'];held=z['heldout'];assert np.array_equal(z['ids'],d['ids'])
    base=read(RUN/'constraints_seed0_v1/config.json')['post_base'];planned=nominal(d['paths'],d['completed'],base);reports=[]
    for label in ('state','aux','route'):
        file=RUN/('body_native_%s_seed0_pilot_v1'%label)/'TRAIN_diagnostic_predictions.npz'
        with np.load(file) as z:p={k:z[k] for k in z.files if k not in ('ids','slots','options','families','heldout')};assert np.array_equal(z['ids'],d['ids'])
        prob=np.zeros((len(binary),17),np.float32)
        for ident in sorted(set(d['ids'])):
            ix=np.flatnonzero(d['ids']==ident);c=d['completed'][ix[0]];h=np.clip(c[:,2]-base,.025,.4)
            cfg=dict(row_x=[float(c[j:j+2,0].mean()) for j in (0,2)],post_y=[c[j:j+2,1].tolist() for j in (0,2)],post_heights=[float(h[j:j+2].max()) for j in (0,2)],post_base_z=base,tip_clearance_m=.02)
            prob[ix]=compose({k:v[ix] for k,v in p.items()},d['paths'][ix],cfg)[0]
        before_region=(np.exp(p['log_weights'])*(1/(1+np.exp(np.clip(p['hazard'],-40,40)))).prod(-1)/(1+np.exp(np.clip(-p['clear'],-40,40)))).sum(-1)
        mass=prob[:,1:].sum(-1);conditional=prob[:,1:]/np.maximum(mass[:,None],1e-8);zero=mass<1e-8
        for i in np.flatnonzero(zero):
            conditional[i]=0
            if planned[i]>0:conditional[i,planned[i]-1]=1
            else:conditional[i]=1/16
        rebased=np.column_stack([1-binary,binary[:,None]*conditional])
        reports.append(dict(model=label,input_sha256=sha(file),original=metrics(prob,d['labels'],held),
            before_region_success_brier=float(np.square(before_region[held]-(d['labels'][held]>0)).mean()),
            common_binary_normalized_conditional_words=metrics(rebased,d['labels'],held),
            mean_before_region_success=float(before_region[held].mean()),mean_after_region_success=float(mass[held].mean()),
            normalization_fallback_rows=int(zero.sum())))
    write(out/'SUMMARY.json',dict(records=reports,scope='TRAIN-held frozen prediction diagnostic only; rebase uses same fitted binary control,not calibrated probabilities or a learned new core. No fit/DEV/native query.',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
