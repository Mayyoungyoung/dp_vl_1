"""Descriptive feature/label shift; no scorer fitting or calibration changes."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,sha


def main(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    results={};hashes={}
    for seed in range(3):
        arm='R2_seed%d'%seed
        train=root/'evaluation/SCORE_TRAIN'/arm/'pool.npz'
        with np.load(train) as a:
            norm={k:(a[k].mean(tuple(range(a[k].ndim-1))),np.maximum(a[k].std(tuple(range(a[k].ndim-1))),.01)) for k in ('nodes','context')}
        for role in ('SCORE_TRAIN','DEV_SCORE','CALIBRATION','paired_dev'):
            p=root/'evaluation'/role/arm/'pool.npz';hashes[str(p)]=sha(p)
            with np.load(p) as a:
                row=dict(valid=float(a['labels'].mean()),frozen_mean_q=float(a['q'].mean()),groups={})
                for key,groups in [('context',{'learned_context':slice(0,128),'endpoint_relative_to_anchor':slice(128,131)}),
                                   ('nodes',{'observed_local_geometry_rgb_events':slice(0,15),'learned_point_features':slice(15,79)})]:
                    z=(a[key]-norm[key][0])/norm[key][1]
                    for name,sl in groups.items():
                        x=z[...,sl];row['groups'][name]=dict(standardized_rms=float(np.sqrt(np.mean(x*x))),fraction_abs_z_gt3=float(np.mean(np.abs(x)>3)))
            if role=='paired_dev':
                p=root/'reliability'/arm/('calibration_seed%d'%seed)/'paired_dev.npz';hashes[str(p)]=sha(p)
                with np.load(p) as a:row['calibrated_mean_q']=float(a['q'].mean());row['calibrated_mean_q_minus_valid']=float((a['q']-a['labels']).mean())
            results[arm+'/'+role]=row
    write(output/'RESULTS.json',dict(results=results,input_sha256=hashes,
        scope='Post-result descriptive diagnostic. Normalization uses SCORE_TRAIN only, exactly matching scorer normalization. Feature RMS or z>3 is not an OOD detector, a causal explanation, or a validity guarantee; no fitted parameters, threshold selection or changed q.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.root,a.output)
