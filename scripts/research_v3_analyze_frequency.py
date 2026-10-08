"""Paired family uncertainty and frozen frequency gate; retain every arm."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


ARMS=['empirical_uniform','empirical_90','empirical_98','balanced','set_matching']


def read(p):return json.loads(Path(p).read_text())


def row_metrics(r):
    return dict(valid=r['raw']['valid_fraction'],oracle=r['raw']['any_valid'],
        modes8=r['raw']['distinct'],recall8=r['raw']['recall'],rare8=r['rare_recall8'],
        modes4=r['selected']['distinct'],rare4=r['rare_recall4'],all_valid4=r['selected']['all_valid'],
        brier=float(np.mean((np.array(r['q'])-np.array(r['valid']))**2)))


def paired(a,b):
    am={r['id']:r for r in a};bm={r['id']:r for r in b};assert set(am)==set(bm)
    families=sorted({r['family'] for r in a});rng=np.random.default_rng(610091)
    draws=rng.integers(len(families),size=(10000,len(families)))
    result={}
    for key in row_metrics(a[0]):
        values=np.array([np.mean([row_metrics(bm[i])[key]-row_metrics(am[i])[key]
            for i in am if am[i]['family']==family]) for family in families])
        boot=values[draws].mean(1)
        result[key]=dict(delta=float(values.mean()),CI95=np.quantile(boot,[.025,.975]).tolist())
    return result


def main(root,output,evaluation='evaluation'):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    metrics={};rows={};training={}
    for arm in ARMS:
        f=root/('frequency_'+arm)
        metrics[arm]=read(f/evaluation/'metrics.json');rows[arm]=read(f/evaluation/'rows.json')
        training[arm]=read(f/'summary.json')
    contrasts={a+'_minus_empirical_uniform':paired(rows['empirical_uniform'],rows[a]) for a in ARMS[1:]}
    contrasts['set_matching_minus_balanced']=paired(rows['balanced'],rows['set_matching'])
    gate={}
    for arm in ('empirical_90','empirical_98'):
        rare=contrasts[arm+'_minus_empirical_uniform']['rare8']
        gate[arm]=bool(rare['delta']<=-.1 and rare['CI95'][1]<0)
    result=dict(metrics=metrics,training=training,contrasts=contrasts,frequency_retention_gate=gate,evaluation=evaluation,
        uncertainty_scope='32 DEV_MODEL families; fixed seed0 trained models. No training-seed uncertainty or final-test claim.',
        decision='Analyze ordinary controls before any novel mechanism; do not substitute an easy empirical arm for the strongest baseline')
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'SUMMARY.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','seed','steps','input_draws','reference_draws','unique_routes','training_seconds','peak_cuda_bytes',
            'valid8','oracle8','distinct8','known_recall8','rare_recall8','distinct4','rare_recall4','selected_valid','brier','nll','ece','checkpoint_sha256'])
        for arm in ARMS:
            m=metrics[arm];t=training[arm];r=m['reliability']
            w.writerow([arm,0,t['steps'],t['sampler']['observation_draws'],sum(t['mode_exposure'].values()),t['unique_routes'],
                t['elapsed_seconds'],t['peak_allocated_bytes'],m['raw']['valid_fraction'],m['raw']['any_valid'],m['raw']['distinct'],
                m['raw']['recall'],m['rare_recall8'],m['selected']['distinct'],m['rare_recall4'],r['selected_valid'],r['brier'],r['nll'],
                r['reliability_bins_ece'],t['last_sha256']])
    text='# Controlled frequency retention: single-seed pilot\n\nAll five arms actually completed; fixed final1200 checkpoints.\n\n'
    text+='| Arm | Valid@8 | Distinct@8 | Known recall@8 | Rare recall@8 | Distinct@4 | Brier |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for arm in ARMS:
        m=metrics[arm]
        text+='| %s | %.3f%% | %.4f | %.3f%% | %.3f%% | %.4f | %.4f |\n'%(arm,100*m['raw']['valid_fraction'],m['raw']['distinct'],
            100*m['raw']['recall'],100*m['rare_recall8'],m['selected']['distinct'],m['reliability']['brier'])
    text+='\nFrequency gate: '+json.dumps(gate)+'. See RESULTS.json for all contrasts and family intervals.\n\n'
    text+='''The majority is a registered left/left passage and minority mass is shared
among all other witnessed passages. Rarity means controlled training frequency,
not path invalidity. These are biased fine-tuning results from a previously
trained R1, not experiments with fresh VLM training. All inputs and geometry
penalties are identical. Set matching processes more reference labels per input
than eight-draw arms; report actual reference counts and wall time in SUMMARY.csv
before interpreting data/compute fairness. The strong ordinary baseline must
remain the comparison target for a new mechanism.

The q head and calibration are frozen from historical R1 seed0. Its probability
metrics measure transfer onto each generator's own pool; they do not establish
an improved scorer on a fixed public pool. No independent final evaluation,
multi-seed new-method result, intermediate task constraint or novelty claim
follows from this pilot. All failures and full candidate sets are retained.
'''
    (out/'REPORT.md').write_text(text)
    print(json.dumps(dict(gate=gate,rows={a:dict(raw=metrics[a]['raw'],rare8=metrics[a]['rare_recall8']) for a in ARMS})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--evaluation',default='evaluation');a=p.parse_args();main(a.root,a.output,a.evaluation)
