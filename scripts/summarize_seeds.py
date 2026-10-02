"""Aggregate all declared training seeds without treating repeats as candidates."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',default='runs/main')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--output',default='reports/multiseed_summary.json')
    a=p.parse_args()
    metrics=['unique_valid','valid_rate','coverage','success','mode_exclusion_survival','generation_ms_per_set']
    result={'training_seeds':a.seeds,'candidate_budget':3,'summary':{}}
    for kind in ['independent','set_diffusion','regressor']:
        runs=[]
        for seed in a.seeds:
            root=Path(a.runs)/(kind+'_seed'+str(seed))
            runs.append(json.loads((root/'evaluation/metrics.json').read_text(encoding='utf-8')))
        result['summary'][kind]={}
        for split in ['test','ood']:
            result['summary'][kind][split]={}
            for metric in metrics:
                values=[x[split]['metrics'][metric] for x in runs]
                result['summary'][kind][split][metric]={'mean':float(np.mean(values)),
                                                       'std_across_seeds':float(np.std(values,ddof=1)) if len(values)>1 else None,
                                                       'values':values}
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
