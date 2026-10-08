"""Run a new public process on the first registered DEV request and compare."""
import argparse
import subprocess
import sys
import time
import numpy as np
from scripts.run_observed_probability import read,write,sha,lines
from scripts.paired_modes_data import DATA,RUN as OLD_RUN
from scripts.research_v3_frequency import RUN
from scripts.analyze_paired_selection import select


def main(output):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    arm='frequency_set_matching';checkpoint=RUN/arm/'last.pt'
    bundle=OLD_RUN/'reliability/R1_seed0/deployment_seed0/planner.pt'
    row=next(r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL')
    label=next(r for r in lines(DATA/'export/supervision.jsonl') if r['id']==row['id'])
    cmd=[sys.executable,'-m','scripts.predict_research_v3','--bundle',str(bundle),'--checkpoint',str(checkpoint),
        '--manifest',str(DATA/'export/observations.jsonl'),'--id',row['id'],'--observation',label['observation'],
        '--qwen-cache',str(DATA/'export/qwen_cache'),'--output',str(out/'prediction.json'),'--k','4']
    start=time.monotonic();subprocess.run(cmd,check=True);elapsed=time.monotonic()-start
    pred=read(out/'prediction.json')
    pool=RUN/arm/'evaluation_fixed_q_v2/pool.npz'
    with np.load(pool) as z:
        j=list(z['ids']).index(row['id'])
        for key in ('paths','events','q'):np.testing.assert_array_equal(pred[key],z[key][j])
        selected=select(z['paths'][j],z['q'][j],4)
        np.testing.assert_array_equal(pred['selected_indices'],selected)
        np.testing.assert_array_equal(pred['selected_paths'],z['paths'][j,selected])
    write(out/'receipt.json',dict(command=cmd,elapsed_seconds=elapsed,exact=True,request=row['id'],
        generator_sha256=sha(checkpoint),scorer_sha256=sha(bundle),pool_sha256=sha(pool),prediction_sha256=sha(out/'prediction.json'),
        timing_scope='New Python process and head/score inference with genuine cached Qwen; not end-to-end online VLM latency'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
