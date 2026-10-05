"""Run the public observation-only CLI and compare its actual saved output."""
import argparse
import subprocess
import sys
import time
import numpy as np
from scripts.run_observed_probability import read,write,lines,sha
from scripts.paired_modes_data import DATA
from scripts.run_factored_q import RUN


def main(seed,version='deployment_v2'):
    deployment=RUN/('conditional_g%d_s0'%seed)/version
    manifest=read(deployment/'manifest.json');identifier=manifest['example_id']
    # The registry supplies only an observed-state file pointer to the CLI.
    row=next(r for r in lines(DATA/'export/observations.jsonl') if r['id']==identifier)
    record=next(r for r in lines(DATA/'export/supervision.jsonl') if r['id']==identifier)
    assert row['split']==record['split']=='DEV_MODEL'
    output=deployment.parent/('public_cli_'+version);output.mkdir(exist_ok=False)
    command=[sys.executable,'-m','scripts.predict_factored_planner','--bundle',str(deployment/'planner.pt'),
        '--manifest',str(DATA/'export/observations.jsonl'),'--id',identifier,'--observation',record['observation'],
        '--qwen-cache',str(DATA/'export/qwen_cache'),'--output',str(output/'prediction.json'),'--k','4']
    write(output/'command.json',command);tic=time.monotonic();subprocess.run(command,check=True)
    result=read(output/'prediction.json')
    with np.load(deployment/'example.npz') as a:
        for key in ('paths','events','q_task','q_feas','q','selected_indices'):np.testing.assert_array_equal(np.asarray(result[key]),a[key])
    assert 'pi' not in result and result['internal_candidates']==8 and result['returned_candidates']==4
    write(output/'receipt.json',dict(example_id=identifier,exact_saved_pool_and_package_match=True,
        actual_public_cli=True,complete_path_states=8,new_qwen_requests=0,
        wall_seconds_including_process_and_head_load=time.monotonic()-tic,
        prediction_sha256=sha(output/'prediction.json'),bundle_sha256=sha(deployment/'planner.pt'),
        observed_input_sha256=sha(record['observation']),
        scope='Actual genuine cached-Qwen inference; CLI receives no target/box/reference labels. Not a fresh end-to-end Qwen latency measurement.'))


if __name__=='__main__':
    main(0)
