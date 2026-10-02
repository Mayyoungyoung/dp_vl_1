"""Compare the completed same-initialization online RGB-D frozen/LoRA pair."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records={}
    configs={}
    predictions={}
    for arm in ('frozen','lora'):
        run=args.runs/(arm+'_seed0')
        config=json.loads((run/'config.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        status=json.loads((run/'status.json').read_text())
        assert status==dict(status='completed',step=1000,exit_code=0)
        assert summary['online_train_requests']==4000 and summary['trajectory_exposures']==16000
        configs[arm]=config
        with np.load(run/'dev_model/predictions.npz',allow_pickle=False) as archive:
            predictions[arm]={key:archive[key].copy() for key in archive.files}
        records[arm]=dict(best_step=summary['best_step'],best=summary['metrics'],last=summary['last_metrics'],
                          elapsed_s=summary['elapsed_s'],setup_seconds_total=summary['setup_seconds_total'],
                          gpu_hours_reserved=summary['gpu_hours_reserved'],peak_cuda_memory_mb=summary['peak_cuda_memory_mb'],
                          online_train_requests=summary['online_train_requests'],additional_trajectory_exposures=summary['trajectory_exposures'],
                          head_parameters=summary['head_parameters'],lora_parameters=summary['lora_parameters'],
                          last_adapter_tensors_updated=sum(value['changed_from_initial'] for value in summary['adapter_last'].values()),
                          best_adapter_tensors_updated=sum(value['changed_from_initial'] for value in summary['adapter_best'].values()),
                          last_adapter_tensors_with_gradient=sum(value['ever_nonzero_gradient'] for value in summary['adapter_last'].values()),
                          artifact_sha256={str(path.relative_to(run)):sha(path) for path in run.rglob('*') if path.is_file() and path.suffix in ('.json','.pt','.npz')})
    shared=('head_init_sha256','initial_head_state_sha256','dataset_fingerprint','seed','steps','batch_size','accumulation',
            'lr','lora_lr','grounding_weight','grounding_sigma','horizon','candidates','width','depth','point_width',
            'pixel_stride','endpoint_residual_bound','pooling','max_pixels','event_scale','model_revision','processor_revision')
    assert all(configs['frozen'][key]==configs['lora'][key] for key in shared)
    assert np.array_equal(predictions['frozen']['scene_ids'],predictions['lora']['scene_ids'])
    best_difference=float(np.max(np.abs(predictions['frozen']['paths']-predictions['lora']['paths'])))
    result=dict(status='both completed',comparison='same pretrained RGB-D head, online frozen versus real last-two-layer q/v LoRA',
                shared_settings_verified={key:configs['frozen'][key] for key in shared},
                common_pretraining=json.loads((args.runs/'frozen_seed0/summary.json').read_text())['common_pretraining'],
                best_prediction_max_abs_difference_m=best_difference,records=records,
                conclusion='both DEV-ADE selections retain step0; true adapter updates do not establish useful fine tuning',
                protocol='observation_eval_v2: 24 semantic inputs and 23 positive-reference inputs',
                script_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(best_prediction_max_abs_difference_m=best_difference,
                         results={arm:dict(best_step=row['best_step'],best_semantic=row['best']['semantic_goal_accuracy'],
                                           last_semantic=row['last']['semantic_goal_accuracy'],last_adapter_tensors_updated=row['last_adapter_tensors_updated'],
                                           online_request_ms_median=row['best']['online_request_ms_median']) for arm,row in records.items()})))


if __name__=='__main__':
    main()
