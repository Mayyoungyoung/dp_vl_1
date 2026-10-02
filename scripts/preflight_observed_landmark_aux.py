"""CPU actual-data initialization/sampler audit before the optional auxiliary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from routeset.common import seed_all,sha256,write_json
from routeset.observed_geometry import ObservedGeometryRouteHead
from routeset.observed_grounding_targets import prepare_event_grounding_targets
from routeset.observed_multitask import check_multitask_model_gate,draw_observation_batch
from routeset.observed_route_head import load_observed_dataset
from scripts.train_observed_geometry import load_geometry,batch_inputs
from scripts.check_multitask_initialization import state_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ordinary-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh preflight output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU audit must hide GPUs')
    torch.set_num_threads(1);started=time.perf_counter();runtime=Path(__file__).resolve().parents[1]
    config=json.loads((args.ordinary_run/'config.json').read_text());summary=json.loads((args.ordinary_run/'summary.json').read_text())
    if (config['endpoint_mode'],config['grounding_weight'],config['refinement_mode'],config['steps'],config['batch_size'],config['seed'])!=('free_offset',0.,'none',1500,32,0):raise ValueError('Registered ordinary control required')
    if sha256(args.ordinary_run/'last.pt')!=summary['last_checkpoint_sha256']:raise ValueError('Actual completed control checkpoint changed')
    source=args.ordinary_run.parents[2]/'research_v2/releases'/config['code_commit']
    common=('routeset/observed_geometry.py','routeset/observed_route_head.py','routeset/observed_multitask.py','routeset/common.py','routeset/models.py','routeset/train_v2.py','scripts/train_observed_routes.py')
    common_hashes={name:sha256(source/name) for name in common}
    if any(sha256(runtime/name)!=value for name,value in common_hashes.items()):raise ValueError('Model/sampler source changed relative to ordinary control')
    if sha256(source/'scripts/train_observed_geometry.py')!=config['source_script_sha256']:raise ValueError('Original trainer source provenance changed')
    gate=check_multitask_model_gate(config['observations'],config['supervision'],config['multitask_snapshot_manifest'])
    data=load_observed_dataset(config['observations'],config['supervision'],config['cache_dir'],config['horizon'],config['pooling'])
    geometry=load_geometry(data,config['observations'],config['supervision'],config['pixel_stride'])
    if geometry['fingerprint']!=config['dataset_fingerprint']:raise ValueError('Input/reference data differ from ordinary control')
    ids=np.flatnonzero((data['splits']=='TRAIN')&data['path_mask'].any(1))
    def model():return ObservedGeometryRouteHead(config['feature_dim'],config['horizon'],config['candidates'],config['width'],config['depth'],config['point_width'],config['endpoint_residual_bound'],anchor_mode=config['anchor_mode'],endpoint_mode=config['endpoint_mode']).eval()
    seed_all(config['seed']);ordinary=model();ordinary_rng=torch.get_rng_state().clone()
    seed_all(config['seed']);before=torch.get_rng_state().clone();targets,metadata=prepare_event_grounding_targets(data,geometry,ids)
    if not torch.equal(before,torch.get_rng_state()):raise ValueError('Target preparation consumed model RNG')
    auxiliary=model()
    initial=state_digest(ordinary.state_dict())
    if state_digest(auxiliary.state_dict())!=initial or not torch.equal(ordinary_rng,torch.get_rng_state()):raise ValueError('Reconstructed parameter/RNG initialization changed')
    with torch.no_grad():
        inputs=batch_inputs(data,geometry,ids[:1],'cpu');a=ordinary(**inputs);b=auxiliary(**inputs)
    if not all(torch.equal(a[i],b[i]) for i in (0,1)) or not all(torch.equal(a[2][key],b[2][key]) for key in a[2]):raise ValueError('Initial actual observed-input forward changed')
    sampler=np.random.default_rng(config['seed']+100000);sequence=hashlib.sha256()
    for _ in range(config['steps']):
        selected=draw_observation_batch(data,ids,sampler,config['batch_size'],config['sampling_mode']);sequence.update(np.asarray(selected,dtype='<i8').tobytes())
    checkpoint=torch.load(args.ordinary_run/'last.pt',map_location='cpu',weights_only=False)
    if sampler.bit_generator.state!=checkpoint['sampler_state']:raise ValueError('Full registered TRAIN sampling stream state differs from actual ordinary checkpoint')
    if checkpoint['trajectory_exposures']!=config['steps']*config['batch_size']*config['candidates']:raise ValueError('Ordinary exposure accounting changed')
    args.output.mkdir(parents=True)
    write_json(args.output/'grounding_target_selection.json',metadata)
    result=dict(status='passed',code_commit=os.environ.get('CODE_COMMIT'),elapsed_seconds=time.perf_counter()-started,cpu_threads=1,gpu_hours=0,
        ordinary_run=str(args.ordinary_run),ordinary_config_sha256=sha256(args.ordinary_run/'config.json'),ordinary_checkpoint_sha256=sha256(args.ordinary_run/'last.pt'),
        reconstructed_initial_state_sha256=initial,initial_forward_bit_equal=True,parameters=ordinary.active_parameter_count(),
        sampler_draws=config['steps']*config['batch_size'],sampler_sequence_sha256=sequence.hexdigest(),sampler_final_state_equals_completed_control=True,
        grounding_target_fingerprint=metadata['target_fingerprint'],target_selection_sha256=sha256(args.output/'grounding_target_selection.json'),
        positive_train_observations=len(ids),positive_reference_observation_slots=metadata['positive_reference_observation_slots'],unique_positive_reference_routes=metadata['unique_parent_reference_slots'],
        common_immutable_source_sha256=common_hashes,actual_preflight_sha256=sha256(__file__),actual_target_helper_sha256=sha256(runtime/'routeset/observed_grounding_targets.py'),
        actual_new_trainer_sha256=sha256(runtime/'scripts/train_observed_geometry.py'),current_model_use_gate=gate,
        scope='Actual cached Qwen/RGB-D input with CPU reconstruction and full sampled-index replay. No saved historical initial checkpoint exists. Target computation only reads positive TRAIN arrays; no auxiliary training, semantic/contact/collision or execution certification.')
    write_json(args.output/'summary.json',result);print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','parameters','positive_train_observations','unique_positive_reference_routes','sampler_draws')}))


if __name__=='__main__':main()
