"""CPU reconstruction of recorded ordinary multi-task initialization.

No historical initial checkpoint was saved. This verifies the recorded seeds,
runtime initialization and immutable source equivalence, not a nonexistent
initial artifact. No observation or trajectory content is read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from routeset.common import seed_all,sha256,write_json
from routeset.observed_geometry import ObservedGeometryRouteHead


SOURCES=('routeset/observed_geometry.py','routeset/observed_route_head.py','routeset/observed_multitask.py',
         'scripts/train_observed_geometry.py','scripts/train_observed_routes.py','routeset/train_v2.py',
         'routeset/common.py','routeset/models.py')
KEYS=('feature_dim','horizon','candidates','width','depth','point_width','anchor_mode','endpoint_mode',
      'refinement_mode','seed','lr','steps','batch_size','eval_every','sampling_mode','metric_aggregation','grounding_weight',
      'pooling','geometry_pooling','pixel_stride','endpoint_residual_bound','grounding_sigma','event_scale',
      'checkpoint_selection','checkpoint_selection_protocol')


def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(project_root,runs,output):
    project_root,output=Path(project_root).resolve(),Path(output)
    if output.exists():raise FileExistsError('Fresh initialization proof output required')
    if len(runs)!=2:raise ValueError('Exactly the two predeclared training runs are required')
    torch.set_num_threads(1);runtime=Path(__file__).resolve().parents[1]
    runtime_hashes={name:sha256(runtime/name) for name in SOURCES}
    records=[];reference=None
    for run_path in runs:
        run_path=Path(run_path).resolve();config=json.loads((run_path/'config.json').read_text())
        version=config['code_commit']
        if len(version)!=40 or any(c not in '0123456789abcdef' for c in version):raise ValueError('Immutable source commit required')
        source=project_root/'research_v2/releases'/version
        hashes={name:sha256(source/name) for name in SOURCES}
        if hashes!=runtime_hashes:raise ValueError('Model/initialization code differs between recorded source and evaluator')
        if config['source_script_sha256']!=hashes['scripts/train_observed_geometry.py']:raise ValueError('Recorded actual trainer source SHA differs')
        selected={key:config[key] for key in KEYS}
        if reference is not None and selected!=reference:raise ValueError('Paired architecture/seed/exposure options differ')
        reference=selected
        if config['endpoint_mode']!='free_offset' or config['refinement_mode']!='none':raise ValueError('Ordinary free endpoint heads only')
        seed_all(config['seed'])
        model=ObservedGeometryRouteHead(config['feature_dim'],config['horizon'],config['candidates'],config['width'],
            config['depth'],config['point_width'],config['endpoint_residual_bound'],
            geometry_pooling=config['geometry_pooling'],anchor_mode=config['anchor_mode'],endpoint_mode=config['endpoint_mode'])
        records.append(dict(run=str(run_path),source_commit=version,config_sha256=sha256(run_path/'config.json'),
            reconstructed_initial_state_sha256=state_digest(model.state_dict()),parameters=model.active_parameter_count(),
            post_initialization_torch_rng_sha256=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
            actual_source_sha256=hashes,initial_checkpoint_was_saved=False))
    if records[0]['reconstructed_initial_state_sha256']!=records[1]['reconstructed_initial_state_sha256']:
        raise ValueError('Reconstructed initialization differs')
    if records[0]['post_initialization_torch_rng_sha256']!=records[1]['post_initialization_torch_rng_sha256']:
        raise ValueError('Reconstructed initialization RNG consumption differs')
    output.mkdir(parents=True)
    result=dict(status='passed',records=records,matched_options=reference,torch_version=torch.__version__,
        source_script_sha256=sha256(__file__),proof_scope='CPU reconstruction under actual equivalent source and recorded seeds; no saved initial checkpoint exists; observations and reference content are not read')
    write_json(output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--run',action='append',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    run(args.project_root,args.run,args.output)
