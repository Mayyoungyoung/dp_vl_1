"""Three explicit ordinary-head continuations; only B/C's path cost differs.

The original model/Adam/scheduler/all RNG states are restored, never reset.
Train, CPU parent inspection and fixed-last TRAIN evaluation are separate CLI
stages. Interrupted issued work is never replayed behind a checkpoint.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
PROCESS_STARTED=time.perf_counter()
import numpy as np

from routeset.observed_qwen_continuation import (RequestJournal,atomic_torch_save,canonical,
    composite_evaluator,digest,exclusive_lock,write_json)
from routeset.observed_training_audit import append_indices,tensor_state_digest
from scripts import train_observed_two_row_diffusion as shared
from scripts.train_observed_diffusion_minsnr_probe import nested_digest,read_attempt_cost

PROJECT=Path(__file__).resolve().parents[1]
PROTOCOL='observed_ordered_relation_continuation_v1'
CONFIG='configs/observed_ordered_relation_training_v1.json'
SOURCES=tuple(dict.fromkeys(shared.SOURCE_FILES+(
    'scripts/train_observed_ordered_relation.py',CONFIG,'routeset/observed_ordered_relation_loss.py',
    'configs/observed_ordered_relation_v1.json','scripts/train_observed_diffusion_minsnr_probe.py',
    'routeset/observed_multitask.py','routeset/observed_path_refinement.py',
    'scripts/audit_two_row_train_reference_quality.py','scripts/export_observation_roles.py',
    'scripts/observation_cache_qwen.py')))
POOL_FILES={'predictions.npz','generation.json','metrics.json','per_scene.json'}
PARENT_FIELDS={'model','optimizer','scheduler','scaler','step','config','rng','sampler_state',
    'best','history','elapsed_s','trajectory_exposures','sample_stream_audit'}
COMPONENT_FIELDS=('model','optimizer','scheduler','rng','sampler_state','sample_stream_audit')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def policy():
    p=read(PROJECT/CONFIG)
    fixed=dict(protocol=PROTOCOL,parent_step=12000,steps=3000,batch_size=32,candidates=4,horizon=24,
        lr=.0003,weight_decay=.0001,gradient_clip=1.,event_scale=.2,grounding_weight=.02,
        grounding_sigma=.025,eval_every=250,checkpoint_every=25,train_inputs=285,train_references=1663,
        dev_inputs=36,additional_input_draws=96000,additional_path_states=384000,
        dev_selection_opportunities=12,fixed_last_train_calls=285,wall_cap_seconds_per_arm=4500.,
        reset_optimizer=False,new_dev_or_reserved_allowed=False,automatic_stage_chaining=False)
    if any(p.get(k)!=v for k,v in fixed.items()) or p['arms']!=['original_saturation','xyz_divergence','observed_divergence']:
        raise ValueError('Only the one fixed ordered-relation continuation is supported')
    return p


def check_parent(state,cfg,summary):
    if (not PARENT_FIELDS.issubset(state) or state['step']!=12000 or state['config']!=cfg or
            state['scaler'] is not None or state['trajectory_exposures']!=1536000 or
            state['sample_stream_audit']!=summary['sample_stream_audit'] or
            state['scheduler'].get('last_epoch')!=12000 or len(state['history'])!=48 or
            set(state['rng'])!={'numpy_generator','numpy','python','torch','cuda'}):
        raise ValueError('Actual ordinary parent schema/config/budget differs; no invented diffusion streams')
    for key,value in dict(lr=.0003,batch_size=32,candidates=4,horizon=24,steps=12000,
            sampling_mode='uniform',objective='saturation',grounding_weight=.02,grounding_sigma=.025,
            event_scale=.2,refinement_mode='none',anchor_mode='straight_through_peak',
            endpoint_mode='surface_anchor').items():
        if cfg.get(key)!=value:raise ValueError('Ordinary parent policy changed: '+key)
    return {key:nested_digest(state[key]) for key in COMPONENT_FIELDS}


def parent_metadata(folder,p):
    folder=Path(folder)
    for name,key in [('last.pt','parent_checkpoint_sha256'),('config.json','parent_config_sha256'),
            ('summary.json','parent_summary_sha256'),('composite_training_receipt.json','parent_receipt_sha256'),
            ('source_hashes.json','parent_source_index_sha256')]:
        if digest(folder/name)!=p[key]:raise ValueError('Fixed ordinary parent artifact differs: '+name)
    cfg,summary=read(folder/'config.json'),read(folder/'summary.json')
    if summary['last_checkpoint_sha256']!=p['parent_checkpoint_sha256'] or summary['last_step']!=12000:
        raise ValueError('Original actual last12000 required')
    for name,value in cfg['composite_source_sha256'].items():
        if digest(PROJECT/name)!=value:raise ValueError('Historical source bytes changed: '+name)
    return cfg,summary


def math_gate(folder,p):
    folder=Path(folder)
    for stage in ('tests','bench'):
        status=read(folder/'job_records'/(stage+'.status.json'));receipt=read(folder/(stage+'_receipt.json'))
        if (status.get('status')!='completed' or status.get('exit_code')!=0 or
                status.get('code_commit')!=p['math_source_commit'] or receipt.get('status')!='completed' or
                receipt.get('source_commit')!=p['math_source_commit']):
            raise ValueError('Actual frozen math tests and resource benchmark required')
        if not receipt.get('artifacts'):raise ValueError('Math gate has no sealed artifacts')
        for name,value in receipt['artifacts'].items():
            path=folder/name;path.resolve().relative_to(folder.resolve())
            if digest(path)!=value:raise ValueError('Math gate artifact changed')
    test=read(folder/'tests_receipt.json');bench=read(folder/'benchmark.json')
    if (test.get('passed')!=20 or test.get('skipped')!=0 or bench['completed_loss_calls']!=15 or
            bench['model_forwards']!=0 or bench['settings']['observed_points']!=12544 or
            any(not c['finite_loss_and_gradients'] for arm in bench['arms'].values() for c in arm['all_calls'])):
        raise ValueError('Complete20/0skip + all15 finite full-size benchmark required')
    if (digest(PROJECT/'routeset/observed_ordered_relation_loss.py')!=p['math_module_sha256'] or
            digest(PROJECT/'configs/observed_ordered_relation_v1.json')!=p['math_config_sha256']):
        raise ValueError('The measured loss mathematics/config changed')
    return {name:digest(folder/name) for name in ('tests_receipt.json','bench_receipt.json','benchmark.json')}


def inspect_parent(args):
    import torch
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available() or
            not hasattr(os,'sched_getaffinity') or sorted(os.sched_getaffinity(0))!=[0] or
            os.environ.get('CODE_COMMIT')!=PROJECT.name or torch.__version__.split('+')[0]!='2.4.1'):
        raise ValueError('Inspection requires immutable source, CPU0/Torch2.4.1 and GPU hidden')
    torch.set_num_threads(1);p=policy();out=Path(args.output)
    if out.exists():raise FileExistsError('Fresh CPU inspection output required')
    cfg,summary=parent_metadata(args.parent_run,p);state=torch.load(Path(args.parent_run)/'last.pt',map_location='cpu',weights_only=False)
    components=check_parent(state,cfg,summary)
    write_json(out,dict(protocol=PROTOCOL,status='completed',parent_checkpoint_sha256=p['parent_checkpoint_sha256'],
        actual_keys=sorted(state),components_sha256=components,optimizer_entries=len(state['optimizer']['state']),
        scheduler_keys=sorted(state['scheduler']),scheduler_last_epoch=state['scheduler']['last_epoch'],
        rng_keys=sorted(state['rng']),sampler_state=state['sampler_state'],
        historical_cost_seconds=summary['elapsed_s'],model_calls=0,dataset_reads=0,
        source_sha256={name:digest(PROJECT/name) for name in SOURCES}))


def draw_plan(parent,ids,steps,batch):
    """Clone the saved sampler, not any training/global RNG. No new positives."""
    sampler=np.random.default_rng(0);sampler.bit_generator.state=copy.deepcopy(parent['sampler_state'])
    draws=np.stack([sampler.choice(ids,batch,replace=True) for _ in range(steps)])
    audit=copy.deepcopy(parent['sample_stream_audit'])
    for row in draws:audit=append_indices(audit,row)
    body=dict(ids=list(map(int,ids)),steps=steps,batch_size=batch,draw_sha256=shared.array_digest(draws),
        initial_sampler_sha256=nested_digest(parent['sampler_state']),final_sampler_sha256=nested_digest(sampler.bit_generator.state),
        final_index_chain_sha256=audit['index_chain_sha256'])
    return draws,dict(body,sha256=hashlib.sha256(canonical(body)).hexdigest())


class Clock:
    def __init__(self,prior,limit=4500.,start=None,now=time.perf_counter):
        if not math.isfinite(prior) or prior<0:raise ValueError('Known actual prior process cost required')
        self.prior,self.limit,self.now,self.start=prior,limit,now,now() if start is None else start
    def elapsed(self):return self.prior+self.now()-self.start
    def check(self):
        if self.elapsed()>=self.limit:raise RuntimeError('Cumulative per-arm4500s boundary exhausted; no automatic extension')


class BoundedJournal(RequestJournal):
    def __init__(self,path,clock):super().__init__(path);self.clock=clock
    def issue(self,kind,key):self.clock.check();super().issue(kind,key)


class OrderedTrainer:
    """Actual B32 update and checkpoint state; synthetic tiny tests use same loop."""
    def __init__(self,model,config,parent,ids):
        import torch
        from routeset.train_v2 import restore_rng
        self.model,self.config=model,config
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda _:1.)
        self.rng=np.random.default_rng(0);self.sampler=np.random.default_rng(0)
        self.draws,self.plan=draw_plan(parent,ids,config['steps'],config['batch_size'])
        self.parent_components={key:nested_digest(parent[key]) for key in COMPONENT_FIELDS}
        self.initial_sampler_state=copy.deepcopy(parent['sampler_state'])
        self.initial_audit=copy.deepcopy(parent['sample_stream_audit'])
        self.model.load_state_dict(parent['model'],strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(parent['optimizer']));self.scheduler.load_state_dict(copy.deepcopy(parent['scheduler']))
        self.sampler.bit_generator.state=copy.deepcopy(parent['sampler_state']);restore_rng(copy.deepcopy(parent['rng']),self.rng)
        self.audit=copy.deepcopy(parent['sample_stream_audit']);self.step=0;self.history=[];self.best=None
        self.gradient_audit={};self.first_gradient_audits={};self.last_loss=None
        self.positive_chain=hashlib.sha256(PROTOCOL.encode()).hexdigest();self.ids=np.asarray(ids)
        self._validate_scheduler()
        restored=self.components()
        if restored!=self.parent_components:raise ValueError('Not all actual parent model/Adam/scheduler/sampler/RNG bytes restored')
    def _validate_scheduler(self):
        if (self.scheduler.last_epoch!=self.config['parent_step']+self.step or self.scheduler.base_lrs!=[self.config['lr']] or
                any(g['lr']!=self.config['lr'] or g['weight_decay']!=self.config['weight_decay'] for g in self.optimizer.param_groups)):
            raise ValueError('Restored actual constant scheduler/Adam policy differs')
    def validate_draw_prefix(self):
        sampler=np.random.default_rng(0);sampler.bit_generator.state=copy.deepcopy(self.initial_sampler_state)
        audit=copy.deepcopy(self.initial_audit)
        for expected in self.draws[:self.step]:
            actual=sampler.choice(self.ids,self.config['batch_size'],replace=True)
            if not np.array_equal(actual,expected):raise ValueError('Stored draw plan differs from cloned parent')
            audit=append_indices(audit,actual)
        if audit!=self.audit or nested_digest(sampler.bit_generator.state)!=nested_digest(self.sampler.bit_generator.state):
            raise ValueError('Restored sampler or actual draw-chain prefix differs')
    def validate_positive_prefix(self,hashes):
        value=hashlib.sha256(PROTOCOL.encode()).hexdigest()
        for ids in self.draws[:self.step]:
            value=hashlib.sha256(bytes.fromhex(value)+canonical([hashes[int(i)] for i in ids])).hexdigest()
        if value!=self.positive_chain:raise ValueError('Full-positive reference exposure prefix differs')
    def components(self):
        from routeset.train_v2 import rng_state
        return {key:nested_digest(value) for key,value in dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),rng=rng_state(self.rng),sampler_state=self.sampler.bit_generator.state,
            sample_stream_audit=self.audit).items()}
    def train_step(self,loss_function,journal,positive_hashes):
        import torch
        if self.step>=self.config['steps']:raise ValueError('Update budget exhausted')
        step=self.step+1;self.model.train()
        ids=self.sampler.choice(self.ids,self.config['batch_size'],replace=True)
        if not np.array_equal(ids,self.draws[self.step]):raise ValueError('Actual draw differs from cloned parent stream')
        self.audit=append_indices(self.audit,ids)
        self.positive_chain=hashlib.sha256(bytes.fromhex(self.positive_chain)+canonical([positive_hashes[int(i)] for i in ids])).hexdigest()
        loss,details=loss_function(ids,step,journal,self.rng)
        if loss.ndim or not bool(torch.isfinite(loss)):raise ValueError('Finite scalar training loss required')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        if step<=2 or step%self.config['checkpoint_every']==0:
            self.gradient_audit={n:dict(present=p.grad is not None,finite=p.grad is None or bool(torch.isfinite(p.grad).all()),
                nonzero=p.grad is not None and bool(p.grad.count_nonzero()),norm=None if p.grad is None else float(p.grad.double().norm()))
                for n,p in self.model.named_parameters()}
            if step<=2:self.first_gradient_audits[str(step)]=copy.deepcopy(self.gradient_audit)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['gradient_clip'],error_if_nonfinite=True)
        journal.issue('optimizer',step);self.optimizer.step();self.scheduler.step();self.step=step
        self.optimizer.zero_grad(set_to_none=True)
        self.last_loss=dict(step=step,global_step=self.config['parent_step']+step,loss=float(loss.detach()),
            gradient_norm=float(norm),**details)
        return self.last_loss
    def state_dict(self,journal,elapsed):
        from routeset.qwen_prefix_replay import cpu_copy
        from routeset.train_v2 import rng_state
        return dict(protocol=PROTOCOL,config=self.config,model=cpu_copy(self.model.state_dict()),
            optimizer=cpu_copy(self.optimizer.state_dict()),scheduler=copy.deepcopy(self.scheduler.state_dict()),
            rng=rng_state(self.rng),sampler_state=copy.deepcopy(self.sampler.bit_generator.state),
            sample_stream_audit=copy.deepcopy(self.audit),step=self.step,history=copy.deepcopy(self.history),best=copy.deepcopy(self.best),
            gradient_audit=copy.deepcopy(self.gradient_audit),first_gradient_audits=copy.deepcopy(self.first_gradient_audits),
            last_loss=copy.deepcopy(self.last_loss),plan=self.plan,parent_components=self.parent_components,
            positive_chain_sha256=self.positive_chain,journal=journal.snapshot(),pending_gradients=False,
            additional_input_draws=self.step*self.config['batch_size'],
            additional_path_states=self.step*self.config['batch_size']*self.config['candidates'],elapsed_seconds=elapsed)
    def load_state_dict(self,state,journal,allow_sealed_evaluation=False):
        from routeset.train_v2 import restore_rng
        if (state['protocol']!=PROTOCOL or state['config']!=self.config or state['plan']!=self.plan or
                state['parent_components']!=self.parent_components or state['pending_gradients'] is not False or
                not 0<=state['step']<=self.config['steps'] or
                state['additional_input_draws']!=state['step']*self.config['batch_size'] or
                state['additional_path_states']!=state['additional_input_draws']*self.config['candidates']):
            raise ValueError('Incomplete or changed continuation checkpoint')
        if not allow_sealed_evaluation:journal.require_boundary(state['journal'])
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.scheduler.load_state_dict(state['scheduler']);self.sampler.bit_generator.state=copy.deepcopy(state['sampler_state'])
        restore_rng(state['rng'],self.rng);self.step=state['step'];self.audit=copy.deepcopy(state['sample_stream_audit'])
        self.history=copy.deepcopy(state['history']);self.best=copy.deepcopy(state['best'])
        self.gradient_audit=copy.deepcopy(state['gradient_audit']);self.first_gradient_audits=copy.deepcopy(state['first_gradient_audits'])
        self.last_loss=copy.deepcopy(state['last_loss']);self.positive_chain=state['positive_chain_sha256'];self._validate_scheduler()
        if (self.audit['batches']!=self.config['parent_step']+self.step or
                self.audit['observation_draws']!=(self.config['parent_step']+self.step)*self.config['batch_size']):
            raise ValueError('Sampler audit exposure differs')
        self.validate_draw_prefix()


def build_reference_cache(data,geometry,ids,out,journal,device):
    import torch
    from scripts import train_observed_geometry as base
    from routeset.observed_ordered_relation_loss import prepare_reference_descriptors
    out=Path(out);receipt_path=out/'reference_cache.json';tensor_path=out/'reference_cache.pt'
    if receipt_path.exists():
        receipt=read(receipt_path)
        if (receipt['dataset_fingerprint']!=data['fingerprint'] or receipt['ids']!=list(map(int,ids)) or
                digest(tensor_path)!=receipt['sha256']):raise ValueError('Reference cache identity changed')
        return torch.load(tensor_path,map_location='cpu',weights_only=False),receipt
    if tensor_path.exists():raise ValueError('Unsealed reference preparation preserved; no replay')
    started=time.perf_counter();cache={}
    from routeset.qwen_prefix_replay import cpu_copy
    for index in ids:
        journal.issue('reference_descriptor',str(data['scene_ids'][index]))
        inputs=base.batch_inputs(data,geometry,np.asarray([index]),device)
        cache[int(index)]=cpu_copy(prepare_reference_descriptors(torch.as_tensor(data['paths'][index:index+1],device=device),
            torch.as_tensor(data['path_mask'][index:index+1],device=device),{k:inputs[k] for k in ('world_xyz','valid_mask')}))
    atomic_torch_save(tensor_path,cache)
    receipt=dict(dataset_fingerprint=data['fingerprint'],ids=list(map(int,ids)),sha256=digest(tensor_path),
        model_calls=0,predicted_descriptors_cached=False,seconds=time.perf_counter()-started,journal=journal.snapshot())
    write_json(receipt_path,receipt);return cache,receipt


@contextmanager
def observe_alignment_terms(mask,output):
    """Observe the three existing DP calls; never alter values or autograd."""
    from routeset import observed_ordered_relation_loss as relation
    original=relation.soft_dtw;calls=[]
    def observed(cost,gamma):
        value=original(cost,gamma);position=len(calls)
        if position==0:selected=value[mask[:,None,:].expand_as(value)]
        elif position==1:
            expanded=value[:,:,None].expand(len(mask),value.shape[1],mask.shape[1])
            selected=expanded[mask[:,None,:].expand_as(expanded)]
        elif position==2:selected=value[mask]
        else:raise ValueError('Frozen loss unexpectedly added a DP call')
        calls.append(float(selected.detach().mean()))
        return value
    relation.soft_dtw=observed
    try:yield
    finally:relation.soft_dtw=original
    if len(calls)!=3:raise ValueError('Frozen XY/XX/YY DP call order changed')
    factor=.75*.1**2/23
    output.update(dp_calls_observed=3,extra_dp_calls=0,
        all_positive_pair_xy_mean=calls[0],all_positive_pair_xx_mean=calls[1],all_positive_reference_yy_mean=calls[2],
        normalized_all_positive_shape_mean=factor*(calls[0]-.5*calls[1]-.5*calls[2]),
        component_scope='Unmatched all-valid-pair means; C shape includes observed relation. These are not separately matched loss terms.')


def training_loss(trainer,data,geometry,device,cache):
    import torch
    from scripts import train_observed_geometry as base
    from routeset.train_v2 import positive_assignment_loss
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.observed_ordered_relation_loss import ordered_relation_costs,saturation_loss_from_costs,batch_reference_descriptors
    arm=trainer.config['arm']
    def loss(ids,step,journal,rng):
        inputs=base.batch_inputs(data,geometry,ids,device)
        journal.issue('train_model',step);xyz,opened,details=trainer.model(**inputs)
        target=torch.as_tensor(data['paths'][ids],device=device);events=torch.as_tensor(data['events'][ids],device=device)
        mask=torch.as_tensor(data['path_mask'][ids],device=device);metadata={}
        if arm=='original_saturation':
            pred=torch.cat([xyz[:,:,1:],opened[:,:,1:,None]*.2],-1)
            reference=torch.cat([target[:,:,1:],events[:,:,1:,None]*.2],-1)
            route=positive_assignment_loss(pred,reference,data['path_mask'][ids],'saturation',rng)
        else:
            kwargs={}
            if arm=='observed_divergence':
                observation={k:inputs[k] for k in ('world_xyz','valid_mask')}
                descriptor=batch_reference_descriptors([cache[int(i)] for i in ids],target,mask,observation)
                kwargs=dict(observation=observation,reference_descriptor=descriptor)
            with observe_alignment_terms(mask,metadata):
                costs,statistics=ordered_relation_costs(xyz,opened,target,events,mask,arm,**kwargs)
            metadata.update(statistics)
            route=saturation_loss_from_costs(costs,mask)
        # Pure detached scalar diagnostics: not extra matching, DP or model work.
        with torch.no_grad():
            valid_pairs=mask[:,None,:].expand(len(mask),xyz.shape[1],mask.shape[1])
            diagonal_xyz=(xyz[:,:,None,1:]-target[:,None,:,1:]).square().mean((-1,-2))
            fixed_event=((opened[:,:,None,1:]-events[:,None,:,1:])*.2).square().mean(-1)
            metadata.update(all_positive_diagonal_xyz_mse=float(diagonal_xyz[valid_pairs].mean()),
                all_positive_weighted_event_mse=float(.25*fixed_event[valid_pairs].mean()))
        ground=positive_endpoint_attention_loss(details['attention'],inputs['world_xyz'],inputs['valid_mask'],
            target[:,:,-1],mask,trainer.config['grounding_sigma'])
        return route+trainer.config['grounding_weight']*ground,dict(path_loss=float(route.detach()),
            grounding_loss=float(ground.detach()),relation_statistics=metadata)
    return loss


def pool_identity(trainer,ids,kind):
    return dict(protocol=PROTOCOL,config=trainer.config,step=trainer.step,kind=kind,
        model_sha256=tensor_state_digest(trainer.model.state_dict()),ids=list(map(str,ids)),
        requests=len(ids),candidates=4,path_states=4*len(ids))


def verify_pool(folder,identity):
    folder=Path(folder);receipt=read(folder/'pool_receipt.json');names=set(receipt['artifacts'])
    if (receipt['identity']!=identity or not POOL_FILES.issubset(names) or
            names-POOL_FILES-{'paired_language_predictions.npz'}):raise ValueError('Incomplete or changed sealed pool')
    for name,value in receipt['artifacts'].items():
        if Path(name).name!=name or digest(folder/name)!=value:raise ValueError('Sealed pool bytes changed')
    if read(folder/'metrics.json')!=receipt['metrics']:raise ValueError('Sealed metrics differ')
    return receipt


def reconcile_pool(journal,boundary,receipt):
    if receipt['journal_before']!=boundary or receipt['journal_after']!=journal.snapshot():
        raise ValueError('Not the exact complete sealed evaluation after checkpoint')
    identity=receipt['identity'];kind=identity['kind']
    expected=[(kind+'_model',str(identity['step'])+':'+i) for i in identity['ids']]
    actual=[(r['kind'],r['key']) for r in journal.records[boundary['records']:]]
    if actual!=expected:raise ValueError('Uncheckpointed work is not precisely this sealed pool')


def evaluate_pool(trainer,data,geometry,ids,data_root,folder,kind,journal,device):
    import torch
    from scripts import train_observed_geometry as base
    from scripts import train_observed_two_row as ordinary
    from scripts.export_two_row_composite_observations import verify_export
    folder=Path(folder);identity=pool_identity(trainer,data['scene_ids'][ids],kind)
    if folder.exists():return verify_pool(folder,identity)
    staging=folder.with_name(folder.name+'.staging')
    if staging.exists():raise ValueError('Incomplete issued pool preserved; no automatic regeneration')
    staging.mkdir(parents=True);before=journal.snapshot();started=shared.synchronize()
    paths=np.full((len(ids),4,24,3),np.nan,np.float32);events=np.full((len(ids),4,24),np.nan,np.float32)
    anchors=np.full((len(ids),3),np.nan,np.float32);generation=[]
    def seal():
        np.savez_compressed(staging/'predictions.npz',paths=paths,gripper_open=events,learned_surface_anchor=anchors,
            scene_ids=data['scene_ids'][ids],parent_ids=data['parent_ids'][ids])
    try:
        trainer.model.eval()
        with torch.no_grad(),shared.isolated_evaluation_rng(trainer):
            for position,index in enumerate(ids):
                inputs=base.batch_inputs(data,geometry,np.asarray([index]),device)
                begin=shared.synchronize();journal.issue(kind+'_model',str(trainer.step)+':'+str(data['scene_ids'][index]))
                xyz,opened,details=trainer.model(**inputs)
                paths[position]=xyz.cpu().numpy()[0];events[position]=opened.cpu().numpy()[0]
                anchors[position]=details['anchor_xyz'].cpu().numpy()[0]
                generation.append(dict(id=str(data['scene_ids'][index]),seconds=shared.synchronize()-begin,
                    finite_candidates=np.isfinite(paths[position]).all((1,2)).tolist()))
        seal();sealed=digest(staging/'predictions.npz')
        write_json(staging/'generation.json',dict(identity=identity,requests=generation,
            predictions_sha256=sealed,all_predictions_sealed_before_geometry_labels=True,new_qwen_calls=0))
        metrics,rows=ordinary.observation_metrics(paths,events,data,ids)
        control,arrays=ordinary.reused_language_control(paths,events,data,geometry,ids)
        metrics['paired_language_control']=control
        with composite_evaluator(ordinary,verify_export):
            ordinary.add_two_row_metrics(metrics,rows,paths,events,data,ids,
                (Path(data_root)/'observations.jsonl',Path(data_root)/'supervision.jsonl'))
        metrics.update(selection_score=base.checkpoint_selection_score(metrics,'tip_unique_valid'),
            checkpoint_selection_criterion='UniqueClassifiedTipValidAtK + .05 * TipValidAtK',
            generation_budget=dict(requests=len(ids),candidates=4,path_states=4*len(ids),
                extra_candidates=0,language_control_additional_requests=0,repairs=0),
            latency_scope='cached frozen-Qwen head only; not actual full Qwen online E2E')
        if arrays is not None:np.savez_compressed(staging/'paired_language_predictions.npz',**arrays)
        write_json(staging/'metrics.json',metrics);write_json(staging/'per_scene.json',rows)
        if digest(staging/'predictions.npz')!=sealed or pool_identity(trainer,data['scene_ids'][ids],kind)!=identity:
            raise ValueError('Checking changed the original pool or model')
        receipt=dict(identity=identity,journal_before=before,journal_after=journal.snapshot(),metrics=metrics,
            seconds=shared.synchronize()-started,artifacts={f.name:digest(f) for f in staging.iterdir() if f.is_file()})
        reconcile_pool(journal,before,receipt);write_json(staging/'pool_receipt.json',receipt);staging.rename(folder)
        return receipt
    except BaseException as exc:
        seal();write_json(staging/'failure.json',dict(exception=repr(exc),completed_requests=len(generation),
            requested_requests=len(ids),requested_slots=4*len(ids),journal=journal.snapshot(),retry=False))
        raise


def verify_history(out,trainer):
    for row in trainer.history:
        folder=Path(out)/row['pool']
        if folder.resolve().parent!=(Path(out)/'dev').resolve() or digest(folder/'pool_receipt.json')!=row['pool_receipt_sha256']:
            raise ValueError('History pool changed')
        receipt=read(folder/'pool_receipt.json')
        if receipt['identity']['config']!=trainer.config or receipt['identity']['step']!=row['step']:
            raise ValueError('Historical pool config/step differs')
        verify_pool(folder,receipt['identity'])


def positive_hashes(data,ids):
    return {int(i):hashlib.sha256(canonical({k:shared.array_digest(data[k][i]) for k in ('paths','events','path_mask')})).hexdigest() for i in ids}


def expected_counts(arm,steps=3000,eval_every=250,dev=36):
    result=dict(train_model=steps,optimizer=steps,dev_model=(steps//eval_every)*dev)
    if arm=='observed_divergence':result['reference_descriptor']=285
    return result


def runtime():
    result=shared.require_runtime()
    if (result['cpu_affinity']!=[0] or os.environ.get('CODE_COMMIT')!=PROJECT.name or
            any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','LP_NUM_THREADS'))):
        raise ValueError('Immutable source CPU0/all-thread1 required')
    return result


def new_model(cfg,device):
    from routeset.observed_geometry import ObservedGeometryRouteHead
    return ObservedGeometryRouteHead(cfg['feature_dim'],cfg['horizon'],cfg['candidates'],cfg['width'],cfg['depth'],
        cfg['point_width'],cfg['endpoint_residual_bound'],anchor_mode=cfg['anchor_mode'],endpoint_mode=cfg['endpoint_mode']).to(device)


def check_inspection(path,parent,p):
    inspection=read(path)
    if (inspection.get('status')!='completed' or inspection['protocol']!=PROTOCOL or
            inspection['parent_checkpoint_sha256']!=p['parent_checkpoint_sha256'] or
            inspection['model_calls']!=0 or inspection['dataset_reads']!=0 or
            set(inspection['components_sha256'])!=set(COMPONENT_FIELDS) or
            inspection['components_sha256']!={k:nested_digest(parent[k]) for k in COMPONENT_FIELDS} or
            inspection['source_sha256']!={name:digest(PROJECT/name) for name in SOURCES}):
        raise ValueError('Actual CPU parent-state inspection does not match loaded parent')
    return inspection


def full_summary(out,trainer,parent,parent_summary,journal,clock,cache_receipt):
    import torch
    if (trainer.step!=3000 or [r['step'] for r in trainer.history]!=list(range(250,3001,250)) or
            journal.counts!=expected_counts(trainer.config['arm']) or
            trainer.audit['index_chain_sha256']!=trainer.plan['final_index_chain_sha256'] or
            nested_digest(trainer.sampler.bit_generator.state)!=trainer.plan['final_sampler_sha256']):
        raise ValueError('Full same-draw3000/12pool/count budget required')
    verify_history(out,trainer)
    best=torch.load(Path(out)/'best.pt',map_location='cpu',weights_only=False)
    receipt=read(Path(out)/trainer.best['pool']/'pool_receipt.json')
    if best['step']!=trainer.best['step'] or tensor_state_digest(best['model'])!=receipt['identity']['model_sha256']:
        raise ValueError('Saved selected checkpoint is not the sealed best pool')
    changed={name:nested_digest(value)!=nested_digest(parent['model'][name]) for name,value in trainer.model.state_dict().items()}
    return dict(protocol=PROTOCOL,status='completed',arm=trainer.config['arm'],additional_steps=3000,global_last_step=15000,
        additional_input_draws=96000,additional_path_states=384000,selection_opportunities=12,history=trainer.history,
        best=trainer.best,best_metrics=next(r['metrics'] for r in trainer.history if r['step']==trainer.best['step']),
        last_metrics=trainer.history[-1]['metrics'],plan=trainer.plan,sample_stream_audit=trainer.audit,
        positive_reference_chain_sha256=trainer.positive_chain,actual_calls=journal.snapshot(),
        model_parameter_count=sum(v.numel() for v in trainer.model.parameters()),parameter_updates=changed,
        first_gradient_audits=trainer.first_gradient_audits,last_gradient_audit=trainer.gradient_audit,
        parent_components=trainer.parent_components,parent_cost_seconds_shared_not_recharged=parent_summary['elapsed_s'],
        cumulative_arm_seconds=clock.elapsed(),gpu_hours_reserved=clock.elapsed()/3600,reference_cache=cache_receipt,
        best_checkpoint_sha256=digest(Path(out)/'best.pt'),last_checkpoint_sha256=digest(Path(out)/'last.pt'),
        config_sha256=digest(Path(out)/'config.json'),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),fixed_last_train_automatically_run=False,
        cost_scope='Current and prior attempt process time including input/cache/DEV/checkpoints. record_job outer contains it; nested costs not summed. Historical Qwen/12000 training separate.')


def run(args):
    import torch
    out=Path(args.output);p=policy()
    if args.arm not in p['arms']:raise ValueError('One explicit registered arm required')
    if not args.resume and out.exists():raise FileExistsError('Fresh independent output required')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Completed stage is sealed; do not issue more work')
    prior=read_attempt_cost(out)
    if args.stage=='fixed-last-train':prior+=read_attempt_cost(Path(args.train_run))
    clock=Clock(prior,p['wall_cap_seconds_per_arm'],PROCESS_STARTED);clock.check()
    with exclusive_lock(out/'active.lock'):
        attempts=out/'attempts';attempts.mkdir(exist_ok=True);attempt=attempts/('%04d.json'%len(list(attempts.glob('*.json'))))
        status=dict(protocol=PROTOCOL,stage=args.stage,arm=args.arm,status='running',pid=os.getpid(),
            prior_known_process_seconds=prior,resume_command=[sys.executable,'-m','scripts.train_observed_ordered_relation',
                '--stage',args.stage,'--arm',args.arm,'--parent-run',args.parent_run,'--parent-inspection',args.parent_inspection,
                '--math-gate',args.math_gate,'--data',args.data,'--quality-audit',args.quality_audit,'--output',args.output,'--resume']+
                ([] if not args.train_run else ['--train-run',args.train_run]))
        write_json(attempt,status);write_json(out/'status.json',status)
        try:
            actual_runtime=runtime();gate=math_gate(args.math_gate,p);clock.check()
            parent_cfg,parent_summary=parent_metadata(args.parent_run,p)
            parent=torch.load(Path(args.parent_run)/'last.pt',map_location='cpu',weights_only=False)
            components=check_parent(parent,parent_cfg,parent_summary);check_inspection(args.parent_inspection,parent,p)
            # Reuse the existing strict composite guards/loader, not a collector root.
            args.ordinary_run=args.parent_run
            data,geometry,train_ids,dev_ids,ordinary_receipt=shared.read_inputs(args,p)
            if data['fingerprint']!=parent_cfg['dataset_fingerprint']:raise ValueError('Historical all-positive dataset changed')
            config=dict(p,arm=args.arm,dataset_fingerprint=data['fingerprint'],geometry_fingerprint=geometry['fingerprint'],
                parent_components=components,runtime=actual_runtime,math_gate=gate,
                source_sha256={name:digest(PROJECT/name) for name in SOURCES},code_commit=PROJECT.name)
            clock.check();model=new_model(parent_cfg,'cuda');trainer=OrderedTrainer(model,config,parent,train_ids)
            if not all(v.requires_grad for v in model.parameters()):raise ValueError('Original full geometry/head must remain trainable')
            journal=BoundedJournal(out/'requests.jsonl',clock)
            if args.stage=='fixed-last-train':
                fixed_last_stage(args,out,trainer,data,geometry,train_ids,journal,clock)
                status['status']='completed';return
            if args.resume:
                if read(out/'config.json')!=config:raise ValueError('Resume source/policy/cache/arm differs')
                state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
                pending=state['step']>0 and state['step']%p['eval_every']==0 and not any(x['step']==state['step'] for x in state['history'])
                folder=out/'dev'/('step%05d'%state['step']);sealed=pending and folder.exists()
                if sealed:
                    receipt=read(folder/'pool_receipt.json');verify_pool(folder,receipt['identity']);reconcile_pool(journal,state['journal'],receipt)
                trainer.load_state_dict(state,journal,allow_sealed_evaluation=sealed)
                if sealed and receipt['identity']!=pool_identity(trainer,data['scene_ids'][dev_ids],'dev'):
                    raise ValueError('Recovered pool was generated by another model')
                verify_history(out,trainer)
            else:
                write_json(out/'config.json',config);write_json(out/'initialization.json',dict(parent_components=components,
                    restored_components=trainer.components(),plan=trainer.plan,reset_optimizer=False))
                np.savez_compressed(out/'draw_plan.npz',indices=trainer.draws)
            cache={};cache_receipt=None
            if args.arm=='observed_divergence':
                rng_before=trainer.components()['rng']
                cache,cache_receipt=build_reference_cache(data,geometry,train_ids,out,journal,'cuda')
                if trainer.components()['rng']!=rng_before:raise ValueError('Reference descriptor preparation consumed training RNG')
            def save():atomic_torch_save(out/'last.pt',trainer.state_dict(journal,clock.elapsed()))
            def select():
                folder=out/'dev'/('step%05d'%trainer.step)
                receipt=evaluate_pool(trainer,data,geometry,dev_ids,args.data,folder,'dev',journal,'cuda')
                if shared.accept_selection(trainer,receipt,folder.relative_to(out).as_posix(),digest(folder/'pool_receipt.json')):
                    atomic_torch_save(out/'best.pt',trainer.state_dict(journal,clock.elapsed()))
                save();write_json(out/'history.json',trainer.history)
            if not args.resume:save()
            if trainer.step and trainer.step%p['eval_every']==0 and not any(x['step']==trainer.step for x in trainer.history):select()
            if args.stop_after is not None and not trainer.step<=args.stop_after<=p['steps']:
                raise ValueError('Administrative stop-after must be an unreached additional logical step')
            loss_function=training_loss(trainer,data,geometry,'cuda',cache);hashes=positive_hashes(data,train_ids)
            trainer.validate_positive_prefix(hashes)
            while trainer.step<p['steps']:
                if args.stop_after==trainer.step:
                    save();status.update(status='paused',step=trainer.step);return
                clock.check();row=trainer.train_step(loss_function,journal,hashes)
                if trainer.step<=2 or trainer.step%25==0:
                    with (out/'training_history.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(row,cumulative_arm_seconds=clock.elapsed(),stream=trainer.audit),allow_nan=False)+'\n')
                    print(json.dumps(row),flush=True)
                if trainer.step%p['eval_every']==0:save();select()
                elif trainer.step%p['checkpoint_every']==0:save()
            save();trainer.validate_positive_prefix(hashes);parent_metadata(args.parent_run,p)
            summary=full_summary(out,trainer,parent,parent_summary,journal,clock,cache_receipt)
            write_json(out/'summary.json',summary);status.update(status='completed',step=trainer.step)
        except BaseException as exc:
            status.update(status='failed',exception=repr(exc));raise
        finally:
            status.update(process_seconds=time.perf_counter()-PROCESS_STARTED,cumulative_arm_seconds=clock.elapsed())
            if 'journal' in locals():status['issued_calls']=journal.snapshot()
            write_json(attempt,status);write_json(out/'status.json',status)


def fixed_last_stage(args,out,trainer,data,geometry,ids,journal,clock):
    import torch
    source=Path(args.train_run);summary=read(source/'summary.json');config=read(source/'config.json')
    if (summary['status']!='completed' or summary['arm']!=args.arm or config!=trainer.config or
            digest(source/'last.pt')!=summary['last_checkpoint_sha256'] or digest(source/'config.json')!=summary['config_sha256'] or
            read(source/'status.json')['status']!='completed'):
        raise ValueError('Exact completed training checkpoint required for fixed-last diagnostic')
    training_journal=RequestJournal(source/'requests.jsonl')
    if training_journal.snapshot()!=summary['actual_calls']:raise ValueError('Training issued ledger changed')
    state=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    trainer.load_state_dict(state,training_journal);verify_history(source,trainer)
    folder=Path(out)/'pool';identity=pool_identity(trainer,data['scene_ids'][ids],'train_diag')
    if args.resume and journal.records:
        # Only fully sealed evaluations can be reused without issuing again.
        receipt=verify_pool(folder,identity);reconcile_pool(journal,receipt['journal_before'],receipt)
    else:receipt=evaluate_pool(trainer,data,geometry,ids,args.data,folder,'train_diag',journal,'cuda')
    if journal.counts!={'train_diag_model':285}:raise ValueError('Exactly285 fixed-last TRAIN calls required')
    write_json(Path(out)/'summary.json',dict(protocol=PROTOCOL,status='completed',stage='fixed-last-train',arm=args.arm,
        source_last_checkpoint_sha256=summary['last_checkpoint_sha256'],training_summary_sha256=digest(source/'summary.json'),
        receipt=receipt,actual_calls=journal.snapshot(),model_calls=285,path_states=1140,new_qwen_calls=0,
        cumulative_arm_seconds=clock.elapsed(),cumulative_arm_gpu_hours=clock.elapsed()/3600,
        current_diagnostic_process_seconds=time.perf_counter()-PROCESS_STARTED,
        cost_scope='Cumulative includes all known training and diagnostic attempts; this is cached-head CPU/GPU diagnostic, not online E2E.'))


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--stage',choices=['inspect-parent','train','fixed-last-train'],required=True)
    a.add_argument('--arm',choices=['original_saturation','xyz_divergence','observed_divergence'])
    for key in ('parent-run','parent-inspection','math-gate','data','quality-audit','output','train-run'):a.add_argument('--'+key)
    a.add_argument('--resume',action='store_true');a.add_argument('--stop-after',type=int);args=a.parse_args()
    if not args.parent_run or not args.output:a.error('--parent-run and --output required')
    if args.stage=='inspect-parent':inspect_parent(args);return
    if any(getattr(args,k) is None for k in ('arm','parent_inspection','math_gate','data','quality_audit')):
        a.error('All explicit arm/source/data/gate arguments required')
    if args.stage=='fixed-last-train' and (not args.train_run or args.stop_after is not None):a.error('Fixed-last requires --train-run and no stop-after')
    run(args)


if __name__=='__main__':main()
