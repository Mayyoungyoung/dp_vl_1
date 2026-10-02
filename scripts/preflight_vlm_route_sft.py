"""Actual pinned-Qwen SFT boundary/gradient/memory preflight on TRAIN only.

Not a quality experiment. Saves small adapter states and exact source receipts;
never saves another full backbone or loads a frozen hidden-state feature cache.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from routeset.vlm_route_serialization import (PROTOCOL,depth_image,prompt,serialize_paths,
                                             parse_paths,assistant_only_labels)


def resolve(root,value):
    path=Path(value)
    return path if path.is_absolute() else root/path


def read_first_train(observations,supervision,horizon):
    from routeset.observed_route_head import OBSERVATION_KEYS,resample_event_segments
    observations=Path(observations);supervision=Path(supervision)
    rows=[json.loads(line) for line in observations.read_text().splitlines() if line.strip()]
    if any(set(row)!=OBSERVATION_KEYS for row in rows):raise ValueError('Observation whitelist mismatch')
    if any(row['split'] not in ('TRAIN','DEV_MODEL') for row in rows):raise ValueError('Use isolated development export')
    # Only selected TRAIN labels are joined; other labels are not inference inputs.
    labels={row['id']:row for row in map(json.loads,supervision.read_text().splitlines()) if row['split']=='TRAIN'}
    eligible=sorted([row for row in rows if row['split']=='TRAIN' and labels[row['id']]['routes']],key=lambda row:row['id'])
    if not eligible:raise ValueError('No positive TRAIN reference for preflight')
    row=eligible[0];label=labels[row['id']]
    if label['parent_id']!=row['parent_id']:raise ValueError('TRAIN parent join mismatch')
    current_file=resolve(supervision.parent,label['observation'])
    with np.load(current_file,allow_pickle=False) as source:
        current=np.r_[source['gripper_pose'],source['gripper_open'].reshape(-1)]
        observed={key:source[key].copy() for key in ('depth','camera_intrinsics','camera_extrinsics')}
    image_file=resolve(observations.parent,row['image'])
    with Image.open(image_file) as image:rgb=image.convert('RGB').copy()
    paths=[];events=[];files=[observations,supervision,image_file,current_file]
    for reference in label['routes']:
        path=resolve(supervision.parent,reference);files.append(path)
        with np.load(path,allow_pickle=False) as source:
            xyz,opened=resample_event_segments(source['gripper_pose'],source['gripper_open'],horizon)
        paths.append(xyz);events.append(opened)
    return row,current,observed,rgb,np.stack(paths),np.stack(events),files


def prepare_example(processor,row,current,observed,rgb,paths,events,k,horizon):
    text=prompt(row['instruction'],current,observed['camera_intrinsics'],observed['camera_extrinsics'],k,horizon)
    chosen=np.arange(k)%len(paths)  # Fixed TRAIN preflight, no invented unseen modes.
    answer=serialize_paths(paths[chosen],events[chosen])
    depth=Image.fromarray(depth_image(observed['depth']))
    messages=[dict(role='user',content=[dict(type='image'),dict(type='image'),dict(type='text',text=text)])]
    prefix_text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    full_text=processor.apply_chat_template(messages+[dict(role='assistant',content=[dict(type='text',text=answer)])],
                                             tokenize=False,add_generation_prompt=False)
    prefix=processor(text=[prefix_text],images=[rgb,depth],return_tensors='pt')
    full=processor(text=[full_text],images=[rgb,depth],return_tensors='pt')
    labels=assistant_only_labels(full['input_ids'][0].numpy(),prefix['input_ids'][0].numpy())
    import torch
    full['labels']=torch.as_tensor(labels,dtype=torch.long)[None]
    parsed,_,receipt=parse_paths(answer,k,horizon)
    if receipt['format_valid_candidates']!=k:raise ValueError('Reference serialization rejected')
    return prefix,full,dict(k=k,prompt_tokens=len(prefix['input_ids'][0]),sequence_tokens=len(labels),
        supervised_tokens=int((labels!=-100).sum()),max_coordinate_quantization_error_m=float(np.abs(parsed-paths[chosen]).max()),
        prefix_exactly_matches=True,prompt_fully_masked=True,generation_prefix_contains_no_answer=True,
        reference_count=len(paths),chosen_reference_indices=chosen.tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','supervision','model','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous preflight')
    args.output.mkdir(parents=True)
    import torch
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from routeset.common import seed_all,sha256,write_json
    from routeset.observed_route_head import QWEN_REVISION
    from scripts.train_observed_lora import install_lora,adapters,adapter_state,tensor_hash
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only authorized GPU1')
    if transformers.__version__!='4.57.1' or not torch.__version__.startswith('2.4.1'):raise ValueError('Pinned .venv-qwen required')
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35);torch.cuda.reset_peak_memory_stats()
    seed_all(0)
    provenance=json.loads((args.model/'provenance.json').read_text())
    if provenance.get('revision')!=QWEN_REVISION or not provenance.get('all_hashes_verified'):raise ValueError('Verified fixed model required')
    started=time.perf_counter();row,current,observed,rgb,paths,events,files=read_first_train(args.observations,args.supervision,24)
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,min_pixels=65536,max_pixels=65536)
    prepared={k:prepare_example(processor,row,current,observed,rgb,paths,events,k,24) for k in (1,4)}
    write_json(args.output/'processor_audit.json',{str(k):item[2] for k,item in prepared.items()})
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'})
    modules=install_lora(model);model.eval()
    params=adapters(model)
    if any(parameter.requires_grad for name,parameter in model.named_parameters() if name not in params):
        raise ValueError('Unexpected trainable backbone parameters')
    before={name:tensor_hash(value) for name,value in params.items()}
    base={name:tensor_hash(value) for name,value in model.named_parameters() if '.base.weight' in name}
    torch.save(adapter_state(model),args.output/'initial_adapters.pt')
    optimizer=torch.optim.AdamW(params.values(),lr=1e-4,weight_decay=0.)
    records=[]
    for step,k in enumerate((1,4,1,4),1):
        full=prepared[k][1].to('cuda');torch.cuda.synchronize();tic=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output=model(**full,use_cache=False,return_dict=True)
        loss=output.loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite real SFT loss')
        loss.backward()
        norms={name:None if value.grad is None else float(value.grad.float().norm()) for name,value in params.items()}
        if not all(value is not None and np.isfinite(value) for value in norms.values()) or not any(value>0 for value in norms.values()):
            raise RuntimeError('Missing/nonfinite real adapter gradient')
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        optimizer.step();torch.cuda.synchronize()
        records.append(dict(step=step,k=k,loss=float(loss.detach()),gradient_norms=norms,wall_seconds=time.perf_counter()-tic))
        del full,output,loss
    after={name:tensor_hash(value) for name,value in params.items()}
    changed={name:before[name]!=after[name] for name in params}
    if not all(changed.values()):raise RuntimeError('Some real adapter tensors never changed')
    if base!={name:tensor_hash(value) for name,value in model.named_parameters() if '.base.weight' in name}:
        raise RuntimeError('Frozen projection weights changed')
    torch.save(adapter_state(model),args.output/'final_adapters.pt')
    sources=files+[Path(__file__),Path(__file__).resolve().parents[1]/'routeset/vlm_route_serialization.py',
        Path(__file__).with_name('train_observed_lora.py'),args.model/'provenance.json']
    summary=dict(protocol=PROTOCOL,status='actual_sft_boundary_gradient_memory_preflight_complete',
        code_commit=os.environ.get('CODE_COMMIT'),model_revision=QWEN_REVISION,train_scene_id=row['id'],train_parent_id=row['parent_id'],
        processor={str(k):item[2] for k,item in prepared.items()},records=records,adapter_modules=modules,
        adapter_parameters=sum(value.numel() for value in params.values()),adapter_hash_before=before,adapter_hash_after=after,
        all_adapter_tensors_updated=all(changed.values()),frozen_lora_projection_hashes_unchanged=True,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started,
        source_sha256={str(path):sha256(path) for path in sources},
        artifacts_sha256={path.name:sha256(path) for path in args.output.glob('*.pt')},
        scope='Four optimizer steps on one fixed TRAIN observation; not trained route quality, autoregressive inference, convergence or novelty evidence')
    write_json(args.output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('status','all_adapter_tensors_updated','peak_cuda_allocated_bytes','elapsed_seconds')}))


if __name__=='__main__':main()
