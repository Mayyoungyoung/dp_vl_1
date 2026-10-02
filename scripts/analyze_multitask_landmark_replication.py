"""Three paired training seeds, retaining original best and fixed final step."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from scripts.analyze_multitask_landmark_pair import analyze, sha


FIELDS=('candidate_matched_ADE_m','candidate_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')


def aggregate(ordinary,auxiliary):
    if len(ordinary)!=3 or len(auxiliary)!=3:raise ValueError('Exactly three paired training seeds required')
    delta=[b-a for a,b in zip(ordinary,auxiliary)]
    return dict(ordinary=dict(values=ordinary,mean=mean(ordinary),sample_std=stdev(ordinary)),
        auxiliary=dict(values=auxiliary,mean=mean(auxiliary),sample_std=stdev(auxiliary)),
        paired_auxiliary_minus_ordinary=dict(values=delta,mean=mean(delta),sample_std=stdev(delta)))


def run(root,output):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError('Fresh summary required')
    pairs=[analyze(root/'runs/observed_multitask_prefix108_v1/free_offset_seed0',
        root/'runs/observed_multitask_landmark_aux_v1/event_supported_seed0',
        root/'reports/observed_multitask_landmark_aux_preflight_v1/actual_control/summary.json')]
    family=root/'runs/observed_multitask_landmark_replication_v1'
    for seed in (1,2):pairs.append(analyze(family/('ordinary_seed'+str(seed)),family/('event_supported_seed'+str(seed))))
    if [pair['seed'] for pair in pairs]!=[0,1,2]:raise ValueError('Different seed set')
    reference={k:v for k,v in pairs[0]['matched_config'].items() if k!='seed'}
    if any({k:v for k,v in pair['matched_config'].items() if k!='seed'}!=reference for pair in pairs[1:]):
        raise ValueError('Dataset, exposure, model, selection or metric changed between seeds')
    metrics={};tasks=[];parents=[]
    for checkpoint in ('best','last','train_best'):
        metrics[checkpoint]={field:aggregate([pair['metrics'][checkpoint]['ordinary'][field] for pair in pairs],
            [pair['metrics'][checkpoint]['auxiliary'][field] for pair in pairs]) for field in FIELDS}
        metrics[checkpoint]['denominators_per_seed']=pairs[0]['metrics'][checkpoint]['denominators']
        for category,key,destination in [('per_task','task',tasks),('per_parent','parent_id',parents)]:
            rows=[{row[key]:row for row in pair[category] if row['checkpoint']==checkpoint} for pair in pairs]
            if any(row.keys()!=rows[0].keys() for row in rows[1:]):raise ValueError('Per-task/parent set changed')
            for identifier in sorted(rows[0]):
                fields={}
                for field in FIELDS:
                    a=[row[identifier]['ordinary'][field] for row in rows];b=[row[identifier]['auxiliary'][field] for row in rows]
                    if any(v is None for v in a+b):
                        if not all(v is None for v in a+b):raise ValueError('Changing reference denominator')
                        fields[field]=None
                    else:fields[field]=aggregate(a,b)
                destination.append(dict(checkpoint=checkpoint,identifier=identifier,task=rows[0][identifier]['task'],metrics=fields))
    result=dict(protocol='multitask_landmark_three_paired_training_seeds_v1',status='passed',seeds=[0,1,2],
        analysis_source_sha256=sha(__file__),pair_analyzer_sha256=sha(Path(__file__).with_name('analyze_multitask_landmark_pair.py')),
        unchanged_configuration=reference,metrics=metrics,per_task=tasks,per_parent=parents,pairs=pairs,
        standard_deviation='Sample standard deviation over three separately trained seeds; no pooling of candidates or test-set certainty claim.',
        initialization_scope='Seed0 retains historical source/seed reconstruction and sampler replay. Seeds1/2 persist actual initial model/RNG hashes and actual ordered draw chains in every checkpoint.',
        scope='Same reused 12-parent DEV. Original ADE-selected best and fixed step1500 both retained. Generic reference/event reconstruction only; no semantic, collision, execution or task-success measurements. Conventional baseline repair, not core mechanism.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',metrics=metrics)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.output)
