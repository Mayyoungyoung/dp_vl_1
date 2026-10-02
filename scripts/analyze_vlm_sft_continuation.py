"""Read-only journal/history accounting for the completed bounded SFT child."""
import argparse
from collections import Counter
import json
from pathlib import Path

from routeset.common import sha256,write_json
from routeset.vlm_sft_continued_eval import validate_receipts,validate_inherited_metadata
from routeset.vlm_sft_train_probe import PARENTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--continued',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh analysis output required')
    read=lambda folder,name:json.loads((folder/name).read_text(encoding='utf-8'))
    original=read(args.original,'summary.json');summary=read(args.continued,'summary.json')
    config=read(args.continued,'config.json');lineage=read(args.continued,'continuation_source.json')
    validate_receipts(summary,config,lineage,summary['artifacts_sha256']['best.pt'])
    validate_inherited_metadata(lineage,original)
    rows=lambda folder:[json.loads(s) for s in (folder/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    before,after=rows(args.original),rows(args.continued)
    if after[:len(before)]!=before:raise ValueError('Inherited journal prefix changed')
    history=read(args.continued,'history.json');old_history=read(args.original,'history.json')
    if history[:len(old_history)]!=old_history:raise ValueError('Inherited training/DEV history changed')
    train=[r for r in after if r['kind']=='train_request']
    dev=[r for r in after if r['kind']=='dev_request']
    if [r['step'] for r in train]!=list(range(1,6001)):
        raise ValueError('This analysis expects the actual completed run without replay')
    if any(r['k']!=(1 if r['step']%2 else 4) for r in train):raise ValueError('Actual alternating schedule differs')
    for split,events in [('TRAIN',train),('DEV_MODEL',dev)]:
        counters=summary['exposure'][split]
        for key,value in [('requests',len(events)),('candidate_slots',sum(r['k'] for r in events)),
                          ('prompt_tokens',sum(r['prompt_tokens'] for r in events)),
                          ('supervised_tokens',sum(r['supervised_tokens'] for r in events))]:
            if counters[key]!=value:raise ValueError('Journal/summary exposure differs: '+split+'/'+key)
    if not all(r['prefix_exactly_matches'] and r['prompt_fully_masked'] and r['generation_prefix_contains_no_answer'] for r in after):
        raise ValueError('Recorded prompt/mask invariant failed')
    def sample_counts(events,p):
        selected=[r for r in events if r['kind']=='train_request' and r['scene_id']==p+'_target0']
        return dict(requests=len(selected),route_slots=sum(r['k'] for r in selected),
            k_counts=dict(Counter(r['k'] for r in selected)),
            first_reference_slot_exposures=sum(r['reference_indices'].count(0) for r in selected))
    curve=[dict(step=r['step'],token_nll=r['dev_token_nll'],selected=r['selected']) for r in history if 'dev_token_nll' in r]
    result=dict(status='completed',protocol='vlm_sft_bounded_continuation_accounting_v1',
        original_summary_sha256=sha256(args.original/'summary.json'),continued_summary_sha256=sha256(args.continued/'summary.json'),
        analyzer_sha256=sha256(__file__),inherited_journal_exact=True,inherited_history_exact=True,
        actual_journal_event_count=len(after),actual_training_k_counts=dict(Counter(r['k'] for r in train)),
        actual_positive_train_parents=len({r['parent_id'] for r in train}),actual_positive_train_instructions=len({r['scene_id'] for r in train}),
        continuation_accounting=summary['continuation_accounting'],dev_token_nll_curve=curve,
        best_step=summary['best_step'],last_step=6000,best_token_nll=summary['best_dev_token_nll'],
        relative_best_token_nll_reduction=1-summary['best_dev_token_nll']/original['best_dev_token_nll'],
        last_token_nll=curve[-1]['token_nll'],max_recorded_coordinate_quantization_error_m=max(r['max_coordinate_quantization_error_m'] for r in after),
        train8_target0_exposure=[dict(parent_id=p,original=sample_counts(before,p),cumulative=sample_counts(after,p)) for p in PARENTS],
        actual_added_all_eight_adapters_updated=all(r['ever_nonzero_gradient'] and r['changed_from_initial'] for r in summary['added_gradient_audit'].values()),
        scope='Only completed request logs/metadata; no new model calls or data labels. Teacher token NLL is not generated route quality; '
              'reference slot exposure counts include repeated positive references, not distinct modes.')
    args.output.mkdir(parents=True);write_json(args.output/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,3.8),constrained_layout=True)
    ax.plot([r['step'] for r in curve],[r['token_nll'] for r in curve],'.-',color='#2471a3')
    ax.axvline(1500,linestyle='--',color='#777777',label='Original training limit')
    ax.scatter([summary['best_step']],[summary['best_dev_token_nll']],color='#be4b3a',label='Selected best')
    ax.set(xlabel='Cumulative optimizer steps',ylabel='Fixed-plan DEV token NLL',title='SFT optimization: bounded continuation, one training seed')
    ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.savefig(args.output/'dev_token_nll.png',dpi=180);plt.close(fig)
    print(json.dumps({key:result[key] for key in ('inherited_journal_exact','inherited_history_exact','best_step','best_token_nll','last_token_nll','relative_best_token_nll_reduction')}))


if __name__=='__main__':main()
