"""Read the completed, sealed two-run logs only; no neural inference."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'observed_two_row_prefix76_convergence_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(root, names):
    found = {}
    for name in names:
        for line in (root / name).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and 'path_loss' in row:
                step = row['step']
                if step in found:
                    raise ValueError('Repeated logged optimization interval')
                found[step] = row
    assert sorted(found) == list(range(100, 12001, 100))
    return [found[step] for step in sorted(found)]


def main():
    result = {'protocol': 'paired_12000_history_readonly_v1', 'new_forward_requests': 0,
              'loss_scope': 'Each logged loss is mean of latest100 optimization batches, not full TRAIN reevaluation.',
              'train_pool_scope': 'Only selected best and fixedlast have full TRAIN pools; DEV history is not TRAIN history.',
              'methods': {}}
    fields = ('TipValidAtK', 'semantic_goal_accuracy', 'TipClearAtK',
              'UniqueClassifiedTipValidAtK', 'selection_score')
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    for method, root, logs, color in (
            ('constant', OLD, ['stage1500.log', 'finish.log'], '#0072B2'),
            ('no_direct', HERE, ['train12000.log'], '#D55E00')):
        training = rows(root, logs)
        history = read(root / 'peak_seed0/history.json')
        assert [x['step'] for x in history] == list(range(250, 12001, 250))
        dev = [dict(step=x['step'], loss=x['loss'], **{k:x['dev_model'][k] for k in fields})
               for x in history]
        result['methods'][method] = dict(
            training_intervals=training, dev_history=dev,
            sources_sha256={name:sha(root/name) for name in logs + ['peak_seed0/history.json']},
            final_path_loss=training[-1]['path_loss'],
            final_grounding_loss=training[-1]['grounding_loss'],
            final_total_loss=training[-1]['loss'],
            last3000_dev_tip_range=[min(x['TipValidAtK'] for x in dev if x['step']>9000),
                                   max(x['TipValidAtK'] for x in dev if x['step']>9000)])
        steps = [x['step'] for x in training]
        axes[0, 0].semilogy(steps, [x['path_loss'] for x in training], label=method, color=color)
        axes[0, 1].plot(steps, [x['grounding_loss'] for x in training], label=method, color=color)
        for ax, key in zip(axes[1], ('TipValidAtK', 'semantic_goal_accuracy', 'UniqueClassifiedTipValidAtK')):
            ax.plot([x['step'] for x in dev], [x[key] for x in dev], label=method, color=color)
    paired=read(HERE/'paired_stream_receipt.json')
    assert paired['constant_scheduler_exact'] and paired['shared_initialization_exact']
    result['constant_lr_scope']={'value':.0003,'scheduler_final_state_exact':True,'source_sha256':sha(HERE/'paired_stream_receipt.json')}
    axes[0,2].plot(range(1,12001),[.0003]*12000,color='#0072B2',label='both constant LR')
    titles = ('Path loss, latest100 batches', 'Grounding loss, latest100 batches',
              'Unchanged constant learning rate', 'DEV TipValid@4', 'DEV semantic endpoint accuracy',
              'DEV unique classified tip-valid')
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title); ax.set_xlabel('Optimizer update'); ax.grid(alpha=.18); ax.legend()
    fig.suptitle('Same shared initialization,12000 updates, full sample chain;48 DEV evaluations each\n'
                 'No new inference or checkpoint selection in this analysis')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(HERE/'training_history.png',dpi=150);plt.close(fig)
    result['source_script_sha256'] = sha(Path(__file__))
    (HERE/'TRAINING_HISTORY_COMPARISON.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:{field:v[field] for field in ('final_path_loss','final_grounding_loss','final_total_loss','last3000_dev_tip_range')}
                      for k,v in result['methods'].items()}))


if __name__ == '__main__':
    main()
