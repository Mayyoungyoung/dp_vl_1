"""Explicit bounded SFT continuation; normal resume still requires exact config."""
import json
from pathlib import Path
import shutil

from .common import sha256, write_json

PROTOCOL = 'vlm_route_sft_explicit_continuation_v1'
TRAINER = 'scripts/train_vlm_route_sft.py'
HELPER = 'routeset/vlm_sft_continuation.py'


def validate_continuation_config(current, original):
    if set(current) != set(original):
        raise ValueError('Continuation configuration keys differ')
    if current['steps'] <= original['steps']:
        raise ValueError('Explicit continuation must increase the declared total step limit')
    for key in current:
        if key in ('steps','source_sha256'):
            continue
        if current[key] != original[key]:
            raise ValueError('Continuation cannot change '+key)
    before, after = original['source_sha256'], current['source_sha256']
    if set(after) != set(before) | {HELPER} or TRAINER not in before:
        raise ValueError('Continuation source must preserve original helpers and record its new helper')
    for name in before:
        if name not in (TRAINER,HELPER) and before[name] != after[name]:
            raise ValueError('Trained model/input/loss source changed: '+name)
    if not after[TRAINER] or not after[HELPER]:
        raise ValueError('Actual continuation source hashes required')


def prepare_continuation_files(source, output, saved, current):
    """Copy small artifacts/complete ledger; never rewrite the source run."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.name != 'last.pt' or source.parent == output or source.parent in output.parents:
        raise ValueError('Continuation needs source last.pt and an independent new output tree')
    if set(p.name for p in output.iterdir()) - {'train.lock'}:
        raise ValueError('Continuation destination must be fresh except for its exclusive lock')
    original = saved['config']; validate_continuation_config(current, original)
    run = source.parent
    required = ('last.pt','best.pt','initial_adapters.pt','summary.json','config.json',
                'requests.jsonl','data_source_hashes.json','dataset_accounting.json','dev_plan.json')
    hashes = {name:sha256(run/name) for name in required}
    summary = json.loads((run/'summary.json').read_text())
    if (summary['status'] != 'completed' or summary['step'] != original['steps']
            or saved['step'] != original['steps'] or summary['planned_steps'] != original['steps']):
        raise ValueError('Continue only a completed declared source run; interrupted runs use strict resume')
    if json.loads((run/'config.json').read_text()) != original:
        raise ValueError('Source last checkpoint and original config disagree')
    if any(summary['artifacts_sha256'].get(name) != hashes[name] for name in ('last.pt','best.pt','initial_adapters.pt')):
        raise ValueError('Original completed checkpoint artifacts changed')
    if hashes['best.pt'] != saved['best_checkpoint_sha256']:
        raise ValueError('Original selected best no longer matches last.pt receipt')
    if summary['exposure'] != saved['exposure']:
        raise ValueError('Completed source exposure differs from source checkpoint')
    journal = [json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()]
    if len(journal) != saved['journal_event_count']:
        raise ValueError('Completed source must have a fully checkpointed request journal')
    # Prefix digest is rechecked by the unchanged shared-loop restore below.
    for name in ('requests.jsonl','data_source_hashes.json','dataset_accounting.json','dev_plan.json','initial_adapters.pt','best.pt'):
        shutil.copyfile(run/name, output/name)
    shutil.copyfile(run/'best.pt', output/'inherited_best.pt')
    write_json(output/'config.json', current)
    lineage = dict(protocol=PROTOCOL, source_checkpoint=str(source), source_run=str(run),
        source_file_sha256=hashes, original_config=original, original_planned_steps=original['steps'],
        restored_step=saved['step'], added_planned_steps=current['steps']-saved['step'],
        total_planned_steps=current['steps'], inherited_elapsed_seconds=summary['elapsed_seconds'],
        source_checkpoint_elapsed_seconds=saved['elapsed_seconds'], inherited_exposure=saved['exposure'],
        inherited_best_step=saved['best_step'], inherited_best_checkpoint_sha256=hashes['best.pt'],
        source_checkpoint_not_modified=True,
        scope='Explicit larger-budget run; original metadata/checkpoints stay immutable. '
              'Inherited best.pt bytes/config remain original until a new DEV-token-NLL winner is saved.')
    write_json(output/'continuation_source.json', lineage)
    return lineage


def verify_continuation_source(lineage):
    run = Path(lineage['source_run'])
    for name, expected in lineage['source_file_sha256'].items():
        if sha256(run/name) != expected:
            raise ValueError('Original continuation source changed: '+name)


def continuation_accounting(lineage, exposure, elapsed_seconds):
    added = {}
    for split, counters in exposure.items():
        added[split] = {key:value-lineage['inherited_exposure'][split][key] for key,value in counters.items()}
        if any(value < 0 for value in added[split].values()):
            raise ValueError('Cumulative continuation exposure cannot be smaller than its source')
    elapsed_added = elapsed_seconds-lineage['inherited_elapsed_seconds']
    if elapsed_added < 0:
        raise ValueError('Cumulative elapsed time cannot erase source cost')
    return dict(inherited_elapsed_seconds=lineage['inherited_elapsed_seconds'],
        added_elapsed_seconds=elapsed_added, cumulative_elapsed_seconds=elapsed_seconds,
        inherited_exposure=lineage['inherited_exposure'], added_exposure=added, cumulative_exposure=exposure,
        scope='Add only added cost when aggregating this child with its original parent; cumulative values include both. '
              'Logged uncheckpointed replay is charged by the shared resume ledger.')
