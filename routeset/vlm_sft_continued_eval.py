"""Strict provenance for the single bounded 1500-to-6000 TRAIN8 probe.

This does not weaken the original same-checkpoint inference guard. An inherited
best must reuse the original measured result instead of relabelling its config.
"""
import hashlib
import json
from pathlib import Path

from .common import sha256
from .vlm_sft_continuation import (PROTOCOL, TRAINER, HELPER,
    validate_continuation_config, verify_continuation_source, continuation_accounting)
from .vlm_sft_data import TRAINING_PROTOCOL

SOURCE_COMMIT = '5bb9087c9ca511c2a68a4da08798e95e8c6d4cdc'
PROBE_PROTOCOL = 'vlm_continued_train8_greedy_v1'
ORIGINAL_RUN = '/home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_v1/seed0'
DATA_FINGERPRINT = '40f33d663944402247667c425424b99474afaaf1234141910f5c3189e7c50da0'
ORIGINAL_FILES = {
    'last.pt': 'd604b5a213bdf281e7976b460b7ceb2fc428488610b1b84670ddca04711cb00c',
    'best.pt': '675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f',
    'initial_adapters.pt': '966970d0a68904a6828e036fbf8476e9c5956378843224a388aa4579d78bcf8b',
    'summary.json': '8f30f8e5b18bc8eb7d14fcc621c338b55ff78f272837e8ce946e5a71ba2b2143',
    'config.json': '84ffe585096a24c45be225059fa25b6020ebda5458dda380f0b94690d78f75dd',
    'requests.jsonl': '477b70f751a76a79e32e3689ba8dbf1fe70ae0467ff199d58a07b359566c8b5d',
    'data_source_hashes.json': 'ce79a32dcf88fb083f7ed9422a14846c40a1697760b963cdd244716834d996b7',
    'dataset_accounting.json': '4a6d53791fc62852d62bc7a9b042e4d3d9ce772d081a74d8a193caae33609670',
    'dev_plan.json': '0d47b4c07eaf1eb6dbf9cbed98f3b9e80e1b50ad707db409a35510566d78cead',
}
CONTINUATION_SOURCES = {
    TRAINER: '0b6a0f0d76b853d008f0580468933f49697e16d3af86ec39aea2731eafb19f52',
    HELPER: '0d73f95c70e35b902710fd030b140a3129586ca1e201084104ac885886390709',
}


def validate_receipts(summary, config, lineage, checkpoint_hash):
    """Pure checks before any model or observation is opened."""
    if (summary['protocol'] != TRAINING_PROTOCOL or summary['status'] != 'completed'
            or summary['step'] != 6000 or summary['planned_steps'] != 6000
            or summary['code_commit'] != SOURCE_COMMIT or config['steps'] != 6000):
        raise ValueError('Only the completed registered 6000-step continuation is eligible')
    if (lineage != summary['continuation'] or lineage['protocol'] != PROTOCOL
            or lineage['source_run'] != ORIGINAL_RUN
            or lineage['source_checkpoint'] != ORIGINAL_RUN+'/last.pt'
            or lineage['source_file_sha256'] != ORIGINAL_FILES
            or lineage['restored_step'] != 1500 or lineage['original_planned_steps'] != 1500
            or lineage['added_planned_steps'] != 4500 or lineage['total_planned_steps'] != 6000
            or lineage['source_checkpoint_not_modified'] is not True):
        raise ValueError('Original completed1500 lineage changed')
    validate_continuation_config(config, lineage['original_config'])
    if any(config['source_sha256'].get(name) != value for name,value in CONTINUATION_SOURCES.items()):
        raise ValueError('Unregistered continuation training source')
    if (config['data_fingerprint'] != DATA_FINGERPRINT or summary['data_fingerprint'] != DATA_FINGERPRINT
            or summary['original_source_files_unchanged'] is not True
            or summary['frozen_lora_projection_hashes_unchanged'] is not True):
        raise ValueError('Data, original source or frozen backbone provenance changed')
    if summary['best_is_inherited'] is not False:
        raise ValueError('Inherited best: reuse original greedy measurement; do not regenerate or relabel')
    if (not 1500 < summary['best_step'] <= 6000
            or summary['artifacts_sha256']['best.pt'] != checkpoint_hash
            or checkpoint_hash == ORIGINAL_FILES['best.pt']):
        raise ValueError('Actual newly selected checkpoint/hash required')
    audit = summary['added_gradient_audit']
    if len(audit) != 8 or not all(row['ever_nonzero_gradient'] and row['changed_from_initial'] for row in audit.values()):
        raise ValueError('All eight adapters must have actual added-stage updates')
    expected = continuation_accounting(lineage,summary['exposure'],summary['elapsed_seconds'])
    if summary['continuation_accounting'] != expected:
        raise ValueError('Inherited/added/cumulative exposure or cost changed')
    if expected['cumulative_exposure']['TRAIN']['candidate_slots'] < 15000:
        raise ValueError('Incomplete declared training exposure')
    return expected


def validate_saved_checkpoint(saved, summary, config, lineage):
    if (saved['protocol'] != TRAINING_PROTOCOL or saved['config'] != config
            or saved['continuation'] != lineage or saved['best_is_inherited'] is not False
            or saved['step'] != summary['best_step'] or saved['best_step'] != summary['best_step']
            or saved['best_nll'] != summary['best_dev_token_nll']):
        raise ValueError('Selected checkpoint state disagrees with completed lineage')


def validate_inherited_metadata(lineage, original):
    if (original['status'] != 'completed' or original['step'] != 1500 or original['planned_steps'] != 1500
            or original['exposure'] != lineage['inherited_exposure']
            or original['elapsed_seconds'] != lineage['inherited_elapsed_seconds']
            or original['best_step'] != lineage['inherited_best_step']
            or original['artifacts_sha256']['best.pt'] != lineage['inherited_best_checkpoint_sha256']):
        raise ValueError('Inherited exposure, time or best differs from original completed summary')


def inspect_completed_run(run):
    """Read small run metadata and hashes only; no routes/images/target labels."""
    run = Path(run).resolve()
    read = lambda name: json.loads((run/name).read_text())
    summary, config, lineage = read('summary.json'), read('config.json'), read('continuation_source.json')
    checkpoint_hash = sha256(run/'best.pt')
    validate_receipts(summary,config,lineage,checkpoint_hash)
    if Path(summary['selected_checkpoint_origin']).resolve() != run/'best.pt':
        raise ValueError('Selected origin differs from this new run')
    verify_continuation_source(lineage)
    source = Path(lineage['source_run'])
    validate_inherited_metadata(lineage,json.loads((source/'summary.json').read_text()))
    if json.loads((source/'config.json').read_text()) != lineage['original_config']:
        raise ValueError('Original config receipt changed')
    for name in ('initial_adapters.pt','data_source_hashes.json','dataset_accounting.json','dev_plan.json'):
        if sha256(run/name) != ORIGINAL_FILES[name]:
            raise ValueError('Inherited shared training artifact changed: '+name)
    if sha256(run/'inherited_best.pt') != ORIGINAL_FILES['best.pt']:
        raise ValueError('Immutable inherited best bytes changed')
    index = read('data_source_hashes.json')
    if hashlib.sha256(json.dumps(index,sort_keys=True).encode()).hexdigest() != DATA_FINGERPRINT:
        raise ValueError('Training data index changed')
    receipt = dict(protocol='continued_sft_train8_lineage_v1',
        training_summary_sha256=sha256(run/'summary.json'), config_sha256=sha256(run/'config.json'),
        continuation_source_sha256=sha256(run/'continuation_source.json'),
        selected_checkpoint_sha256=checkpoint_hash, selected_step=summary['best_step'],
        original_source_files_sha256=ORIGINAL_FILES, training_accounting=summary['continuation_accounting'],
        training_code_commit=SOURCE_COMMIT, original_source_files_verified=True)
    return summary,config,lineage,index,receipt
