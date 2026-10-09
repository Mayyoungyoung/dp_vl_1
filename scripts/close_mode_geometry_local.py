"""Verify the copied research artifacts without loading data or models."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'runs/mode_geometry_v1'
    report = root / 'research_mode_geometry_v1/results'
    audit = json.loads((report / 'ARTIFACT_AUDIT.json').read_text())
    checked = {}
    for rel, spec in audit['files'].items():
        path = run / rel
        assert path.stat().st_size == spec['bytes'], rel
        assert sha(path) == spec['sha256'], rel
        checked[rel] = spec['sha256']
    for job, digest in audit['source_hash_receipts'].items():
        assert sha(root / 'runs/verified_set_v1/jobs' / job / 'receipt.json') == digest, job
    for commit, digest in audit['source_exports'].items():
        assert sha(run / ('source_' + commit + '.tar')) == digest, commit
    assert sha(run / 'fixed_assets/scorer_bundle.pt') == audit['scorer_sha256']
    assert sha(run / 'fixed_assets/parent_last.pt') == audit['initial_sha256']
    archive_digest = sha(run / 'mode_geometry_v1_closure_20261009.tar')
    assert archive_digest == 'f18be65d83d1d2d5f726388a395b2f7a925a4c705943eee058a0986e4a0b5825'
    final = json.loads((root / 'runs/verified_set_v1/jobs/mg_close_audit/receipt.json').read_text())
    assert final['status'] == 'completed' and final['exit_code'] == 0
    jobs = audit['jobs'] + [{k: final[k] for k in (
        'id', 'command', 'source_commit', 'status', 'exit_code', 'elapsed_seconds', 'end_utc')}]
    budget = audit['budget_excluding_this_audit']
    spent = budget['spent'] + final['elapsed_seconds']
    closure = {
        'status': 'closed; complete-method scientific acceptance failed',
        'end_utc': final['end_utc'], 'job_count': len(jobs),
        'successful_jobs': sum(j['exit_code'] == 0 for j in jobs),
        'failed_jobs': [j['id'] for j in jobs if j['exit_code'] != 0],
        'checkpoint_count': len(audit['checkpoints']), 'main_three_seed_runs': 15,
        'local_files_hash_verified': len(checked),
        'source_exports_verified': len(audit['source_exports']),
        'closure_archive_sha256': archive_digest,
        'budget_allowance_seconds': budget['allowance'],
        'total_spent_seconds': spent,
        'this_research_spent_seconds': budget['research_family_spent'] + final['elapsed_seconds'],
        'remaining_seconds': budget['allowance'] - spent,
        'last_audit_receipt_sha256': sha(root / 'runs/verified_set_v1/jobs/mg_close_audit/receipt.json'),
        'environment': audit['environment'], 'locked_access': False,
        'default_parent_unchanged': True, 'jobs': jobs,
    }
    (report / 'CLOSURE.json').write_text(json.dumps(closure, indent=2) + '\n', encoding='utf-8')
    (report / 'LOCAL_HASH_VERIFICATION.json').write_text(
        json.dumps({'all_match': True, 'files': checked}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in closure.items() if k != 'jobs'}, indent=2))


if __name__ == '__main__':
    main()
