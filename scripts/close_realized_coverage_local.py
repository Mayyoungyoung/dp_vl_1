"""Hash-check the copied new-study artifacts; never load reserved observations."""
import hashlib
import json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parents[1];run=root/'runs/realized_coverage_v1';report=root/'research_realized_coverage_v1/results'
    audit=json.loads((run/'ARTIFACT_AUDIT_FINAL.json').read_text());checked={}
    for rel,spec in audit['files'].items():
        p=run/rel;assert p.stat().st_size==spec['bytes'],rel;assert sha(p)==spec['sha256'],rel;checked[rel]=spec['sha256']
    for job,digest in audit['job_receipt_sha256'].items():assert sha(run/'jobs'/job/'receipt.json')==digest,job
    for commit,digest in audit['source_exports'].items():assert sha(run/('source_'+commit+'.tar'))==digest,commit
    assert sha(root/'runs/mode_geometry_v1/canonical_C_seed0/last.pt')==audit['initial_sha256']
    assert sha(root/'runs/mode_geometry_v1/fixed_assets/scorer_bundle.pt')==audit['scorer_sha256']
    final=json.loads((run/'jobs/rc_artifact_audit_final/receipt.json').read_text())
    assert final['status']=='completed' and final['exit_code']==0
    jobs=audit['jobs']+[final]
    closure=dict(status='closed with negative mechanism evidence; strongest new system is ordinary success allocation',
        paper_ready=False,default_changed=False,locked_access=False,end_utc=final['end_utc'],
        elapsed_seconds=sum(j['elapsed_seconds'] for j in jobs),time_limit=None,jobs=len(jobs),
        failed_jobs=[j['id'] for j in jobs if j['exit_code']!=0],successful_jobs=sum(j['exit_code']==0 for j in jobs),
        checkpoints=len(audit['checkpoints']),evaluations=len(audit['evaluated']),
        allocation_training_seeds=[0,1,2,3,4],geometry_replication_scope='exploratory seed0, not five geometry seeds',
        files_verified=len(checked),source_exports_verified=len(audit['source_exports']),
        archive_sha256=sha(run/'realized_coverage_closure_v2.tar'),environment=audit['environment'],
        feedback_verified_routes=audit['feedback_verified_routes'],discarded_feedback_routes=audit['additional_discarded_feedback_routes'],
        reference_checker_queries=audit['reference_diagnostic_checker_queries'],deployment_routes=audit['deployment_routes'],
        deployment_replay_benchmark_routes=audit['deployment_replay_benchmark_routes'],
        audit_sha256=sha(run/'ARTIFACT_AUDIT_FINAL.json'),final_receipt_sha256=sha(run/'jobs/rc_artifact_audit_final/receipt.json'))
    report.mkdir(exist_ok=True)
    (report/'CLOSURE.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf-8')
    (report/'LOCAL_HASH_VERIFICATION.json').write_text(json.dumps(dict(all_match=True,files=checked),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(closure,indent=2))

if __name__=='__main__':main()
