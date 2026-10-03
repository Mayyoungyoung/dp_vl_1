"""Server-side read-only final archive; requires both recorded jobs completed.

No torch, raw corpus reads or model calls. Checkpoints are hashed, not packed.
"""
import hashlib
import json
from pathlib import Path
import tarfile

PROJECT=Path('/home/wzy/dpvlm/route_set_v1')
RUN=PROJECT/'runs/observed_two_row_composite108_v1'
SOURCE=PROJECT/'research_v2/releases/71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def main():
    for job in ('train','fixed-last-train'):
        status=read(RUN/(job+'.status.json'))
        if status.get('status')!='completed' or status.get('exit_code')!=0 or status.get('code_commit')!=SOURCE.name:
            raise RuntimeError('Final archive waits for actual completed fixed-source '+job)
    if any(RUN.rglob('*.lock')):raise RuntimeError('Active/stale lock requires review before final archive')
    summary=read(RUN/'peak_seed0/summary.json');receipt=read(RUN/'peak_seed0/composite_training_receipt.json')
    diagnostic=read(RUN/'fixed_last_train/diagnostic_receipt.json')
    if summary['last_step']!=12000 or summary['trajectory_exposures']!=1536000 or diagnostic['new_forward_requests']!=285:
        raise RuntimeError('Final step/exposure/actual TRAIN285 denominator changed')
    if receipt['summary_sha256']!=sha(RUN/'peak_seed0/summary.json'):raise RuntimeError('Summary seal mismatch')
    for name,value in receipt['checkpoint_sha256'].items():
        if Path(name).name!=name or sha(RUN/'peak_seed0'/name)!=value:raise RuntimeError('Checkpoint seal mismatch')
    for row in receipt['prediction_artifacts'].values():
        if sha(Path(row['path']))!=row['sha256']:raise RuntimeError('Saved training pool changed')
    for name,value in diagnostic['artifact_sha256'].items():
        if Path(name).name!=name or sha(RUN/'fixed_last_train'/name)!=value:raise RuntimeError('Fixed-last pool changed')
    allowed={'.json','.jsonl','.log','.npz','.xml'}
    paths={str(p.relative_to(RUN)).replace('\\','/'):p for p in RUN.rglob('*') if p.is_file() and p.suffix in allowed}
    for name,value in receipt['source_sha256'].items():
        path=(SOURCE/name).resolve()
        try:path.relative_to(SOURCE)
        except ValueError:raise RuntimeError('Executed Python source leaves release')
        if path.suffix!='.py' or sha(path)!=value:
            raise RuntimeError('Executed Python source differs: '+name)
        paths['source/'+name]=path
    policy=SOURCE/'configs/observed_two_row_composite108_training_v1.json'
    if sha(policy)!=receipt['policy_sha256']:raise RuntimeError('Policy source differs')
    paths['source/configs/observed_two_row_composite108_training_v1.json']=policy
    wrapper=PROJECT/'research_v2/incoming/composite108_pipeline_71cf0c1.sh'
    paths['source/composite108_pipeline_71cf0c1.sh']=wrapper
    index=dict(protocol='composite108_completed_saved_artifact_archive_v1',source_commit=SOURCE.name,
        run=str(RUN),no_new_forward=True,no_raw_corpus_reads=True,
        files=[dict(path=name,source=str(path),sha256=sha(path),bytes=path.stat().st_size) for name,path in sorted(paths.items())],
        checkpoint_index=[dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,packed=False)
                          for path in sorted((RUN/'peak_seed0').glob('*.pt'))])
    destination=PROJECT/'research_v2/incoming/composite108_completed_saved_archive_v1.tar.gz'
    if destination.exists():raise FileExistsError('Fresh final archive required')
    import io
    blob=json.dumps(index,indent=2).encode()
    with tarfile.open(destination,'w:gz') as archive:
        for name,path in sorted(paths.items()):archive.add(path,arcname=name,recursive=False)
        info=tarfile.TarInfo('REMOTE_ARTIFACT_INDEX.json');info.size=len(blob);archive.addfile(info,io.BytesIO(blob))
    print(json.dumps(dict(archive=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,
                         files=len(paths),checkpoint_count=len(index['checkpoint_index']),new_forward_requests=0)))


if __name__=='__main__':main()
