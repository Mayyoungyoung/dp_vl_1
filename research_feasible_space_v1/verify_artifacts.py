"""Local byte-for-byte verification of dedicated artifacts and source exports."""
import argparse,hashlib,json,time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(root,index,output):
    tic=time.monotonic();root=Path(root).resolve();index=Path(index).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    d=json.loads(index.read_text(encoding='utf-8'));run=root/'runs/feasible_space_v1';data=root/'data/feasible_space_generalization_v1'
    errors=[];files=0;total=0
    for base,key in ((run,'files'),(data,'data_files')):
        for relative,v in d.get(key,{}).items():
            p=(base/relative).resolve();p.relative_to(base.resolve())
            if not p.is_file():errors.append(dict(file=str(p),reason='missing'));continue
            files+=1;total+=p.stat().st_size
            if p.stat().st_size!=v['bytes'] or digest(p)!=v['sha256']:errors.append(dict(file=str(p),reason='hash_or_size_mismatch'))
    archives={}
    for commit,v in d['sources'].items():
        p=run/'source_exports'/(commit+'.tar');actual=digest(p) if p.exists() else None
        archives[commit]=dict(expected=v['archive_sha256'],actual=actual,equal=actual==v['archive_sha256'])
        if not archives[commit]['equal']:errors.append(dict(file=str(p),reason='source_archive_mismatch'))
    report=dict(all_equal=not errors,files_verified=files,bytes_verified=total,source_archives=archives,errors=errors,
                artifact_index_sha256=digest(index),elapsed_seconds=time.monotonic()-tic,
                command=['verify_artifacts','--root',str(root),'--index',str(index),'--output',str(output)],locked_access=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_archives','errors')}))
    if errors:raise RuntimeError(errors[:10])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--index',required=True);p.add_argument('--output',required=True);main(**vars(p.parse_args()))
