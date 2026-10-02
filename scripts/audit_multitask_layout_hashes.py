"""Compare pre-saved mechanical hashes only, including across dataset roots.

Never opens reference.json, images, observations.npz, routes or task results.
For legacy collection, only the mechanical reference_pointer image hash is
available; absence of a layout hash must stay unknown.
"""
import argparse
import json
from pathlib import Path
from routeset.multitask_fingerprints import audit_fingerprints


def read_mechanical_rows(sources):
    rows=[];missing=[]
    for source in sources:
        source=Path(source).resolve();plan=json.loads((source/'partition_manifest.json').read_text())
        for spec in plan['parents']:
            folder=source/spec['split']/'parents'/spec['parent_id']
            metadata=folder/'mechanical_fingerprint.json';pointer=folder/'reference_pointer.json'
            if metadata.exists():
                row=json.loads(metadata.read_text())
            elif pointer.exists():
                saved=json.loads(pointer.read_text())
                row={key:spec[key] for key in ('parent_id','task','split')}
                row.update(rgb_file_sha256=saved['image_sha256'],physical_layout_status='legacy_missing_hash_unverified')
            else:
                missing.append(dict(source_root=str(source),parent_id=spec['parent_id'],split=spec['split']))
                continue
            for key in ('parent_id','task','split'):
                if row[key]!=spec[key]:raise ValueError('Mechanical metadata identity/role differs from registration')
            rows.append(dict(row,source_root=str(source)))
    return rows,missing


def audit(sources):
    rows,missing=read_mechanical_rows(sources)
    result=audit_fingerprints(rows)
    result.update(source_roots=[str(Path(source).resolve()) for source in sources],
        parents_with_physical_hash=sum('physical_layout_sha256' in row for row in rows),
        parents_with_only_legacy_image_hash=sum('physical_layout_sha256' not in row for row in rows),
        missing_mechanical_metadata=missing,
        scope='Only registration, mechanical_fingerprint.json and reference_pointer.json; no images, paths, reference world or outcomes opened, including locked roles')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior mechanical audit')
    result=audit(args.source);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('inspected_mechanical_fingerprints','parents_with_physical_hash','parents_with_only_legacy_image_hash','task_usage_status')}))
