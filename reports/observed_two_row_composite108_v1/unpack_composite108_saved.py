"""Local extraction and byte validation of the completed saved-only archive."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'.bootstrap/composite108_completed_saved_archive_v1.tar.gz'
DESTINATION=ROOT/'reports/observed_two_row_composite108_v1'
EXPECTED='0b6013f4d511a320772adc1acd342d2bae60781bc548e1786076380f760da92a'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if sha(ARCHIVE)!=EXPECTED:raise ValueError('Downloaded archive SHA differs')
    if DESTINATION.exists():raise FileExistsError('Fresh archive destination required')
    DESTINATION.mkdir()
    with tarfile.open(ARCHIVE) as archive:
        for member in archive.getmembers():
            path=(DESTINATION/member.name).resolve()
            try:path.relative_to(DESTINATION.resolve())
            except ValueError:raise ValueError('Archive path escapes destination')
            if not member.isfile() or member.issym() or member.islnk():raise ValueError('Only regular artifact files allowed')
            path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,path.open('wb') as target:shutil.copyfileobj(source,target)
    index=json.loads((DESTINATION/'REMOTE_ARTIFACT_INDEX.json').read_text())
    for row in index['files']:
        path=DESTINATION/row['path']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Original artifact bytes differ')
    for name in ('archive_composite108_final.py','analyze_composite108_saved.py','unpack_composite108_saved.py','composite108_progress.py'):
        shutil.copyfile(ROOT/'.bootstrap'/name,DESTINATION/name)
    receipt=dict(archive_sha256=EXPECTED,bytes=ARCHIVE.stat().st_size,original_files=len(index['files']),
        all_original_files_hash_matched=True,checkpoint_index=index['checkpoint_index'],checkpoint_bytes_downloaded=False,
        new_forward_requests=0,new_qwen_encodings=0,new_searches=0,no_raw_corpus_reads=True)
    (DESTINATION/'SYNC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
