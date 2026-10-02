"""Download a pinned official Qwen snapshot; verify official blob/LFS hashes.

Network mirrors may transport bytes, but the checked-in manifest comes from the
official Hugging Face model API. Existing valid files are reused. No credentials
or tokens are required for this public snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def verify(path, entry):
    if not path.exists() or path.stat().st_size != entry['size']:
        return False
    h = hashlib.sha256() if 'lfs' in entry else hashlib.sha1()
    if 'lfs' not in entry:
        h.update(('blob %d\0' % entry['size']).encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    expected = entry['lfs']['sha256'] if 'lfs' in entry else entry['blobId']
    return h.hexdigest() == expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', default='https://huggingface.co')
    p.add_argument('--manifest', type=Path, default=Path(__file__).with_name('observation_model_manifest.json'))
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for entry in manifest['files']:
        path = a.output / entry['rfilename']
        if not verify(path, entry):
            temp = path.with_name(path.name + '.part')
            url = '%s/%s/resolve/%s/%s' % (a.endpoint.rstrip('/'), manifest['model_id'], manifest['revision'], entry['rfilename'])
            print('Downloading', entry['rfilename'], entry['size'], flush=True)
            subprocess.run(['curl', '--fail', '-L', '--retry', '4', '--connect-timeout', '15', '--speed-time', '90', '--speed-limit', '1024', '--continue-at', '-', '--output', str(temp), url], check=True)
            if not verify(temp, entry):
                raise RuntimeError('Official hash verification failed: ' + str(temp))
            temp.replace(path)
        print('Verified', entry['rfilename'], flush=True)
    provenance = dict(manifest, transport_endpoint=a.endpoint, all_hashes_verified=True, elapsed_seconds=time.time()-started)
    (a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
