"""Read pinned public model metadata only; never download weights or open datasets."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

MODELS = {
    'yili18/Hamster_dev': '794f1f925c87e861d2f562943e978cc11f8c344d',
    'DAVIAN-Robotics/3D_HAMSTER': 'ddc5987a56cdcb14e5e2297817612532e46e912b',
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous metadata audit')
    rows = []
    for model, revision in MODELS.items():
        url = 'https://huggingface.co/api/models/' + model + '/revision/' + revision + '?blobs=true'
        raw = fetch(url)
        data = json.loads(raw)
        if data['sha'] != revision:
            raise ValueError('Public model revision changed')
        weights = [dict(path=f['rfilename'], bytes=f['size'], published_lfs_sha256=f.get('lfs', {}).get('sha256'))
                   for f in data['siblings'] if f['rfilename'].endswith(('.safetensors', '.bin'))]
        configs = []
        for entry in data['siblings']:
            name = entry['rfilename']
            if name == 'config.json' or name.endswith('/config.json'):
                config_url = 'https://huggingface.co/' + model + '/resolve/' + revision + '/' + urllib.parse.quote(name, safe='/')
                content = fetch(config_url)
                config = json.loads(content)
                configs.append(dict(path=name, url=config_url, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                                    architecture_metadata=config))
        total = sum(f['bytes'] for f in weights)
        rows.append(dict(model=model, revision=revision, metadata_url=url, metadata_response_sha256=hashlib.sha256(raw).hexdigest(),
                         gated=data.get('gated'), weights=weights, weight_bytes=total, weight_gib=total / 2**30,
                         config_metadata=configs, weights_downloaded=False, model_loaded=False,
                         weight_bytes_exceed_current_35pct_gpu_cap=total > 24576 * 2**20 * .35))
    result = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), models=rows,
                  authorized_gpu_memory_fraction=.35, physical_gpu_total_mib=24576,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Public metadata/configs only. LFS checksums are publisher metadata, not local verified weight hashes. '
                        'Serialized weight size comparison is a resource screening calculation, not a measured inference peak. '
                        'Neither model has been reproduced. CPU offload/quantization may change feasibility and cost and have not been tested.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([dict(model=r['model'], revision=r['revision'], weight_gib=r['weight_gib'], model_loaded=False) for r in rows]))


if __name__ == '__main__':
    main()
