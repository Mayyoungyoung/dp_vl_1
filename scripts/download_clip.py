"""Download a pinned official CLIP snapshot for offline server use (about 605MB)."""
import argparse
import concurrent.futures
import json
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/clip-vit-base-patch32')
    p.add_argument('--revision',default='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
    a=p.parse_args()
    repo='openai/clip-vit-base-patch32'
    revision=a.revision
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'provenance.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old.get('revision')!=revision:
            raise ValueError('Use a different output directory for another revision')
    names=['config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json','tokenizer.json','vocab.json','merges.txt','pytorch_model.bin']
    def download(name):
        dest=out/name
        if dest.exists() and dest.stat().st_size>0:
            return str(dest)+' already exists'
        with urllib.request.urlopen('https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name,timeout=60) as response:
            with open(str(dest)+'.partial','wb') as f:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk: break
                    f.write(chunk)
        Path(str(dest)+'.partial').replace(dest)
        return name+' downloaded '+str(dest.stat().st_size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(download,names): print(result,flush=True)
    (out/'provenance.json').write_text(json.dumps({'model':repo,'revision':revision},indent=2),encoding='utf-8')
    print('revision '+revision,flush=True)


if __name__=='__main__': main()
