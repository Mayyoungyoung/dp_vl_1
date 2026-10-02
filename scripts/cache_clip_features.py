"""Cache a real frozen pretrained VLM encoder, with dataset/revision provenance.

CLIP is a compact image-text encoder, not an instruction-following reasoning VLM.
Geometry and task anchors remain privileged inputs in this first prototype.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import model_info

from routeset.common import load_data, sha256, write_json
from routeset.data import render_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/routes.npz')
    p.add_argument('--output', default='data/clip_features.npz')
    p.add_argument('--model', default='openai/clip-vit-base-patch32')
    p.add_argument('--revision')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    if Path(a.model).is_dir():
        import json
        provenance=json.loads((Path(a.model)/'provenance.json').read_text(encoding='utf-8'))
        revision=provenance['revision']
        source_id=provenance['model']
    else:
        revision = a.revision or model_info(a.model).sha
        source_id=a.model
    model = CLIPModel.from_pretrained(a.model, revision=revision).eval().to(a.device)
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(a.model, revision=revision)
    data = load_data(a.data)
    features=[]
    with torch.inference_mode():
        for start in range(0,len(data['scenes']),a.batch_size):
            scenes = data['scenes'][start:start+a.batch_size]
            words = data['instructions'][start:start+a.batch_size].tolist()
            images=[]
            for scene in scenes:
                views=render_scene(scene)
                images.extend([Image.fromarray(v) for v in views])
            image_batch = processor(images=images, return_tensors='pt')['pixel_values'].to(a.device)
            text_batch = processor(text=words, return_tensors='pt', padding=True, truncation=True).to(a.device)
            vi = model.get_image_features(pixel_values=image_batch)
            vi = vi / vi.norm(dim=-1,keepdim=True)
            vi = vi.reshape(len(scenes),2,-1).mean(1)
            vi = vi / vi.norm(dim=-1,keepdim=True)
            vt = model.get_text_features(**text_batch)
            vt = vt / vt.norm(dim=-1,keepdim=True)
            features.append(torch.cat([vi,vt],-1).cpu().numpy().astype(np.float32))
            print('cached %d/%d scenes'%(min(start+a.batch_size,len(data['scenes'])),len(data['scenes'])),flush=True)
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest, features=np.concatenate(features),scene_ids=data['scene_ids'],dataset_sha256=sha256(a.data),model_id=source_id,revision=revision)
    write_json(dest.with_suffix('.json'), {'model':source_id,'revision':revision,'frozen':True,'image_views':2,'features':1024,
                                          'dataset_sha256':sha256(a.data),'condition':'ground truth geometry/anchors plus frozen image-text embedding',
                                          'limitation':'No learned language grounding or semantic generalization demonstrated.'})


if __name__=='__main__':
    main()
