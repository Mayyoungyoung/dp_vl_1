"""CPU-only actual pinned-HF greedy config preparation; no model/inputs loaded."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from routeset.common import sha256, write_json
from routeset.vlm_sft_train_probe import explicit_greedy_config, decoding_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh actual-config preflight required')
    import transformers
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    if transformers.__version__ != '4.57.1': raise ValueError('Actual pinned Transformers4.57.1 required')
    training = json.loads((args.training/'config.json').read_text())
    model_path = Path(training['model'])
    if sha256(model_path/'provenance.json') != training['model_provenance_sha256']:
        raise ValueError('Pinned actual backbone provenance changed')
    original = GenerationConfig.from_pretrained(model_path,local_files_only=True)
    effective = explicit_greedy_config(original)
    # This is the exact pinned helper invoked by generate. No model weights,
    # forward, tokenizer, image, route or target labels are needed to call it.
    prepared, unused = GenerationMixin._prepare_generation_config(
        SimpleNamespace(generation_config=original),effective,use_model_defaults=False)
    for key,value in dict(decoding_kwargs(True),max_new_tokens=512).items():
        if getattr(prepared,key) != value: raise ValueError('Actual HF config drift: '+key)
    prepared.validate(strict=True)
    if prepared.get_generation_mode().value != 'greedy_search': raise ValueError('Actual mode is not greedy')
    result = dict(status='completed',model_forwards=0,raw_inputs_opened=0,transformers=transformers.__version__,
        training_config_sha256=sha256(args.training/'config.json'),
        model_generation_config_sha256=sha256(model_path/'generation_config.json'),
        script_sha256=sha256(__file__),helper_sha256=sha256(Path(__file__).resolve().parents[1]/'routeset/vlm_sft_train_probe.py'),
        original=original.to_dict(),explicit=effective.to_dict(),actual_prepared=prepared.to_dict(),
        actual_generation_mode=prepared.get_generation_mode().value,use_model_defaults=False,unused_kwargs=unused)
    write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],mode=result['actual_generation_mode'],model_forwards=0,raw_inputs_opened=0)))


if __name__=='__main__':main()
