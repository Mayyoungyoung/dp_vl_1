"""Sequential online frozen/LoRA comparison, with identical exposure and inputs."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--head-init')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lora-lr', type=float, default=1e-4)
    args = parser.parse_args()
    root, dataset, output = Path(args.root), Path(args.dataset), Path(args.output)
    failed = 0
    for mode in ('frozen', 'lora'):
        command = [sys.executable, 'scripts/record_job.py', '--output', str(output),
                   '--run-id', mode+'_seed0', '--', sys.executable, '-m', 'scripts.train_observed_lora',
                   '--model', str(root/'data/qwen3-vl-2b-instruct-89644892'),
                   '--observations', str(dataset/'observations.jsonl'),
                   '--supervision', str(dataset/'supervision.jsonl'),
                   '--output', str(output/(mode+'_seed0')), '--adapter-mode', mode,
                   '--steps', str(args.steps), '--eval-every', '100', '--threads', str(args.threads),
                   '--lr', str(args.lr), '--lora-lr', str(args.lora_lr)]
        if args.head_init:
            command += ['--head-init', str(Path(args.head_init).resolve())]
        failed += subprocess.call(command) != 0
    sys.exit(int(failed > 0))


if __name__ == '__main__':
    main()
