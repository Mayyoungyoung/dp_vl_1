"""Run the pre-specified draft-shift repair and evaluate the strict four-route rollout."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    output = Path(a.output)
    run = output / ('seed%d' % a.seed)
    train = [sys.executable, 'scripts/record_job.py', '--output', str(output),
             '--run-id', 'selfdraft_seed%d' % a.seed, '--', sys.executable, '-m', 'scripts.train_completion',
             '--data', a.data, '--output', str(run), '--mechanism', 'both', '--steps', '2500',
             '--seed', str(a.seed), '--threads', str(a.threads), '--eval-every', '500',
             '--self-draft-prob', '.5', '--self-draft-start', '1000', '--device', 'cuda']
    code = subprocess.call(train)
    if code == 0:
        code = subprocess.call([sys.executable, '-m', 'scripts.evaluate_completion_rollout',
             '--data', a.data, '--runs', str(run), '--output', str(output/('rollout_seed%d' % a.seed))])
    sys.exit(code)


if __name__ == '__main__':
    main()
