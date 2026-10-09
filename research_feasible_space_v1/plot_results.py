"""Static publication figures from completed, hashed result JSON only."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root, output):
    tic=time.monotonic()
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = {}

    def read(relative):
        p = root / relative
        source[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text(encoding='utf-8'))

    stats = read('three_seed_statistics_v1/RESULTS.json')
    summary = read('summary_v3/RESULTS.json')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    kinds = ['xyz', 'bounded', 'refitted_projection', 'refitted_center', 'refitted_boundcenter']
    labels = ['Free XYZ', 'Bounded', 'Projection', 'XYZ center', 'Bounded center']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for seed in range(3):
        values = [stats['seed_results'][k][seed]['U8'] for k in kinds]
        axes[0].plot(range(5), values, 'o-', alpha=.75, lw=1, label='Continuation seed %d' % seed)
    axes[0].axhline(6.9806, color='.4', ls='--', lw=1, label='Historical C + success (head seeds)')
    axes[0].axhline(7.238, color='.6', ls=':', lw=1, label='Historical Gate (reference)')
    axes[0].set_xticks(range(5), labels, rotation=18)
    axes[0].set_ylabel('Distinct actual valid modes @8')
    axes[0].legend(fontsize=7, loc='lower left')
    axes[0].set_title('All three full generator continuations')
    comparisons = ['xyz', 'refitted_projection', 'refitted_center', 'refitted_boundcenter']
    for i, k in enumerate(comparisons):
        c = stats['comparisons']['bounded minus ' + k]
        mean, (low, high) = c['mean'][0], c['crossed_CI95'][0]
        axes[1].plot([low, high], [i, i], color='#235c98', lw=2)
        axes[1].plot(mean, i, 'o', color='#235c98')
    axes[1].axvline(0, color='.4', lw=1)
    axes[1].axvline(.15, color='#ae4d28', ls='--', lw=1, label='Practical gain gate: +0.15')
    axes[1].set_yticks(range(4), ['Free XYZ', 'Projection', 'XYZ center', 'Bounded center'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Bounded minus control: valid modes @8')
    axes[1].legend(fontsize=8)
    axes[1].set_title('Conditional seed × family bootstrap 95% intervals')
    fig.suptitle('288 reused DEV_MODEL requests / 32 families; shared pretrained C0', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / 'three_seed_comparison.png')
    fig.savefig(output / 'three_seed_comparison.pdf')
    plt.close(fig)

    names = ['screen_bounded_seed0', 'refreshed_envelope_bounded_seed0', 'refreshed_tapered_bounded_seed0']
    xlabels = ['Path loss only\nUniform cells', 'Full envelope loss\nUniform cells', 'Reachable envelope loss\nTapered cells']
    vals = [summary['models'][n] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, label, color in [('corridor_feasibility', 'Whole predicted-region geometry feasibility', '#235c98'),
                              ('containment', 'Path membership in predicted region', '#477f44')]:
        # The saved metric names are explicit and checked, never inferred from a plot.
        numbers = [v[key] for v in vals]
        axes[0].plot(range(3), np.array(numbers) * 100, 'o-', color=color, label=label)
    axes[0].plot(range(3), [v['raw']['valid_fraction'] * 100 for v in vals], 'o-', color='#ae4d28', label='Actual task route validity @8')
    axes[0].set_ylabel('Percent')
    axes[0].set_ylim(0, 105)
    axes[0].set_title('Containment and task validity are different')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[1].plot(range(3), [v['raw']['distinct'] for v in vals], 'o-', label='Raw valid modes @8')
    axes[1].plot(range(3), [v['selected']['distinct'] for v in vals], 'o-', label='Returned valid modes @4')
    axes[1].set_ylabel('Distinct actual valid modes')
    axes[1].set_title('Two measured-cause repairs, seed 0')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(3), xlabels, fontsize=8)
    fig.suptitle('Exploratory seed-0 sequence; matching success heads except initial screen', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / 'repair_diagnostics.png')
    fig.savefig(output / 'repair_diagnostics.pdf')
    plt.close(fig)
    manifest = {'sources_sha256': source, 'scope': stats['scope'], 'locked_access': False,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'runtime': {'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__},
                'elapsed_seconds':time.monotonic()-tic,
                'command': [sys.executable]+sys.argv}
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    main(**vars(p.parse_args()))
