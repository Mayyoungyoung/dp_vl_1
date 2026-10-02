"""Plot all paired training seeds, keeping checkpoint protocols separate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / 'reports'
    files = [reports / 'observed_anchor_peak_three_seed_analysis.json',
             reports / 'obstacle_original_best_evaluation/three_seed_summary.json',
             reports / 'obstacle_fixed_step1000_evaluation/three_seed_summary.json']
    natural, original, fixed = [json.loads(path.read_text()) for path in files]
    arrays = [
        [natural['settings']['natural64']['aggregate'][arm]['best']['semantic_goal_accuracy']['values'] for arm in ('soft', 'peak')],
        [original['results'][arm]['TipValidAtK']['seeds'] for arm in ('soft', 'peak')],
        [fixed['results'][arm]['TipValidAtK']['seeds'] for arm in ('soft', 'peak')],
    ]
    titles = ['Natural layouts\nOriginal ADE-selected best',
              'Obstacle layouts\nOriginal ADE-selected best',
              'Obstacle layouts\nFixed step 1,000: exploratory sensitivity']
    labels = ['Strict target correctness (%)', 'Box-only TipValid@4 (%)', 'Box-only TipValid@4 (%)']
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), sharey=True)
    colors = ['#197278', '#dd6b20', '#7756a7']
    for axis, values, title, label in zip(axes, arrays, titles, labels):
        values = np.array(values) * 100
        for seed in range(3):
            axis.plot([0, 1], values[:, seed], 'o-', color=colors[seed], lw=1.8,
                      markersize=6, label='Training seed '+str(seed), alpha=.85)
        means = values.mean(1)
        axis.plot([0, 1], means, 'D', color='#18232e', markersize=7, label='Mean')
        for x, value in enumerate(means):
            axis.text(x, value + 5, f'{value:.1f}', ha='center', color='#18232e', fontsize=10, weight='bold')
        axis.set_xticks([0, 1], ['Soft anchor', 'Peak anchor'])
        axis.set_xlim(-.23, 1.23)
        axis.set_ylim(-5, 105)
        axis.set_title(title, fontsize=10, pad=13)
        axis.set_ylabel(label)
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Observed-point anchoring: paired gains and checkpoint sensitivity', fontsize=14, weight='bold', y=.99)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='lower center', ncol=4, bbox_to_anchor=(.5, .065), frameon=False)
    fig.text(.5, .01, 'Same data, initialization and 128,000 target slots per paired arm. Reused DEV: 8 natural / 4 obstacle parents.\n'
             'Box-only tip checks do not certify whole-arm motion or execution. Fixed-step results do not replace original best results.',
             ha='center', va='bottom', fontsize=8.5)
    fig.tight_layout(rect=[0, .13, 1, .94])
    for extension in ('png', 'svg'):
        fig.savefig(reports / ('observed_anchor_three_seed.'+extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    files += [Path(__file__)]
    provenance = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (reports / 'observed_anchor_three_seed.sources.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
