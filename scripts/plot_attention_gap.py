"""Render measured localization diagnostics; do not substitute peak predictions."""
import json
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'reports/observation_attention_gap_new64.json'
    result = json.loads(source.read_text())
    assert result['examples'] == result['semantic_denominator'] == 24
    m = result['means']
    labels = ['Peak attention point\n(diagnostic only)', 'Weighted anchor\n(diagnostic only)',
              'Full neural output\n(4 endpoints)', 'Color prototype\n(1 endpoint)']
    names = ['peak', 'anchor', 'candidate', 'prototype']
    colors = ['#BE7940', '#969FA7', '#36789D', '#448879']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    fig.subplots_adjust(left=.075, right=.975, bottom=.27, top=.76, wspace=.3)
    for ax, suffix, scale, title, ylabel, upper in [
        (axes[0], 'semantic_accuracy', 100, 'Strict localization', 'Identity correct and distance <= 3 cm (%)', 105),
        (axes[1], 'goal_error_m', 100, 'Target-center error', 'Mean distance over all 24 instructions (cm)', 19)]:
        values = [m[name+'_'+suffix]*scale for name in names]
        bars = ax.bar(np.arange(4), values, color=colors, width=.62, zorder=3)
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_x()+bar.get_width()/2, value+upper*.018, f'{value:.1f}', ha='center', fontsize=11)
            if i < 2:
                bar.set_hatch('//'); bar.set_edgecolor('white')
        ax.set_xticks(np.arange(4), labels, fontsize=8.5)
        ax.set_ylim(0, upper); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=14)
        ax.grid(axis='y', color='#E0E5E8', linewidth=.7, zorder=0)
        ax.tick_params(axis='x', length=0)
    fig.text(.075, .94, 'Spatial averaging loses otherwise useful localization', fontsize=17, fontweight='bold')
    fig.text(.075, .875, 'Same 64 TRAIN parents; reused 8-parent / 24-instruction DEV set; one neural training seed.', fontsize=10)
    fig.text(.075, .115,
        'Hatched bars are post-hoc diagnostics from the same frozen-Qwen checkpoint, not replacement benchmark outputs.\n'
        'All target centers are used only after inference for evaluation. Prototype failures include large background outliers.\n'
        'These endpoint checks do not measure full-path collision validity, distinct routes, or robot execution.',
        fontsize=9, color='#47545B', linespacing=1.5)
    output = root/'reports/observation_attention_gap'
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    svg = output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    output.with_suffix('.sources.json').write_text(json.dumps({
        'source':str(source.relative_to(root)), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':'fixed-checkpoint diagnostics and same-data task-specific endpoint baseline; no revised benchmark'}, indent=2)+'\n')
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
