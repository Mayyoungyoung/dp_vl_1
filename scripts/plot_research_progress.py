"""Plot measured development evidence only; requires observation_eval_v2.

Run locally: python scripts/plot_research_progress.py --root .
No generated illustrations, predicted results, or independent-seed claims.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cached-run", default="observed_frozen_v1_seed0")
    args = parser.parse_args()
    root, sources = args.root.resolve(), {}

    def read(relative):
        path = root / relative
        payload = path.read_bytes()
        sources[relative] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    controlled = {row["objective"]: row for row in read("reports/v2_controlled/results.json")}
    completion = read("reports/v2_completion/three_seed_analysis.json")
    before = {(row["mechanism"], row["mode"]): row for row in read("reports/v2_completion/rollout_seed0/results.json")}
    after = {(row["mechanism"], row["mode"]): row for row in read("reports/v2_completion_selfdraft/rollout_seed0/results.json")}
    observed_ids = [args.cached_run, "observed_online_v1__frozen_seed0", "observed_online_v1__lora_seed0",
                    "observed_online_warm_v2__frozen_seed0", "observed_online_warm_v2__lora_seed0"]
    observed = [read("reports/observation_eval_v2/" + identifier + "/metrics.json") for identifier in observed_ids]
    grounded = read('reports/observation_geometry_three_seed.json')
    for row in observed:
        if (row.get("evaluation_protocol") != "observation_eval_v2" or row.get("examples") != 24
                or row.get("reference_evaluation_examples") != 23 or row.get("semantic_evaluation_examples") != 24):
            raise ValueError("Figure requires common v2 protocol: 24 semantic instructions / 23 reference instructions")
    if completion["seed_ids"] != [0, 1, 2] or completion["parent_count"] != 128:
        raise ValueError("Unexpected completion analysis cohort")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "axes.axisbelow": True, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 11.4))
    fig.subplots_adjust(left=.075, right=.975, top=.835, bottom=.18, hspace=.80, wspace=.28)
    blue, orange, gray, green = "#3476A8", "#D5793A", "#86929B", "#388777"

    def style(ax, title, subtitle):
        ax.set_title(title, loc="left", fontweight="bold", pad=31)
        ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=9, color="#48525A")
        ax.grid(axis="y", color="#DDE2E6", linewidth=.7)
        ax.tick_params(axis="both", length=3, color="#67737D")

    def labels(ax, bars, digits=3, offset=.06):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{bar.get_height():.{digits}f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 0]
    objectives = ["subset", "positive", "saturation"]
    values = [controlled[key]["UniqueValidAtK"] for key in objectives]
    bars = ax.bar(np.arange(3), values, width=.57, color=[gray, blue, green], zorder=3)
    labels(ax, bars)
    ax.set_xticks(np.arange(3), ["Subset\nmatching", "All-positive\nmatching", "Saturation-aware\nassignment"])
    ax.set_ylim(0, 4.0)
    ax.set_ylabel("UniqueValid@4 (routes)")
    for index, key in enumerate(objectives):
        ax.text(index, .18, f"Valid: {100 * controlled[key]['ValidAtK']:.1f}%", ha="center", color="white", fontsize=9)
    style(ax, "A  Strengthening the set-regression baseline", "128 parent scenes; seed 0; 768k training target slots each")
    ax.text(0, -.24, "Known geometry and endpoints; loss correction is not a novelty claim.",
            transform=ax.transAxes, fontsize=8.6, color="#48525A")

    ax = axes[0, 1]
    values = np.asarray(completion["primary"]["by_seed"]["values"])
    mean, sd = values.mean(), values.std(ddof=1)
    ax.axhline(0, color="#48525A", linewidth=1)
    ax.vlines(np.arange(3), 0, values, color=[blue if value >= 0 else orange for value in values], linewidth=3)
    ax.scatter(np.arange(3), values, s=75, color=[blue if value >= 0 else orange for value in values], zorder=3)
    for index, value in enumerate(values):
        ax.text(index, value + (.011 if value >= 0 else -.013), f"{value:+.4f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=10)
    ax.axhline(mean, color=green, linestyle="--", linewidth=1.3)
    ax.text(.98, .96, f"Seed mean {mean:+.4f}; sample SD {sd:.4f}", ha="right", va="top",
            transform=ax.transAxes, fontsize=9, color=green)
    ax.set_xticks(np.arange(3), ["Seed 0", "Seed 1", "Seed 2"])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(-.10, .23)
    ax.set_ylabel("Additional valid types: coverage − attention")
    style(ax, "B  Initial completion advantage is not stable", "128 paired parents; average over five nonempty draft contexts")
    ax.text(0, -.24, "Points are real training seeds. Parent bootstrap is conditional on these seeds;\nno across-seed confidence interval is implied.",
            transform=ax.transAxes, fontsize=8.6, color="#48525A")

    ax = axes[1, 0]
    x, width = np.arange(2), .23
    specs = [("Reference-trained 2+2", gray, -.25, before, "self2_then2"),
             ("Self-draft mixed 2+2", blue, 0, after, "self2_then2"),
             ("Same mixed checkpoint: joint4", green, .25, after, "joint4")]
    for name, color, shift, records, mode in specs:
        bars = ax.bar(x + shift, [records[key, mode]["UniqueValidAtK"] for key in ("attention", "coverage")],
                      width=width, color=color, label=name, zorder=3)
        labels(ax, bars, digits=3, offset=.045)
    ax.set_xticks(x, ["Attention", "Coverage (max pool)"])
    ax.set_ylim(0, 4.25)
    ax.set_ylabel("UniqueValid@4 (routes)")
    style(ax, "C  Self-draft mixing repairs much of the gap", "128 parents; seed 0; strict total budget of four generated routes")
    ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=.95, facecolor="white", edgecolor="#DDE2E6", fontsize=8.5)
    ax.text(0, -.28, "2+2 uses two forwards; joint4 uses one. Neither mixed 2+2 model beats\nits joint4 control. This known training repair is not the core method.",
            transform=ax.transAxes, fontsize=8.6, color="#48525A")

    ax = axes[1, 1]
    x = np.arange(3)
    semantic = grounded['metrics']['semantic_goal_accuracy']
    for name, shift, color, label in [('plain', -.18, gray, 'Ordinary RGB-D head'),
                                     ('aux', .18, green, '+ Endpoint-attention supervision')]:
        values = 100*np.asarray(semantic[name]['seed_values'])
        bars = ax.bar(x+shift, values, width=.33, color=color, label=label, zorder=3)
        labels(ax, bars, digits=1, offset=.8)
    ax.set_xticks(x, ['Seed 0', 'Seed 1', 'Seed 2'])
    ax.set_ylim(0, 49)
    ax.set_ylabel('Strict semantic goal accuracy (%)')
    style(ax, 'D  Grounding repair improves three training seeds', '8 DEV parents; all 24 instructions; 128k training target slots each')
    ax.legend(loc='upper right', frameon=False, fontsize=8.5)
    ax.text(0, -.28, 'Real frozen Qwen + observed RGB-D; same architecture and data.\nStrict target identity + 3 cm threshold. Conventional baseline repair;\nfull-path validity and independent confirmation remain unmeasured.',
            transform=ax.transAxes, fontsize=8.6, color="#48525A")

    fig.suptitle("Research progress: measured development evidence", x=.075, ha="left", fontsize=18, fontweight="bold", y=.973)
    fig.text(.075, .925, "Measured baseline repairs; a stable task-level route-set mechanism remains unproved.", fontsize=11, color="#48525A")
    fig.text(.075, .029, "A–C: oracle-geometry controlled tier. D: observed RGB-D + language + current state. All panels use DEV-selected checkpoints; no locked-test claim.\nTraining costs with different thread counts or cached/live encoding scopes are not compared in this figure.",
             fontsize=9, color="#48525A", linespacing=1.5)
    output = root / "reports" / "research_progress"
    metadata = {"Title": "Research progress: measured development evidence",
                "Description": json.dumps({"sources_sha256": sources,
                    "observation_protocol": "observation_eval_v2",
                    "scope": "DEV results only; no uncertainty over new training seeds or robot execution claim"}, sort_keys=True)}
    fig.savefig(output.with_suffix(".svg"), metadata=metadata)
    svg_path = output.with_suffix('.svg')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    fig.savefig(output.with_suffix(".png"), dpi=220, metadata={"Title": metadata["Title"], "Description": metadata["Description"]})
    plt.close(fig)
    print(json.dumps({"outputs": [str(output.with_suffix(".png")), str(output.with_suffix(".svg"))],
                      "sources_sha256": sources}, indent=2))


if __name__ == "__main__":
    main()
