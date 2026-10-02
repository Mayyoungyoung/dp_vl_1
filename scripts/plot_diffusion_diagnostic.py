"""Plot recorded denoising errors; no models, inference or new evaluation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/multigate_diffusion_diagnostic_v1/fixed_train32_dev128/diagnostic.json")
    parser.add_argument("--output", default="reports/multigate_diffusion_diagnostic_v1/denoising_diagnostic.png")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for name, record in data["checkpoints"].items():
        if "denoise" not in record:
            continue
        rows = record["denoise"]["DEV128"]["rows"]
        correct = [r for r in rows if r["condition"] == "correct"]
        shuffled = [r for r in rows if r["condition"] == "shuffled"]
        times = [r["timestep"] for r in correct]
        axes[0].plot(times, [r["x0_rmse"] for r in correct], marker="o", label=name)
        axes[1].plot(times, [r["clip_fraction"] for r in correct], marker="o", label=name)
        axes[2].plot(times, [b["x0_rmse"] / a["x0_rmse"] for a, b in zip(correct, shuffled)], marker="o", label=name)
    axes[0].set(yscale="log", ylabel="Raw x0 RMSE (log scale)")
    axes[1].set(ylabel="Fraction of x0 coordinates clipped")
    axes[2].set(ylabel="Shuffled / correct condition x0 RMSE")
    for ax in axes:
        ax.set(xlabel="Diffusion timestep")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed DEV128 diagnostic: terminal error amplification; condition affects denoising")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
