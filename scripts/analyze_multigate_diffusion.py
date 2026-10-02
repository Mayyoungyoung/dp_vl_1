"""Read completed diffusion artifacts; parent-paired DEV analysis and plots.

No model inference, locked data, or training. Three sampling repeats remain
separate candidate sets and do not represent three independent training seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.multigate import load_dataset, route_metrics, wall_boxes


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parent_metrics(paths, data, ids):
    result = route_metrics(paths, data["scenes"][ids],
        reference_modes=data["modes"][ids], reference_mask=data["path_mask"][ids])
    rows = result["per_scene"]
    values = dict(valid_rate=rows["valid"].mean(1), any_valid=rows["valid"].any(1),
        unique_valid=rows["unique_count"], reference_coverage=rows["reference_coverage"],
        collision_rate=rows["collision"].mean(1), mean_length=rows["lengths"].mean(1))
    parents = np.unique(data["parent_ids"][ids])
    aggregated = {key: np.array([np.mean(value[data["parent_ids"][ids] == p]) for p in parents])
                  for key, value in values.items()}
    return parents, aggregated


def paired_difference(a, b, seed=20261002):
    difference = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(difference), size=(10000, len(difference)))
    bootstrap = difference[draws].mean(1)
    return dict(mean=float(difference.mean()), parent_bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
        positive_parents=int((difference > 1e-10).sum()), negative_parents=int((difference < -1e-10).sum()),
        equal_parents=int((np.abs(difference) <= 1e-10).sum()),
        uncertainty_scope="conditional DEV parent bootstrap after averaging separate sampling repeats; not training-seed uncertainty")


def load_predictions(path, data, ids, expected_k):
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["scene_ids"], data["scene_ids"][ids]):
            raise ValueError("prediction scene order mismatch: " + str(path))
        if not np.array_equal(archive["parent_ids"], data["parent_ids"][ids]):
            raise ValueError("prediction parent order mismatch")
        paths = archive["paths"].copy()
    if paths.shape != (len(ids), expected_k, data["paths"].shape[-2], 3):
        raise ValueError("candidate-budget or path-shape mismatch")
    return paths


def analyze(args):
    data = load_dataset(args.data)
    if set(data["splits"]) != {"TRAIN", "DEV_MODEL"}:
        raise ValueError("development-only data required")
    train_ids, ids = [np.flatnonzero(data["splits"] == s) for s in ("TRAIN", "DEV_MODEL")]
    if set(data["parent_ids"][train_ids]) & set(data["parent_ids"][ids]):
        raise ValueError("parent leakage")
    expected_hash = sha256(args.data)
    runs, regression = Path(args.runs), Path(args.regression)
    metadata = Path(args.metadata) if args.metadata else runs
    baseline_metadata = Path(args.regression_metadata) if args.regression_metadata else regression
    baseline_config, baseline_summary = read(baseline_metadata / "config.json"), read(baseline_metadata / "summary.json")
    if baseline_config["dataset_sha256"] != expected_hash:
        raise ValueError("regression data mismatch")
    baseline_path = regression / "dev_model/predictions.npz"
    if sha256(baseline_path) != baseline_summary["prediction_sha256"]:
        raise ValueError("regression prediction hash mismatch")
    baseline_predictions = load_predictions(baseline_path, data, ids, 4)
    parents, baseline_values = parent_metrics(baseline_predictions, data, ids)
    summaries, configs, histories, artifacts, arm_values, examples = {}, {}, {}, {}, {}, {}
    for arm in ("independent", "set_diffusion"):
        folder = metadata / arm
        status = read(folder / "status.json")
        if status["status"] != "completed" or status["exit_code"] != 0:
            raise ValueError("arm not completed: " + arm)
        summary, config = read(folder / "summary.json"), read(folder / "config.json")
        if config["dataset_sha256"] != expected_hash or config["seed"] != 0:
            raise ValueError("data/seed mismatch")
        summaries[arm], configs[arm], histories[arm] = summary, config, read(folder / "history.json")
        artifacts[arm] = {name: sha256(folder / name) for name in ("summary.json", "config.json", "history.json", "status.json")}
        arm_values[arm] = {}
        for key, entry in summary["metrics"].items():
            k = int(key[1:])
            repeats = []
            for repeat in range(config["final_repeats"]):
                relative = Path("dev_model") / key / ("repeat%d" % repeat)
                path = runs / arm / relative / "predictions.npz"
                if sha256(path) != summary["prediction_sha256"][relative.as_posix()]:
                    raise ValueError("prediction hash mismatch: " + str(path))
                predictions = load_predictions(path, data, ids, k)
                p, metrics = parent_metrics(predictions, data, ids)
                np.testing.assert_array_equal(p, parents)
                for name in metrics:
                    if not np.isclose(metrics[name].mean(), entry["repeats"][repeat][name], atol=1e-12):
                        raise ValueError("saved/recomputed metric mismatch: " + name)
                repeats.append(metrics)
                if k == 4 and repeat == 0:
                    examples[arm] = predictions
            arm_values[arm][key] = {name: np.stack([r[name] for r in repeats]) for name in repeats[0]}
    for key in ("training_stream_sha256", "trajectory_exposures", "reference_pool_access"):
        if summaries["independent"][key] != summaries["set_diffusion"][key]:
            raise ValueError("actual paired stream/exposure mismatch: " + key)
    for key in ("dataset_sha256", "steps", "batch_size", "candidates", "seed", "width", "depth", "lr",
                "eval_every", "diffusion_steps", "sampling_steps", "eval_candidates", "final_repeats", "source_hashes"):
        if configs["independent"][key] != configs["set_diffusion"][key]:
            raise ValueError("paired configuration mismatch: " + key)
    parameterizations = {config.get("parameterization", "epsilon") for config in configs.values()}
    if len(parameterizations) != 1:
        raise ValueError("paired parameterization mismatch")
    diffusion_slots = summaries["independent"]["trajectory_exposures"]
    regression_slots = baseline_summary["trajectory_exposures"]
    same_training_exposure = diffusion_slots == regression_slots
    if not same_training_exposure and not getattr(args, "allow_unequal_training_exposure", False):
        raise ValueError("regression/diffusion gradient-slot mismatch")
    if "continuation" in configs["independent"]:
        for key in ("incremental_stream_sha256", "incremental_trajectory_exposures", "incremental_reference_pool_access"):
            if summaries["independent"][key] != summaries["set_diffusion"][key]:
                raise ValueError("continuation segment mismatch: " + key)
    comparisons = {}
    for arm in arm_values:
        comparisons[arm + "_minus_regression_k4"] = {name: paired_difference(value.mean(0), baseline_values[name])
            for name, value in arm_values[arm]["k4"].items()}
    comparisons["set_minus_independent_k4"] = {name: paired_difference(value.mean(0), arm_values["independent"]["k4"][name].mean(0))
        for name, value in arm_values["set_diffusion"]["k4"].items()}
    reference_counts = np.array([np.mean(data["path_mask"][ids[data["parent_ids"][ids] == p]].sum(1)) for p in parents])
    strata = {}
    for count in np.unique(reference_counts):
        mask = reference_counts == count
        strata[str(count)] = dict(parents=int(mask.sum()), regression=float(baseline_values["unique_valid"][mask].mean()),
            **{arm: float(arm_values[arm]["k4"]["unique_valid"][:, mask].mean()) for arm in arm_values})
    result = dict(protocol="controlled multigate, seed0 development screening; no locked data; three sampling repeats per trained arm",
        data_sha256=expected_hash, parents=len(parents), actual_paired_stream_check="passed",
        summaries=summaries, configs=configs, histories=histories, regression=baseline_summary,
        regression_parent_metrics={key: float(value.mean()) for key, value in baseline_values.items()},
        comparisons=comparisons, known_reference_count_strata=strata, artifact_hashes=artifacts,
        same_training_exposure=same_training_exposure,
        diffusion_to_regression_exposure_ratio=diffusion_slots / regression_slots,
        fairness=("same data/information; diffusion %d versus regression %d target slots (%.1fx); "
                  "regression freely matches full positive pool, diffusion samples balanced positives; "
                  "different active parameters/forwards and CPU allocations") %
                  (diffusion_slots, regression_slots, diffusion_slots / regression_slots),
        timing_caution="old regression head latency excludes exact checker; diffusion measured request includes transfer and checker; do not quote speedup as identical timing scope",
        train_cpu_caution="historical regression trainer used4 torch CPU threads; diffusion uses1, so wall-clock training cost is measured but not a compute-matched superiority test",
        metadata_source=str(metadata), binary_source=str(runs), regression_metadata_source=str(baseline_metadata),
        parameterization=next(iter(parameterizations)))
    write_json(args.output, result)
    destination = Path(args.figures)
    destination.mkdir(parents=True, exist_ok=True)
    plot_results(destination, result, arm_values, data, ids, examples, baseline_predictions, parents, baseline_values)
    Path(args.report).write_text(markdown(result), encoding="utf-8")
    return result


def markdown(result):
    lines = ["# Multigate扩散对照：seed0开发筛选", "", result["protocol"], "",
        "训练参数化：`%s`。不同参数化的训练loss不可直接比较数值。" % result.get("parameterization", "epsilon"), "",
        "以下3次重复为同一训练checkpoint的独立采样，不是3个训练种子；每次只生成K条后单独评价。", "",
        "| 方法 | K | Valid | AnyValid | UniqueValid | ReferenceCoverage | 最优步 | 目标槽 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    b = result["regression_parent_metrics"]
    lines.append("| saturation regression | 4 | %.5f | %.5f | %.5f | %.5f | %d | %d |" %
        (b["valid_rate"], b["any_valid"], b["unique_valid"], b["reference_coverage"], result["regression"]["best_step"], result["regression"]["trajectory_exposures"]))
    for arm, summary in result["summaries"].items():
        for key, item in summary["metrics"].items():
            m = item["mean"]
            lines.append("| %s | %s | %.5f | %.5f | %.5f | %.5f | %d | %d |" %
                (arm, key[1:], m["valid_rate"], m["any_valid"], m["unique_valid"], m["reference_coverage"], summary["best_step"], summary["trajectory_exposures"]))
    exposure_note = ("K4为同目标槽训练预算的主比较。" if result["same_training_exposure"] else
        "扩散累计训练目标槽为回归的%.1f倍；两扩散臂彼此等预算，与历史回归不是等总训练预算比较。" % result["diffusion_to_regression_exposure_ratio"])
    lines += ["", exposure_note + "扩散K1/2/8是未见K推断转移；普通固定K4回归无K8结果，表中不造对照。", "",
        "## K4父场景配对差", "", "| 对比 | UniqueValid差 | 条件父bootstrap95% | Valid差 |", "|---|---:|---|---:|"]
    for label, comparison in result["comparisons"].items():
        u = comparison["unique_valid"]
        lines.append("| %s | %.6f | [%.6f, %.6f] | %.6f |" % (label, u["mean"], *u["parent_bootstrap95"], comparison["valid_rate"]["mean"]))
    lines += ["", "区间只重采样当前DEV父场景，不能量化训练种子不确定性或将已用于选择的DEV变成测试。", "",
        "## 收敛与成本", "", "| 方法 | 本输出elapsed秒 | 本输出GPU小时 | active参数 | 最后3次DEV UniqueValid | 最后3次训练loss |", "|---|---:|---:|---:|---|---|"]
    for arm, summary in result["summaries"].items():
        tail = result["histories"][arm][-3:]
        lines.append("| %s | %.3f | %.6f | %d | %s | %s |" % (arm, summary["elapsed_s"], summary["gpu_hours_reserved"], summary["active_parameters"],
            ", ".join("%d:%.4f" % (x["step"], x["dev_model"]["unique_valid"]) for x in tail),
            ", ".join("%.5f" % x["loss"] for x in tail)))
    lines += ["", "实际配对stream/目标槽/参考池访问检查通过；完整哈希见JSON。两扩散臂同父/目标/时间/噪声，未增加隐藏候选、几何修复或评分筛选。", "",
        "公平范围：" + result["fairness"] + "。回归可在整个正例池自由匹配；扩散使用已知正例平衡采样。这是模型目标差异，不等于完全相同目标。", "",
        "历史回归用4个torch CPU线程，扩散用1个；训练墙钟不是同CPU算力优越性证据。旧回归仅测head，而扩散请求计时含生成/传输/checker，不直接宣称同口径加速比。", "",
        "当前没有观测输入、语义目标推断、训练评分器或SelectedValid执行证据。训练末尾仍在改善时，不可称扩散已收敛或给出扩散方法的性能上限。PG强度筛选尚未执行。", ""]
    return "\n".join(lines)


def plot_results(destination, result, values, data, ids, examples, baseline, parents, baseline_values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm, by_k in values.items():
        ks = sorted(int(key[1:]) for key in by_k)
        for ax, metric in zip(axes, ("valid_rate", "unique_valid")):
            per_repeat = np.array([by_k["k%d" % k][metric].mean(1) for k in ks])
            ax.errorbar(ks, per_repeat.mean(1), yerr=per_repeat.std(1), marker="o", label=arm)
            ax.set(xlabel="Returned candidates K", ylabel=metric, xticks=ks)
            ax.grid(alpha=.2)
            ax.scatter([4], [result["regression_parent_metrics"][metric]], color="black", marker="*", s=100)
    axes[0].legend()
    fig.suptitle("DEV seed0; sampling SD, 3 separate repeats; black star: regression K4\nDiffusion/regression training exposure: %.1fx" % result["diffusion_to_regression_exposure_ratio"])
    fig.savefig(destination / "budget_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm, history in result["histories"].items():
        steps = [r["step"] for r in history]
        axes[0].plot(steps, [r["loss"] for r in history], marker="o", label=arm)
        axes[1].plot(steps, [r["dev_model"]["unique_valid"] for r in history], marker="o", label=arm)
    axes[1].axhline(result["regression_parent_metrics"]["unique_valid"], color="black", linestyle="--", label="regression best")
    for ax, label in zip(axes, (result.get("parameterization", "epsilon") + " MSE, tail200", "DEV K4 UniqueValid, fixed noise seed")):
        ax.set(xlabel="Training steps", ylabel=label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("DEV seed0; diffusion/regression cumulative training exposure: %.1fx" % result["diffusion_to_regression_exposure_ratio"])
    fig.savefig(destination / "convergence.png", dpi=170)
    plt.close(fig)
    difference = values["set_diffusion"]["k4"]["unique_valid"].mean(0) - baseline_values["unique_valid"]
    selected = [int(np.argmin(difference)), int(np.argmax(difference))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, parent_pos in enumerate(selected):
        position = int(np.flatnonzero(data["parent_ids"][ids] == parents[parent_pos])[0])
        scene = data["scenes"][ids[position]]
        lower, upper = wall_boxes(scene)
        for col, (name, collection) in enumerate((("regression", baseline), ("independent", examples["independent"]), ("set_diffusion", examples["set_diffusion"]))):
            ax = axes[row, col]
            paths = collection[position]
            metrics = route_metrics(paths[None], scene[None])
            for lo, hi in zip(lower, upper):
                ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], color="0.8"))
            for index, path in enumerate(paths):
                valid = metrics["per_scene"]["valid"][0, index]
                ax.plot(path[:, 0], path[:, 1], color="tab:blue" if valid else "tab:red", alpha=.7,
                        linestyle="-" if valid else "--")
            ax.scatter(scene[[0, 3]], scene[[1, 4]], c=["green", "black"], s=25)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xlabel="x", ylabel="y",
                title="%s: V %.2f U %.0f\n%s" % (name, metrics["validity"], metrics["unique_valid"], parents[parent_pos]))
    fig.suptitle("Lowest / highest set-minus-regression difference (repeat0 shown); red = invalid by full 3D check")
    fig.savefig(destination / "paired_extreme_cases.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/multigate_v1_partitions/development.npz")
    parser.add_argument("--runs", default="runs/multigate_diffusion_v1/seed0")
    parser.add_argument("--metadata", help="Tracked config/summary/history/status export; predictions remain under --runs")
    parser.add_argument("--regression", default="runs/v2_round2/saturation_k4_seed0")
    parser.add_argument("--regression-metadata", help="Tracked regression config/summary export")
    parser.add_argument("--allow-unequal-training-exposure", action="store_true",
                        help="Explicit extra-training diagnostic; prominently label ratio, never claim matched training budget")
    parser.add_argument("--output", default="reports/multigate_diffusion_v1/analysis.json")
    parser.add_argument("--report", default="reports/DIFFUSION_MULTIGATE_RESULTS.md")
    parser.add_argument("--figures", default="reports/multigate_diffusion_v1/figures")
    result = analyze(parser.parse_args())
    print(json.dumps(dict(paired_stream_check=result["actual_paired_stream_check"], comparisons=result["comparisons"])))


if __name__ == "__main__":
    main()
