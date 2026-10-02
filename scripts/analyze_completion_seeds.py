"""Three training-seed completion analysis; bootstrap paired parent clusters.

Each bootstrap resamples parents after averaging the same parent's paired
method difference over the fixed training seeds. Seeds x parents are never
flattened into independent observations. Training-seed variation is separate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


KINDS = ("empty", "single", "two_valid", "duplicate", "invalid", "mixed")
METHODS = ("attention", "coverage")
METRICS = ("additional_unique_valid", "new_valid_rate", "new_success", "new_unique_valid",
           "new_reference_coverage", "new_collision_rate", "union_valid_rate", "union_success",
           "union_unique_valid", "union_reference_coverage", "union_collision_rate")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(values=values.tolist(), mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                minimum=float(values.min()), maximum=float(values.max()))


def paired(values, parent_ids, draws):
    """values [seed, example]; same parent index is drawn for all seeds."""
    parents = sorted(set(parent_ids))
    columns = np.asarray([[np.mean(row[np.asarray(parent_ids) == parent]) for parent in parents] for row in values])
    parent_average = columns.mean(0)
    means = parent_average[draws].mean(1)
    return dict(mean_delta=float(parent_average.mean()), by_seed=stats(columns.mean(1)),
                parent_count=len(parents), parent_wins=int((parent_average > 1e-12).sum()),
                parent_ties=int((np.abs(parent_average) <= 1e-12).sum()), parent_losses=int((parent_average < -1e-12).sum()),
                fixed_seed_parent_bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
                positive_training_seeds=int((columns.mean(1) > 0).sum()),
                negative_training_seeds=int((columns.mean(1) < 0).sum()),
                parent_deltas=dict(zip(parents, parent_average.tolist())))


def counts(rows):
    result = dict(new_candidates=0, invalid_new_candidates=0., collision_new_candidates=0.,
                  valid_new_candidates=0., new_unique_types=0., repeated_or_unclassified_valid_candidates=0.,
                  already_covered_new_types=0., additional_types=0., attainable_additional_types=0.,
                  all_known_covered_scenes=0, invalid_scene_ids=[], deficit_scene_ids=[])
    for row in rows:
        k = row["new_candidates"]
        valid = k * row["new_valid_rate"]
        maximum = min(k, max(0, row["reference_types"] - row["provided_unique_valid"]))
        result["new_candidates"] += k
        result["invalid_new_candidates"] += k - valid
        result["collision_new_candidates"] += k * row["new_collision_rate"]
        result["valid_new_candidates"] += valid
        result["new_unique_types"] += row["new_unique_valid"]
        result["repeated_or_unclassified_valid_candidates"] += valid - row["new_unique_valid"]
        result["already_covered_new_types"] += row["new_unique_valid"] - row["additional_unique_valid"]
        result["additional_types"] += row["additional_unique_valid"]
        result["attainable_additional_types"] += maximum
        result["all_known_covered_scenes"] += row["all_known_types_already_covered"]
        if valid < k:
            result["invalid_scene_ids"].append(row["scene_id"])
        if row["additional_unique_valid"] < maximum:
            result["deficit_scene_ids"].append(row["scene_id"])
    result["additional_deficit"] = result["attainable_additional_types"] - result["additional_types"]
    return result


def analyze(args):
    root = Path(args.runs)
    all_rows, records, sources, context_checks, canonical = {}, {}, {}, [], {}
    config_keys = ("dataset_sha256", "steps", "batch_size", "width", "depth", "lr", "eval_every", "horizon", "cond_dim", "objective")
    first_config, example_ids, parent_ids = None, None, None
    for seed in (0, 1, 2):
        folder = root if seed == 0 else root / ("seed%d" % seed)
        records[seed] = {}
        for method in METHODS:
            run = folder / method
            config, summary = read(run / "config.json"), read(run / "summary.json")
            if config["seed"] != seed or config["mechanism"] != method or config["selection_split"] != "DEV_MODEL":
                raise ValueError("Run identity or selection split mismatch")
            expected = {key: config[key] for key in config_keys}
            if first_config is None:
                first_config = expected
            if expected != first_config:
                raise ValueError("Unpaired training configuration")
            if read(run / "status.json")["status"] != "completed":
                raise ValueError("Only completed runs are valid")
            if sha(run / "best.pt") != summary["best_checkpoint_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            records[seed][method] = dict(config=config, summary=summary, contexts={})
            for name in ("config.json", "summary.json", "history.json", "status.json", "best.pt"):
                sources[str(run / name)] = sha(run / name)
            for kind in KINDS:
                file = run / "dev_model" / kind / "per_scene.json"
                rows = read(file)
                ids, parents = [row["scene_id"] for row in rows], [row["parent_id"] for row in rows]
                if len(ids) != len(set(ids)) or any(row["context_kind"] != kind for row in rows):
                    raise ValueError("Duplicate scene IDs or context mismatch")
                if example_ids is None:
                    example_ids, parent_ids = ids, parents
                if ids != example_ids or parents != parent_ids:
                    raise ValueError("Examples/parents are not aligned across methods, contexts or seeds")
                prediction_file = file.with_name("predictions.npz")
                if sha(prediction_file) != summary["prediction_sha256"][kind]:
                    raise ValueError("Prediction hash mismatch")
                with np.load(prediction_file, allow_pickle=False) as prediction:
                    fields = {name: prediction[name].copy() for name in ("drafts", "draft_present", "draft_valid", "scene_ids", "parent_ids")}
                    if kind not in canonical:
                        canonical[kind] = fields
                    if any(not np.array_equal(value, canonical[kind][name]) for name, value in fields.items()):
                        raise ValueError("Actual supplied drafts differ across paired seeds/methods")
                context_checks.append(dict(seed=seed, method=method, context=kind, identical_saved_inputs=True))
                all_rows[seed, method, kind] = rows
                values = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
                for metric, value in values.items():
                    if abs(value - summary["metrics"][kind][metric]) > 1e-10:
                        raise ValueError("Summary differs from per-scene data")
                # Preserve every original field and failure in a compact table.
                columns = list(rows[0])
                records[seed][method]["contexts"][kind] = dict(metrics=values, counts=counts(rows),
                    per_scene_table=dict(columns=columns, rows=[[row[column] for column in columns] for row in rows]))
                sources[str(file)], sources[str(prediction_file)] = sha(file), sha(prediction_file)
    parents = sorted(set(parent_ids))
    draws = np.random.default_rng(args.bootstrap_seed).integers(0, len(parents), (args.bootstrap_repeats, len(parents)))
    context_results = {}
    for kind in KINDS:
        context_results[kind] = {"methods": {}, "paired": {}}
        for method in METHODS:
            context_results[kind]["methods"][method] = {metric: stats([records[seed][method]["contexts"][kind]["metrics"][metric] for seed in (0, 1, 2)]) for metric in METRICS}
        for metric in METRICS:
            values = np.asarray([[right[metric] - left[metric] for left, right in zip(all_rows[seed, "attention", kind], all_rows[seed, "coverage", kind])] for seed in (0, 1, 2)])
            context_results[kind]["paired"][metric] = paired(values, parent_ids, draws)
    combined = np.asarray([[np.mean([all_rows[seed, "coverage", kind][row]["additional_unique_valid"] - all_rows[seed, "attention", kind][row]["additional_unique_valid"] for kind in KINDS if kind != "empty"])
        for row in range(len(parent_ids))] for seed in (0, 1, 2)])
    primary = paired(combined, parent_ids, draws)
    primary["methods"] = {method: stats([records[seed][method]["summary"]["metrics"]["selection_score"] for seed in (0, 1, 2)]) for method in METHODS}
    strata = {}
    for r in sorted({row["reference_types"] for row in all_rows[0, "attention", "two_valid"]}):
        indices = [i for i, row in enumerate(all_rows[0, "attention", "two_valid"]) if row["reference_types"] == r]
        strata[r] = dict(parent_count=len(indices), methods={})
        for method in METHODS:
            strata[r]["methods"][method] = {metric: stats([np.mean([all_rows[seed, method, "two_valid"][i][metric] for i in indices]) for seed in (0, 1, 2)]) for metric in ("additional_unique_valid", "new_valid_rate", "new_unique_valid")}
        strata[r]["paired_delta"] = stats([strata[r]["methods"]["coverage"]["additional_unique_valid"]["values"][seed] - strata[r]["methods"]["attention"]["additional_unique_valid"]["values"][seed] for seed in (0, 1, 2)])
    result = dict(version="completion_three_seed_v1", split="DEV_MODEL", seed_ids=[0, 1, 2],
        analysis_script_sha256=sha(Path(__file__)), example_count=len(example_ids), parent_count=len(parents),
        paired_configuration=first_config, standard_deviation="sample SD across 3 real training seeds; ddof=1",
        bootstrap=dict(unit="parent scene; average paired method difference over fixed seeds within parent first",
            repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, coverage="descriptive percentile 95%; conditional on these trained seeds and selected checkpoints",
            excluded_claims=["not uncertainty over new training seeds", "not locked-test confirmation", "not independent seed x parent samples"]),
        primary=primary, contexts=context_results, two_valid_reference_strata=strata, runs=records,
        identical_context_checks=context_checks, artifact_sha256=sources)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def report(result, path):
    primary, contexts = result["primary"], result["contexts"]
    def mean_sd(item, scale=1):
        return "%.6f ± %.6f" % (scale * item["mean"], scale * item["sample_sd"])
    lines = ["# 补全机制三训练种子开发分析", "", "更新：2026-10-02。全部使用同一 128 个 DEV_MODEL 父场景，训练 seed 0/1/2。来源为本地 `runs/v2_completion/{attention,coverage}`（seed 0）及 `seed1/`、`seed2/`。未读取锁定结果。", "",
        "## 结论", "", "**目前不能判定 coverage 有稳定的跨训练种子优势。** 五种非空补全情境平均新增有效类型的配对差为 seed 0 **+0.1640625**、seed 1 **+0.015625**、seed 2 **−0.046875**。第一轮积极信号在新增种子明显缩小并出现反转；不能只保留 seed 0 或将父场景区间当作种子稳定性。", "",
        "attention 三种子平均为 **%s**，coverage 为 **%s**；配对差 **%s**（均值 ± 训练种子样本标准差，ddof=1）。" % (mean_sd(primary["methods"]["attention"]), mean_sd(primary["methods"]["coverage"]), mean_sd(primary["by_seed"])), "",
        "父场景配对 bootstrap 区间为 **[%.6f, %.6f]**；先在同父场景内平均三个训练种子，再重采 128 个父场景，未将 384 个种子×父场景当独立样本。该区间只描述这三个已训练且经 DEV 选优模型在当前父场景分布上的平均差，不能覆盖重新训练的不确定性。" % tuple(primary["fixed_seed_parent_bootstrap_95"]), "",
        "## 每种子及预算", "", "| seed | attention 选择分数 | coverage 选择分数 | 配对差 | attention / coverage 最优步 |", "|---:|---:|---:|---:|---|" ]
    for seed in (0, 1, 2):
        a, c = result["runs"][seed]["attention"]["summary"], result["runs"][seed]["coverage"]["summary"]
        lines.append("| %d | %.6f | %.6f | %+.6f | %d / %d |" % (seed, a["metrics"]["selection_score"], c["metrics"]["selection_score"], c["metrics"]["selection_score"] - a["metrics"]["selection_score"], a["best_step"], c["best_step"]))
    lines += ["", "各训练 2500 更新、batch 64、400,000 反传目标槽、403,522 参数，DEV 每 500 步选优。输入、草稿、gate、父场景顺序和分配目标已核验配对；36 组保存草稿数组跨方法/种子逐元素相同。seed 0 与 seed 1/2 的相关实现差异仅为 multigate 的说明文字修正，训练/评价逻辑未改变。每项配置、checkpoint/prediction 哈希及每一行原始逐场景指标均保留于 JSON。", "",
        "empty 是 0 草稿+4 新候选，single 是 1+2，总候选 3；其余是 2+2，总候选 4。无效/重复草稿仍计预算。本表是给定参考草稿的补全，不是自产草稿闭环或观测机器人结果。", "",
        "## 各情境三种子结果", "", "| 情境 | attention 新增均值±SD | coverage 新增均值±SD | 配对差 seed0 / seed1 / seed2 | 父场景bootstrap95% |", "|---|---|---|---|---|" ]
    for kind in KINDS:
        item = contexts[kind]
        paired_result = item["paired"]["additional_unique_valid"]
        lines.append("| %s | %s | %s | %s | [%.6f, %.6f] |" % (kind, mean_sd(item["methods"]["attention"]["additional_unique_valid"]), mean_sd(item["methods"]["coverage"]["additional_unique_valid"]), " / ".join("%+.6f" % x for x in paired_result["by_seed"]["values"]), *paired_result["fixed_seed_parent_bootstrap_95"]))
    lines += ["", "| 情境 | attention 新候选Valid（%）均值±SD | coverage 新候选Valid（%）均值±SD |", "|---|---|---|" ]
    for kind in KINDS:
        item = contexts[kind]["methods"]
        lines.append("| %s | %s | %s |" % (kind, mean_sd(item["attention"]["new_valid_rate"], 100), mean_sd(item["coverage"]["new_valid_rate"], 100)))
    lines += ["", "单独情境均未达到三个训练种子同方向提升。two_valid 更接近核心条件补全问题，其方向也是 + / + / −，不能用 single/mixed 的总体平均替代该机制证据。", "",
        "## two_valid 失败分解", "", "每种子 128 场景、256 个新候选。重复数包含剩余模式不足时允许的有效同类输出；已经覆盖全部已知类型的场景无需生成新类型。", "",
        "| seed / 方法 | 无效新候选 | 新有效独特类型总数 | 与已有类型重叠 | 新增类型 | 新增预算缺口 |", "|---|---:|---:|---:|---:|---:|" ]
    for seed in (0, 1, 2):
        for method in METHODS:
            c = result["runs"][seed][method]["contexts"]["two_valid"]["counts"]
            lines.append("| %d / %s | %.0f | %.0f | %.0f | %.0f | %.0f |" % (seed, method, c["invalid_new_candidates"], c["new_unique_types"], c["already_covered_new_types"], c["additional_types"], c["additional_deficit"]))
    lines += ["", "| 参考类型R | 父场景数 | 新增差 seed0 / seed1 / seed2 | 三种子差均值±SD |", "|---:|---:|---|---|" ]
    for r, item in result["two_valid_reference_strata"].items():
        lines.append("| %s | %d | %s | %s |" % (r, item["parent_count"], " / ".join("%+.4f" % x for x in item["paired_delta"]["values"]), mean_sd(item["paired_delta"])))
    lines += ["", "所有情境、所有种子的无效候选/碰撞、覆盖重叠、预算缺口和失败 scene_id 清单在 JSON 的 `runs.*.*.contexts.*.counts`；完整原始逐场景字段在相邻 `per_scene_table`。不删除失败种子，不只保存方法获胜场景。", "",
        "## 研究决定", "", "当前结果应降级为**不稳定的开发趋势**。保留 coverage 实现与所有失败，但不扩成主要方法优势声明；普通 attention 仍是必要强基线。seed 0 的固定权重干预提示池化与所学表示共同影响已覆盖模式重叠，可作为失败定位材料，不能抵消 seed 2 反转。", "",
        "下一步优先核对相同更长训练预算下的收敛曲线，以及已有自产草稿闭环实验；不要只继续换种子。若平衡训练预算后优势仍不稳定，应恢复最强普通集合方案，将当前 max 聚合仅保留为消融，转向真实观测任务中能直接暴露的终点/覆盖瓶颈。没有证据支持此时叠加更多模块。", "",
        "两个模型对单草稿精确重复和硬屏蔽无效草稿本来都幂等；近重复稳定性、learned gate、模型草稿和机器人执行还须各自验证。当前参考类型由两墙模板定义，三种子仍不是独立任务/设置。", "",
        "## 唯一后备机制建议：约束变化下的局部路线编辑", "",
        "**研究问题换到确实需要更新的场景。** 当前静态空集合 joint4 已接近/达到类型预算上限，自产 2+2 又不如 joint4，继续改变集合池化缺乏空间。先完成已安排的 self-draft 配对修复；若仍无收益，恢复 joint4。之后唯一建议是：任务目标不变、局部通道状态改变时，预测旧路线中必须重写的局部范围，保留其余仍正确段，并在有限新增候选预算内生成替代路线。研究的是任务层路线的条件更新，不是低层控制。", "",
        "**最小机制。** 输入同一旧候选集、旧/新观测条件；一个共享路线编码器预测受约束变化影响的路段，输出受影响候选的局部改写，未受影响段显式复制。训练用新条件下的已知正参考和同父场景配对约束，只对经检查仍可行的部分施加保留一致性；不强迫所有旧路线保留，也不监督真实总解数。碰撞/约束影响标签仅用于训练，同样提供给普通 attention 对照的辅助训练；测试时不输入真值编辑 mask。先不加入新的采样器、max memory、LoRA 或多个评分模块。", "",
        "**最近工作与差别边界。** [MTR](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html) 已有地图读取和轨迹细化，[同伦约束搜索](https://ojs.aaai.org/index.php/AAAI/article/view/7735) 已显式维护路径类别，[条件DPP](https://arxiv.org/abs/1207.6083) 已处理给定集合后的多样化；增量规划中的保留/重规划也不是新概念。因此仅复制路径、预测mask或加一致性损失都不足以构成创新。待证实的区别只能是：从观测约束变化定位编辑范围后，是否在与强全路径补全相同的生成预算内保住有效路线，同时可靠替换受影响路线。新颖性仍需在该具体机制成立后继续核验，不预先命名论文方法。", "",
        "**用现有数据低成本区分的新 setting。** 版本化构造 `multigate_change_v1`，从现有 R=6–16 的父布局派生一项可观测变化：关闭一面墙的一个开口，且保留至少一个有效通路；同目标/起点、其它几何不变。原参考中通过未关闭开口的路线就是新条件已知正例，无须伪造更多解。所有变化版本继承原父 split，不能重分割；新模式用物理开口区间关联，不能因开口编号重排误认保留/替换。旧集合来自冻结的普通 joint4 在旧条件的实际预测，分别报告 0/1/2/更多旧候选失效的场景。此 setting 不修改原静态评测规则，旧主结果保持原样。", "",
        "**对齐预算与强对照。** 所有方法共享同一旧集合和变化信息，比较新增 b=1/2 条路线的更新；每次编辑后的完整路线都算一个新候选，失败/被筛除路线照计。两请求累计产生 4+b 条，不能标成总预算 K=4；可以另报最终保留四条的质量。普通 attention 全路径补全、普通回归 K=b 加同一保留/筛选规则，以及从头 joint4 重算都是必要对照，后者按累计 8 条和实际时间单列。各方法同训练父场景、同参考曝光、同影响标签、同 checker/筛选规则；计入两次观测编码、定位、改写和校验。主指标是变化后有效覆盖与成本，不以复制后必然较高的保留率单独报胜。", "",
        "**证伪与退出。** 若普通 attention 在同 b/时间下已达到相同变化后覆盖，或收益完全由共同保留规则解释，就没有保留新模块的理由。若只在 oracle 编辑mask有效、预测mask失效，也未成立。低成本开发筛选先单种子，只有在不同失效数分层仍有质量—成本优势才做三种子与独立设置。当前 RLBench reach32 无明确通道，不能直接拿其弯曲路径验证此机制；观测验证需要真实 RGB-D 中可见的局部障碍变化和相应 RLBench-derived 数据，待几何条件链路补齐再推进。", "",
        "## 复现", "", "```powershell", "python scripts/analyze_completion_seeds.py --runs runs/v2_completion --output reports/v2_completion/three_seed_analysis.json --report reports/COMPLETION_THREE_SEED.md", "```", "", "Bootstrap：10,000 次、seed 20261002；代码会校验完成状态、配置、checkpoint/prediction 哈希、逐场景聚合与实际草稿一致性。所有置信区间都是开发集描述性结果。" ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/v2_completion")
    parser.add_argument("--output", default="reports/v2_completion/three_seed_analysis.json")
    parser.add_argument("--report", default="reports/COMPLETION_THREE_SEED.md")
    parser.add_argument("--bootstrap-seed", type=int, default=20261002)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    result = analyze(args)
    report(result, args.report)
    print(json.dumps(dict(primary={k: v for k, v in result["primary"].items() if k != "parent_deltas"},
        two_valid_reference_strata=result["two_valid_reference_strata"]), indent=2))


if __name__ == "__main__":
    main()
