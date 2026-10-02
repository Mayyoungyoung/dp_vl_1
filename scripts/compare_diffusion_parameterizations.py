"""Compare separately evaluated epsilon/v repeats and index preserved artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.multigate import load_dataset, route_metrics
from scripts.analyze_multigate_diffusion import load_predictions, paired_difference, parent_metrics, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/multigate_v1_partitions/development.npz")
    parser.add_argument("--epsilon", default="runs/multigate_diffusion_v1/seed0")
    parser.add_argument("--velocity", default="runs/multigate_diffusion_v2/seed0")
    parser.add_argument("--output", default="reports/multigate_diffusion_v2/parameterization_comparison.json")
    parser.add_argument("--report", default="reports/DIFFUSION_PARAMETERIZATION_RESULTS.md")
    args = parser.parse_args()
    data = load_dataset(args.data)
    if set(data["splits"]) != {"TRAIN", "DEV_MODEL"}:
        raise ValueError("development-only archive required")
    ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
    values, details, summaries, configs = {}, {}, {}, {}
    index = dict(server_root="/home/wzy/dpvlm/route_set_v1", dataset_sha256=sha256(args.data), methods={})
    for label, base in (("epsilon", Path(args.epsilon)), ("v", Path(args.velocity))):
        for arm in ("independent", "set_diffusion"):
            name, run = label + "_" + arm, base / arm
            summary, config = read(run / "summary.json"), read(run / "config.json")
            if read(run / "status.json")["status"] != "completed" or config["dataset_sha256"] != sha256(args.data):
                raise ValueError("incomplete run or mismatched data")
            if config.get("parameterization", "epsilon") != label:
                raise ValueError("checkpoint parameterization mismatch")
            summaries[name], configs[name] = summary, config
            values[name], details[name] = {}, {}
            artifacts = {}
            for key, metrics in summary["metrics"].items():
                repeats, decomposition = [], []
                k = int(key[1:])
                for repeat in range(config["final_repeats"]):
                    relative = Path("dev_model") / key / ("repeat%d" % repeat) / "predictions.npz"
                    path = run / relative
                    expected = summary["prediction_sha256"][relative.parent.as_posix()]
                    if sha256(path) != expected:
                        raise ValueError("prediction hash mismatch")
                    paths = load_predictions(path, data, ids, k)
                    parents, parent = parent_metrics(paths, data, ids)
                    repeats.append(parent)
                    checked = route_metrics(paths, data["scenes"][ids])["per_scene"]
                    valid, modes, unique = checked["valid"], checked["modes"], checked["unique_count"]
                    decomposition.append(dict(
                        returned=k, valid_count=float(valid.sum(1).mean()), invalid_count=float((~valid).sum(1).mean()),
                        collision_count=float(checked["collision"].sum(1).mean()),
                        invalid_without_collision=float((~valid & ~checked["collision"]).sum(1).mean()),
                        unclassified_valid=float((valid & (modes < 0)).sum(1).mean()),
                        classified_duplicate_count=float(((valid & (modes >= 0)).sum(1) - unique).mean()),
                        unique_valid=float(unique.mean())))
                    artifacts[relative.as_posix()] = dict(sha256=expected, local_path=str(path),
                        server_path="/home/wzy/dpvlm/route_set_v1/" + path.as_posix(), bytes=path.stat().st_size)
                values[name][key] = {metric: np.stack([r[metric] for r in repeats]) for metric in repeats[0]}
                details[name][key] = {metric: float(np.mean([d[metric] for d in decomposition])) for metric in decomposition[0]}
            for filename in ("best.pt", "last.pt"):
                path = run / filename
                digest = sha256(path)
                if digest != summary[filename.split('.')[0] + "_checkpoint_sha256"]:
                    raise ValueError("checkpoint hash mismatch")
                artifacts[filename] = dict(sha256=digest, local_path=str(path),
                    server_path="/home/wzy/dpvlm/route_set_v1/" + path.as_posix(), bytes=path.stat().st_size)
            index["methods"][name] = dict(artifacts=artifacts, source_commit=config["code_commit"],
                source_hashes=config["source_hashes"], summary_sha256=sha256(run / "summary.json"),
                config_sha256=sha256(run / "config.json"), stream_sha256=summary["training_stream_sha256"],
                target_slots=summary["trajectory_exposures"], elapsed_s=summary["elapsed_s"], gpu_hours=summary["gpu_hours_reserved"])
    for key in ("training_stream_sha256", "trajectory_exposures", "reference_pool_access"):
        if len({s[key] for s in summaries.values()}) != 1:
            raise ValueError("four-arm actual stream/exposure mismatch: " + key)
    for key in ("seed", "steps", "batch_size", "candidates", "width", "depth", "lr", "diffusion_steps", "sampling_steps", "final_repeats"):
        if len({c[key] for c in configs.values()}) != 1:
            raise ValueError("four-arm control mismatch: " + key)
    comparisons = {}
    for arm in ("independent", "set_diffusion"):
        comparisons[arm] = {key: {metric: paired_difference(array.mean(0), values["epsilon_" + arm][key][metric].mean(0))
            for metric, array in current.items()} for key, current in values["v_" + arm].items()}
    result = dict(protocol="one training seed per arm/parameterization; three independent sampling repeats evaluated separately",
        four_arm_stream_check="passed", actual_stream_sha256=next(iter(summaries.values()))["training_stream_sha256"],
        comparisons_v_minus_epsilon=comparisons, candidate_decomposition=details,
        known_opening_pair_capacity_k4=float(np.minimum(data["path_mask"][ids].sum(1), 4).mean()),
        capacity_scope="current controlled opening-pair type definition and enumerated positive representatives; not count of all continuous trajectories",
        summaries=summaries)
    write_json(args.output, result)
    index["reproduce_analysis"] = "python -m scripts.compare_diffusion_parameterizations"
    write_json(Path(args.output).parent / "artifact_index.json", index)
    lines = ["# 标准v参数化配对结果", "", "四臂实际父/目标/时间/噪声stream严格相同，每臂3000步、768000目标槽，采样40步；三次随机重复分别评价，未合并候选。", "",
        "| 训练参数化/模型 | K4 Valid | K4 UniqueValid | 碰撞数/场景 | 有效重复数/场景 | 完美去重上限 |", "|---|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        m, d = summary["metrics"]["k4"]["mean"], details[name]["k4"]
        lines.append("| %s | %.6f | %.6f | %.6f | %.6f | %.6f |" %
            (name, m["valid_rate"], m["unique_valid"], d["collision_count"], d["classified_duplicate_count"], d["valid_count"]))
    lines += ["", "“完美去重上限”是假设现有有效候选全部变为不同类型、但不修复无效路线的有效候选数上界，并非可实现结果。未分类有效数另保留在JSON，未计为普通重复。", "",
        "| v减epsilon（K4） | UniqueValid差 | 父bootstrap95% | Valid差 |", "|---|---:|---|---:|"]
    for arm, by_k in comparisons.items():
        comparison = by_k["k4"]
        u = comparison["unique_valid"]
        lines.append("| %s | %.6f | [%.6f, %.6f] | %.6f |" %
            (arm, u["mean"], *u["parent_bootstrap95"], comparison["valid_rate"]["mean"]))
    lines += ["", "区间为同一训练种子下，先平均独立采样重复再配对重采样DEV父场景；不代表训练种子不确定性。v/epsilon采用各自在相同开发协议选出的best，不是强行对齐best步数。", "",
        "普通集合回归K4为100%有效、UniqueValid=3.3984375，恰等于当前DEV的mean(min(已枚举开口对类型数,4))。这是此受控类型定义下的容量，不是连续解空间总数；不能期待新机制在此指标上超过已经饱和的普通回归。", "",
        "v修复使集合扩散相对epsilon改善，独立扩散未受益；它没有解决有效性。当前约四分之三候选碰撞，排斥有效重复的最大潜在空间远小于无效路线损失。因此不启动PG强度筛选，不把该常规修复写成方法贡献，也不将当前两分钟级训练结果称作充分收敛的扩散上限。", "",
        "下一步建议限定为一次不增加模块的收敛诊断：从两臂last完整恢复到新的输出目录、提高训练上限并保留原结果，披露额外曝光/成本；已有3000步等目标槽对照保持独立。是否实施及具体预算由根线程结合GPU队列确定，不能把额外训练收益偷写成同3000步比较。", "",
        "K1/2/8完整结果与单次预测见 [DIFFUSION_MULTIGATE_V2_RESULTS.md](DIFFUSION_MULTIGATE_V2_RESULTS.md) 及 `multigate_diffusion_v2/parameterization_comparison.json`。当前均为DEV筛选，不能宣称独立测试已证明结论。", ""]
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(four_arm_stream_check="passed", comparison_k4={a: c["k4"]["unique_valid"] for a, c in comparisons.items()},
        decomposition_k4={name: d["k4"] for name, d in details.items()})))


if __name__ == "__main__":
    main()
