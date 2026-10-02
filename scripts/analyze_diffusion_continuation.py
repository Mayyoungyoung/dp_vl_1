"""Read-only analysis of the bounded 3000-to-12000 diffusion continuation."""
import argparse
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.multigate import load_dataset, route_metrics
from scripts.analyze_multigate_diffusion import analyze, load_predictions, paired_difference, parent_metrics, read


def decomposition(paths, data, ids):
    checked = route_metrics(paths, data["scenes"][ids])["per_scene"]
    valid, modes, unique = checked["valid"], checked["modes"], checked["unique_count"]
    return dict(valid_count=float(valid.sum(1).mean()), invalid_count=float((~valid).sum(1).mean()),
        collision_count=float(checked["collision"].sum(1).mean()),
        invalid_without_collision=float((~valid & ~checked["collision"]).sum(1).mean()),
        unclassified_valid=float((valid & (modes < 0)).sum(1).mean()),
        classified_duplicate_count=float(((valid & (modes >= 0)).sum(1) - unique).mean()),
        unique_valid=float(unique.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/multigate_v1_partitions/development.npz")
    parser.add_argument("--prior", default="runs/multigate_diffusion_v2/seed0")
    parser.add_argument("--runs", default="runs/multigate_diffusion_v3/seed0")
    parser.add_argument("--metadata", default="reports/multigate_diffusion_v3/training/seed0")
    parser.add_argument("--regression", default="runs/v2_round2/saturation_k4_seed0")
    parser.add_argument("--regression-metadata", default="reports/v2_round2/saturation_k4_seed0")
    parser.add_argument("--output", default="reports/multigate_diffusion_v3/analysis.json")
    parser.add_argument("--report", default="reports/DIFFUSION_MULTIGATE_V3_RESULTS.md")
    parser.add_argument("--figures", default="reports/multigate_diffusion_v3/figures")
    parser.add_argument("--source-audit", default="reports/multigate_diffusion_v3_validation/source_hash_audit.json")
    args = parser.parse_args()
    args.allow_unequal_training_exposure = True
    result = analyze(args)
    data = load_dataset(args.data)
    ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
    values, parts, originals, costs, index = {}, {}, {}, {}, {}
    source_audit = read(args.source_audit)
    for arm in ("independent", "set_diffusion"):
        for version, root in (("prior3000", Path(args.prior)), ("continued12000", Path(args.runs))):
            run = root / arm
            config, summary = read(run / "config.json"), read(run / "summary.json")
            status = read(run / "status.json")
            if status.get("status") != "completed" or status.get("exit_code") != 0:
                raise ValueError("Incomplete run: " + str(run))
            if config["dataset_sha256"] != sha256(args.data) or config["parameterization"] != "v":
                raise ValueError("Data or v-parameterization mismatch")
            if config["steps"] != (3000 if version == "prior3000" else 12000):
                raise ValueError("Bounded continuation step mismatch")
            if version == "prior3000":
                originals[arm] = dict(config=config, summary=summary,
                    hashes={name: sha256(run / name) for name in ("last.pt", "best.pt", "config.json", "summary.json", "history.json")})
                if originals[arm]["hashes"] != source_audit["originals"][arm]:
                    raise ValueError("Original five artifacts differ from the pre-launch audit")
            else:
                for key in ("arm", "dataset_sha256", "seed", "batch_size", "candidates", "width", "depth", "lr",
                            "eval_every", "diffusion_steps", "sampling_steps", "eval_candidates", "final_repeats",
                            "latency_requests", "threads", "parameterization", "optimizer"):
                    if config[key] != originals[arm]["config"][key]:
                        raise ValueError("Changed continuation method configuration: " + key)
                provenance = config["continuation"]
                for label, name in (("source_checkpoint_sha256", "last.pt"), ("source_best_checkpoint_sha256", "best.pt"),
                                    ("source_summary_sha256", "summary.json")):
                    if provenance[label] != originals[arm]["hashes"][name]:
                        raise ValueError("Continuation/source artifact mismatch")
                if (provenance["source_stream_sha256"] != originals[arm]["summary"]["training_stream_sha256"]
                        or provenance["source_source_hashes"] != originals[arm]["config"]["source_hashes"]):
                    raise ValueError("Source sampler/source-code mismatch")
                if (summary["trajectory_exposures"] != 3072000 or summary["incremental_trajectory_exposures"] != 2304000
                        or summary["cost_origin"]["prior_trajectory_exposures"] != 768000):
                    raise ValueError("Cumulative/incremental budget mismatch")
                np.testing.assert_allclose(summary["cumulative_elapsed_s"],
                    summary["cost_origin"]["prior_elapsed_s"] + summary["elapsed_s"], rtol=0, atol=1e-7)
                costs[arm] = {key: summary[key] for key in ("cost_origin", "elapsed_s", "cumulative_elapsed_s",
                    "gpu_hours_reserved", "cumulative_gpu_hours_reserved", "trajectory_exposures", "incremental_trajectory_exposures",
                    "training_stream_sha256", "incremental_stream_sha256", "selected_checkpoint_from_prior_run")}
            name = version + "_" + arm
            repeats, decomposed, artifacts = [], [], {}
            for k in (1, 2, 4, 8):
                for repeat in range(3):
                    relative = Path("dev_model") / ("k%d" % k) / ("repeat%d" % repeat) / "predictions.npz"
                    path = run / relative
                    digest = sha256(path)
                    if digest != summary["prediction_sha256"][relative.parent.as_posix()]:
                        raise ValueError("Prediction hash mismatch")
                    artifacts[relative.as_posix()] = dict(sha256=digest, local_path=str(path.resolve()),
                        server_path="/home/wzy/dpvlm/route_set_v1/" + path.as_posix(), bytes=path.stat().st_size)
                    if k == 4:
                        paths = load_predictions(path, data, ids, k)
                        _, metrics = parent_metrics(paths, data, ids)
                        repeats.append(metrics)
                        decomposed.append(decomposition(paths, data, ids))
            values[name] = {key: np.mean([r[key] for r in repeats], axis=0) for key in repeats[0]}
            parts[name] = {key: float(np.mean([r[key] for r in decomposed])) for key in decomposed[0]}
            for filename in ("best.pt", "last.pt"):
                path = run / filename
                digest = sha256(path)
                if digest != summary[filename.split(".")[0] + "_checkpoint_sha256"]:
                    raise ValueError("Checkpoint hash mismatch")
                artifacts[filename] = dict(sha256=digest, local_path=str(path.resolve()),
                    server_path="/home/wzy/dpvlm/route_set_v1/" + path.as_posix(), bytes=path.stat().st_size)
            index[name] = dict(source_commit=config["code_commit"], source_hashes=config["source_hashes"], artifacts=artifacts,
                config_sha256=sha256(run / "config.json"), summary_sha256=sha256(run / "summary.json"))
    comparisons = {arm: {key: paired_difference(array, values["prior3000_" + arm][key])
        for key, array in values["continued12000_" + arm].items()} for arm in ("independent", "set_diffusion")}
    result.update(continuation_source_checks="passed", continuation_costs=costs,
        continuation_minus_prior_k4=comparisons, candidate_decomposition=parts,
        stop_rule="Both arms reached the predeclared12000 cap; no automatic extension based on a late selected best. No PG sweep.")
    write_json(args.output, result)
    write_json(Path(args.output).parent / "artifact_index.json", dict(dataset_sha256=sha256(args.data), methods=index,
        source_original_hashes={arm: entry["hashes"] for arm, entry in originals.items()},
        reproduce="python -m scripts.analyze_diffusion_continuation"))
    lines = ["", "## 3000→12000的有限续训", "", "两臂都完整恢复各自last3000；累计目标槽3072000，为旧回归/旧扩散的4倍。原best保留可选，原3k结果不被覆盖。", "",
        "| 模型 | Unique差(12k减3k) | 条件父bootstrap95% | Valid差 | 新增秒 | 累计秒 | 累计GPU小时 |", "|---|---:|---|---:|---:|---:|---:|"]
    for arm, metrics in comparisons.items():
        u, c = metrics["unique_valid"], costs[arm]
        lines.append("| %s | %.6f | [%.6f, %.6f] | %.6f | %.3f | %.3f | %.6f |" %
            (arm, u["mean"], *u["parent_bootstrap95"], metrics["valid_rate"]["mean"], c["elapsed_s"], c["cumulative_elapsed_s"], c["cumulative_gpu_hours_reserved"]))
    lines += ["", "| 版本/模型 | 有效数/场景 | 碰撞数/场景 | 有效重复数/场景 | Unique |", "|---|---:|---:|---:|---:|"]
    for name, d in parts.items():
        lines.append("| %s | %.6f | %.6f | %.6f | %.6f |" %
            (name, d["valid_count"], d["collision_count"], d["classified_duplicate_count"], d["unique_valid"]))
    lines += ["", "累计和新增实际父/正例/时间/噪声采样digest均在两臂间一致。新增成本与原成本只各计一次；不能将累计成本再与原成本相加。", "",
        "已达到预声明12000上限；不因最优点靠近末尾自动续训，不扫PG强度。模型/数据是受控几何层，以上结果不构成观测任务层或机器人执行证据。当前饱和普通回归继续作为质量强对照；扩散有额外训练曝光，不能把差距解释为等总计算预算结论。", ""]
    with Path(args.report).open("a", encoding="utf-8") as output:
        output.write("\n".join(lines))
    print(json.dumps(dict(source_checks="passed", additional_stream=costs["independent"]["incremental_stream_sha256"],
        comparisons_k4=comparisons, decomposition=parts)))


if __name__ == "__main__":
    main()
