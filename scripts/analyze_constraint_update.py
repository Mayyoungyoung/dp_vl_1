"""One-seed developmental constraint-update analysis, with parent pairing.

This script never obtains a reference-derived development edit mask. It uses
stored prediction masks and geometry only to diagnose fragmentation/collision.
Parent bootstrap intervals condition on these trained, DEV-selected models;
they do not quantify uncertainty across independently trained seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from routeset.geometry import segment_aabb_intersection
from routeset.multigate import CLEARANCE, path_validity, route_modes, wall_boxes


ARMS = ("full_free", "full_paired", "local_paired")
METRICS = ("union_unique_valid", "additional_unique_valid", "union_reference_coverage",
           "new_valid_rate", "new_collision_rate", "predicted_edit_fraction")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parent_values(rows, metric):
    parents = sorted(set(row["parent_id"] for row in rows))
    return parents, np.asarray([np.mean([r[metric] for r in rows if r["parent_id"] == parent]) for parent in parents])


def summarize(rows, keys=METRICS):
    if not rows:
        return None
    return dict(parent_count=len(set(r["parent_id"] for r in rows)), change_count=len(rows),
                **{key: float(parent_values(rows, key)[1].mean()) for key in keys})


def compare(left, right, bootstrap_count, seed):
    # Returned differences are right - left; same parent and physical closure.
    by_left = {r["scene_id"]: r for r in left}
    by_right = {r["scene_id"]: r for r in right}
    if set(by_left) != set(by_right) or len(by_left) != len(left):
        raise ValueError("duplicate or unmatched physical intervention rows")
    differences = []
    for scene_id in sorted(by_left):
        a, b = by_left[scene_id], by_right[scene_id]
        for key in ("parent_id", "old_invalid_count", "newly_invalid_old_count", "reference_types", "total_candidates", "closed_wall", "closed_opening"):
            if a[key] != b[key]:
                raise ValueError("paired data/budget mismatch: " + key)
        row = {key: a[key] for key in ("parent_id", "scene_id", "old_invalid_count", "newly_invalid_old_count", "reference_types")}
        row.update({key: b[key] - a[key] for key in METRICS})
        differences.append(row)
    results = {}
    for metric in METRICS:
        parents, values = parent_values(differences, metric)
        generator = np.random.default_rng(seed)
        draws = generator.integers(len(parents), size=(bootstrap_count, len(parents)))
        means = values[draws].mean(-1)
        results[metric] = dict(difference=float(values.mean()),
            parent_bootstrap95=np.quantile(means, [.025, .975]).tolist(),
            positive_parents=int((values > 1e-12).sum()), negative_parents=int((values < -1e-12).sum()),
            tied_parents=int((np.abs(values) <= 1e-12).sum()), parent_count=len(parents))
    results["strata"] = {key: {str(value): summarize([r for r in differences if r[key] == value])
                               for value in sorted(set(r[key] for r in differences))}
                          for key in ("old_invalid_count", "newly_invalid_old_count", "reference_types")}
    results["per_change_differences"] = differences
    return results


def fragmentation(paths, gate, sources, scenes, rows):
    checks = [path_validity(p, s) for p, s in zip(paths, scenes)]
    valid = np.stack([c["valid"] for c in checks])
    collision = np.stack([c["collision"] for c in checks])
    runs = gate[..., 0].astype(int) + ((~gate[..., :-1]) & gate[..., 1:]).sum(-1)
    transitions = (gate[..., 1:] != gate[..., :-1]).sum(-1)
    copied_equal = np.all(paths[:, :, 1:-1] == sources[:, :, 1:-1], axis=-1)
    # Locate collisions relative to *predicted* editing only. This diagnostic
    # uses exact geometry, not a target-derived edit mask. Endpoint constraints
    # are shared; treating them as copied is bookkeeping, not edit supervision.
    full_gate = np.pad(gate, ((0, 0), (0, 0), (1, 1)), constant_values=False)
    segment_kind = full_gate[..., :-1].astype(int) + full_gate[..., 1:].astype(int)
    segment_collision = []
    for path, scene in zip(paths, scenes):
        lower, upper = wall_boxes(scene)
        a, b, lo, hi = np.broadcast_arrays(path[:, :-1, None, :], path[:, 1:, None, :],
                                          lower - CLEARANCE, upper + CLEARANCE)
        segment_collision.append(segment_aabb_intersection(a, b, lo, hi).any(-1))
    segment_collision = np.stack(segment_collision)
    mask_stats = []
    per_candidate = []
    for idx, row in enumerate(rows):
        mask_stats.append(dict(parent_id=row["parent_id"], scene_id=row["scene_id"],
            mean_edit_runs=float(runs[idx].mean()), fragmented_route_fraction=float((runs[idx] > 1).mean()),
            mean_gate_transitions=float(transitions[idx].mean()), mean_edit_fraction=float(gate[idx].mean()),
            no_edit_route_fraction=float((~gate[idx].any(-1)).mean()),
            all_edit_route_fraction=float(gate[idx].all(-1).mean())))
        for slot in range(paths.shape[1]):
            per_candidate.append(dict(parent_id=row["parent_id"], scene_id=row["scene_id"], slot=slot,
                edit_runs=int(runs[idx, slot]), edit_fraction=float(gate[idx, slot].mean()),
                gate_transitions=int(transitions[idx, slot]), valid=bool(valid[idx, slot]),
                collision=bool(collision[idx, slot])))
    strata = {}
    for name, selected in (("no_edit", runs == 0), ("one_interval", runs == 1), ("fragmented", runs > 1)):
        strata[name] = dict(candidate_count=int(selected.sum()),
            collision_rate=float(collision[selected].mean()) if selected.any() else None,
            valid_rate=float(valid[selected].mean()) if selected.any() else None)
    keys = tuple(key for key in mask_stats[0] if key not in ("parent_id", "scene_id"))
    return dict(parent_weighted=summarize(mask_stats, keys), per_candidate=per_candidate,
        gate_shape_collision_diagnostic=strata,
        collision_segment_relation={name: dict(colliding_segments=int((segment_collision & (segment_kind == kind)).sum()),
            routes_with_this_collision=int((segment_collision & (segment_kind == kind)).any(-1).sum()))
            for kind, name in ((0, "both_copied"), (1, "edit_copy_boundary"), (2, "both_edited"))},
        predicted_copy_points=int((~gate).sum()),
        predicted_copy_points_exactly_equal=int((~gate & copied_equal).sum()),
        interpretation="association only; full arms train this auxiliary gate but do not use it in their route output")


def counterfactual_gate(paths, gates, sources, old, scenes, rows):
    """Same saved full model proposals, now applying its own predicted gate."""
    copied = paths.copy()
    copied[:, :, 1:-1] = np.where(gates[..., None], paths[:, :, 1:-1], sources[:, :, 1:-1])
    changes = []
    for row, fresh, draft, scene in zip(rows, copied, old, scenes):
        valid = path_validity(fresh, scene)["valid"]
        union = np.concatenate([draft, fresh])
        union_valid = path_validity(union, scene)["valid"]
        modes = route_modes(union, scene)
        unique = len(set(modes[union_valid & (modes >= 0)].tolist()))
        changes.append(dict(parent_id=row["parent_id"], scene_id=row["scene_id"],
            unique_delta=unique - row["union_unique_valid"], valid_delta=float(valid.mean()) - row["new_valid_rate"]))
    return dict(scope="post-hoc DEV output intervention using own predicted mask; no retraining/no target-derived mask; not a final method result",
                parent_weighted_delta=summarize(changes, ("unique_delta", "valid_delta")))


def analyze(runs, data_path, bootstrap_count=10000, seed=20261002):
    dataset_hash = sha(data_path)
    # Only geometric observations / row IDs, no reference path or proxy labels.
    with np.load(data_path, allow_pickle=False) as archive:
        scenes = archive["scenes"]
        scene_ids = archive["scene_ids"]
    scene_index = {str(scene_id): idx for idx, scene_id in enumerate(scene_ids)}
    result = dict(scope="single real training seed 0, DEV_MODEL model selection and exploratory comparisons",
        dataset_sha256=dataset_hash, bootstrap_count=bootstrap_count, bootstrap_seed=seed,
        interval_scope="parent resampling conditional on the same trained/selected models; not training-seed uncertainty",
        aggregation="mean over closures within each independent parent, then equal mean over parents", arms={}, comparisons={})
    reference_config = None
    stored = {}
    for arm in ARMS:
        path = Path(runs) / arm
        status, config, summary, history = [read(path / name) for name in ("status.json", "config.json", "summary.json", "history.json")]
        if status["status"] != "completed" or status["exit_code"] != 0:
            raise ValueError("arm has not completed: " + arm)
        if config["dataset_sha256"] != dataset_hash or config["arm"] != arm or config["seed"] != 0:
            raise ValueError("expected matched seed0/data/arm")
        if reference_config is not None:
            for key in ("steps", "batch_size", "width", "depth", "lr", "seed", "proxy_threshold", "aux_weight", "dataset_sha256", "code_commit", "threads"):
                if reference_config[key] != config[key]:
                    raise ValueError("training mismatch: " + key)
        reference_config = config
        if sha(path / "best.pt") != summary["best_checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        expected_slots = config["batch_size"] * (config["steps"] // 2 * 3 + config["steps"] % 2)
        if summary["trajectory_exposures"] != expected_slots:
            raise ValueError("incorrect b1/b2 exposure accounting")
        arm_result = dict(config=config, summary=summary, convergence=[dict(step=h["step"], loss=h["loss"],
            b1_unique=h["dev_model"]["b1"]["union_unique_valid"], b2_unique=h["dev_model"]["b2"]["union_unique_valid"],
            b1_valid=h["dev_model"]["b1"]["new_valid_rate"], b2_valid=h["dev_model"]["b2"]["new_valid_rate"])
            for h in history], budgets={})
        for k in (1, 2):
            name = "b%d" % k
            destination = path / "dev_model" / name
            rows = read(destination / "per_scene.json")
            prediction_path = destination / "predictions.npz"
            if sha(prediction_path) != summary["prediction_sha256"][name]:
                raise ValueError("prediction hash mismatch")
            with np.load(prediction_path, allow_pickle=False) as archive:
                paths, gate = archive["paths"], archive["predicted_edit_mask"].astype(bool)
                old, order = archive["old_paths"], archive["source_order"]
                ids = archive["scene_ids"]
            if [r["scene_id"] for r in rows] != ids.tolist() or paths.shape[1] != k:
                raise ValueError("row/prediction alignment or K mismatch")
            new_scenes = scenes[[scene_index[r["scene_id"]] for r in rows]]
            source = old[np.arange(len(old))[:, None], order[:, :k]]
            aggregate = summarize(rows)
            for key in METRICS:
                if abs(aggregate[key] - summary["metrics"][name][key]) > 1e-10:
                    raise ValueError("reported parent aggregation mismatch: " + key)
            fragment = fragmentation(paths, gate, source, new_scenes, rows)
            if arm == "local_paired" and fragment["predicted_copy_points"] != fragment["predicted_copy_points_exactly_equal"]:
                raise ValueError("local model failed to copy predicted retained points exactly")
            if name in stored:
                np.testing.assert_array_equal(stored[name]["old_paths"], old)
                np.testing.assert_array_equal(stored[name]["order"], order)
            else:
                stored[name] = dict(old_paths=old, order=order)
            arm_result["budgets"][name] = dict(aggregate=aggregate, fragmentation=fragment, rows=rows,
                strata={key: {str(value): summarize([r for r in rows if r[key] == value])
                              for value in sorted(set(r[key] for r in rows))}
                        for key in ("old_invalid_count", "newly_invalid_old_count", "reference_types")})
            if arm != "local_paired":
                arm_result["budgets"][name]["counterfactual_apply_own_gate"] = counterfactual_gate(paths, gate, source, old, new_scenes, rows)
        result["arms"][arm] = arm_result
    for arm in ARMS[1:]:
        if result["arms"][arm]["summary"]["positive_pool_access"] != result["arms"][ARMS[0]]["summary"]["positive_pool_access"]:
            raise ValueError("scene/reference exposure differs between arms")
    for a, b in (("full_free", "local_paired"), ("full_paired", "local_paired"), ("full_free", "full_paired")):
        result["comparisons"][b + "_minus_" + a] = {"b%d" % k: compare(result["arms"][a]["budgets"]["b%d" % k]["rows"],
            result["arms"][b]["budgets"]["b%d" % k]["rows"], bootstrap_count, seed) for k in (1, 2)}
    return result


def markdown(result):
    steps = result["arms"]["full_free"]["config"]["steps"]
    exposures = result["arms"]["full_free"]["summary"]["trajectory_exposures"]
    parents = result["arms"]["full_free"]["budgets"]["b1"]["aggregate"]["parent_count"]
    negative = all(result["comparisons"][comparison][budget]["union_unique_valid"]["difference"] <= 0
        for comparison in ("local_paired_minus_full_free", "local_paired_minus_full_paired") for budget in ("b1", "b2"))
    behind_free = all(result["comparisons"]["local_paired_minus_full_free"][budget]["union_unique_valid"]["difference"] <= 0
                      for budget in ("b1", "b2"))
    conclusion = ("%d步筛选未支持局部门控：local_paired在b1/b2均未胜强full_free或同配对full_paired。保留全部负结果。" % steps
                  if negative else ("%d步筛选未支持局部门控优于强基线：local_paired在b1/b2均落后full_free。对固定配对控制的小幅改善不能建立方法优势。" % steps
                  if behind_free else "%d步开发筛选比较见下表；单训练种子尚不能建立方法优势。" % steps))
    lines = ["# 约束变化更新：seed0 开发筛选", "", "**" + conclusion + "**",
        "", "同一真实训练seed0；%d个DEV_MODEL父用于选优与分析。父场景bootstrap不代表训练种子稳定性。" % parents,
        "", "| 臂 | best步 | 新增b | 累计候选 | union UniqueValid | 新增类型 | 新路线有效率 | 编辑内点比例 | 编辑段数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        run = result["arms"][arm]
        for k in (1, 2):
            budget = run["budgets"]["b%d" % k]
            row, frag = budget["aggregate"], budget["fragmentation"]["parent_weighted"]
            lines.append("| %s | %d | %d | %d | %.5f | %.5f | %.2f%% | %.2f%% | %.3f |" % (
                arm, run["summary"]["best_step"], k, 4+k, row["union_unique_valid"], row["additional_unique_valid"],
                100*row["new_valid_rate"], 100*row["predicted_edit_fraction"], frag["mean_edit_runs"]))
    lines += ["", "| 配对差（前者减后者） | b | union UniqueValid差 | 父bootstrap95% | 新路线有效率差 |",
              "|---|---:|---:|---|---:|"]
    for name, budgets in result["comparisons"].items():
        for k, comparisons in budgets.items():
            r = comparisons["union_unique_valid"]
            lines.append("| %s | %s | %+.5f | [%.5f, %.5f] | %+.2fpp |" % (name, k, r["difference"],
                r["parent_bootstrap95"][0], r["parent_bootstrap95"][1], 100*comparisons["new_valid_rate"]["difference"]))
    lines += ["", "每臂累计%d完整路线目标槽、同一父场景/参考曝光/参数量；旧4模型训练成本是共同前置成本，未重复算作三次新训练。" % exposures,
        "", "预测mask碎片仅从模型输出计算；完整线段几何复验用于观察碎片与失败关联。没有计算DEV参考衍生编辑mask，没有将代理标签当影响真值。",
        "", "失效数0/1/2分层、剩余参考类型数分层、逐候选碎片/碰撞、父场景配对差和收敛曲线详见同名JSON。若3/4失效层在DEV没有样本，就不能据此声称该层有效。",
        "", "本次未完成多种子、独立设置、完整两请求同设备固定时间预算或真实观测约束变化；不可将单seed开发优势写成最终方法成立。"]
    if steps != 2000:
        lines += ["", "本表为续训后的新证据。追加成本见summary的incremental_*，累计成本见cumulative_*；不能将累计成本与原实验相加。原2000步的失败定位保留在v1报告，不套用到本轮。"]
        lines += ["", "## 固定续训预算与实际趋势", "",
                  "| 臂 | 原投入秒 | 本段新增秒 | 累计秒 | 新增目标槽 | 累计目标槽 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            summary = result["arms"][arm]["summary"]
            origin = summary["cost_origin"]
            lines.append("| %s | %.3f | %.3f | %.3f | %d | %d |" % (arm, origin["prior_elapsed_s"],
                summary["incremental_elapsed_s"], summary["cumulative_elapsed_s"],
                summary["incremental_trajectory_exposures"], summary["trajectory_exposures"]))
        lines += ["", "上述为训练及开发评估实测墙钟时间，不是单请求端到端推理延迟或固定时间预算推理对照。门控仍先计算完整提案，没有稀疏推理加速证据。",
                  "", "| 臂 | 步 | b1 Unique | b2 Unique | 路线损失 |",
                  "|---|---:|---:|---:|---:|"]
        for arm in ARMS:
            for point in result["arms"][arm]["convergence"][-5:]:
                lines.append("| %s | %d | %.5f | %.5f | %.7f |" % (arm, point["step"], point["b1_unique"],
                    point["b2_unique"], point["loss"][1]))
        lines += ["", "local的训练损失及末段开发指标仍在改善，不能据此宣称绝对收敛；但一次预先确定的等预算续训后仍落后自由匹配，末尾最优本身不构成无限续训的理由。",
                  "", "## 失败定位与研究决定", ""]
        strata = result["comparisons"]["local_paired_minus_full_free"]["b2"]["strata"]
        for key, label in (("old_invalid_count", "旧路线失效数"), ("reference_types", "剩余参考类型数")):
            values = "; ".join("%s: %+.5f" % (value, group["union_unique_valid"]) for value, group in strata[key].items())
            lines.append("b2 local减full_free，按%s分层为%s。" % (label, values))
            lines.append("")
        for budget in ("b1", "b2"):
            frag = result["arms"]["local_paired"]["budgets"][budget]["fragmentation"]
            relation = frag["collision_segment_relation"]
            lines.append("%s局部门控碎片路线比例%.2f%%；碰撞线段的两端均编辑/编辑复制边界/均复制计数为%d/%d/%d。保留点%d/%d精确复制。" % (
                budget, 100*frag["parent_weighted"]["fragmented_route_fraction"],
                relation["both_edited"]["colliding_segments"], relation["edit_copy_boundary"]["colliding_segments"],
                relation["both_copied"]["colliding_segments"], frag["predicted_copy_points_exactly_equal"], frag["predicted_copy_points"]))
            lines.append("")
        if behind_free:
            lines += ["研究决定：放弃当前局部门控作为核心机制，保留full_free普通补全为强基线；停止该分支追加训练，不增加碎片或连通mask模块。负结果、全部权重和预测保留。此决定针对当前实现与受控设置，不声称所有局部编辑机制均无效。", ""]
        return "\n".join(lines) + "\n"
    lines += ["", "## 失败定位与下一项可证伪假设", "",
        "b2时local相对full_free，在旧失效0/1/2的分层均回退（−0.18301/−0.05357/−0.21199），新参考类型数3/4/6/8/9/12的差依次0/−0.08621/−0.10417/−0.18750/−0.28125/−0.47500。不能用某个有利子组掩盖总体失败。",
        "", "mask碎片不是当前主要失败证据：local的b1/b2碎片路线比例为13.65%/11.10%；碎片组碰撞率4.0%/9.52%，单连续编辑段组反而13.86%/14.88%。b1全部224个碰撞线段均为两个端点都预测编辑；b2的527个碰撞线段中522个两端都编辑，只有5个在编辑/复制边界（涉及2条路线）。没有碰撞段的两端都被复制。统计是诊断关联，不能直接当因果结论。",
        "", "同权重事后输出干预也不支持增加碎片模块：给full_paired应用其自己的预测mask，union Unique仅变化+0.00251/+0.00219；给full_free应用同类代理mask反而变化−0.12153/−0.21188。后者的自由匹配可选择不同正例，固定配对的辅助代理与其实际输出不一定相容；不能把该干预当已训练local_free结果。",
        "", "三臂最优均位于2000步。local在1500→2000时b1/b2 Unique继续提升0.09555/0.15639，路线回归损失0.002307→0.001807；配对目标的拟合仍明显弱于full_free的0.000431。唯一下一项修复假设是：固定配对和门控需要更多优化，当前差异部分来自未收敛。按同一三臂各续训至4000步、同新增曝光，并保留v1全部输出，检验局部门控是否仍有质量—成本理由。不得只延长local，不改变数据/阈值/候选数。",
        "", "若公平续训后仍不能胜full_free，则丢弃门控并保留最强普通补全；不再据本次碎片统计追加连通mask模块。续训功能与启动由根线程审阅，本文尚无续训结果。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/constraint_update_v1/seed0")
    parser.add_argument("--data", default="data/multigate_change_v1.npz")
    parser.add_argument("--output", default="reports/constraint_update_v1/seed0_analysis.json")
    parser.add_argument("--report", default="reports/CONSTRAINT_UPDATE_RESULTS.md")
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    args = parser.parse_args()
    result = analyze(args.runs, args.data, args.bootstrap_count)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.report).write_text(markdown(result), encoding="utf-8")
    print(json.dumps({arm: {k: result["arms"][arm]["budgets"][k]["aggregate"] for k in ("b1", "b2")} for arm in ARMS}))


if __name__ == "__main__":
    main()
