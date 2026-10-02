# 六任务闭合前缀快照与 TRAIN 表示审计

固定源 `5fb74b15ee0b4b89f2900c63363e24434ec90ac5` 已在 CPU1 完成真实数据导出与审计，未启动 GPU。数据是 RLBench-derived 六任务路径/事件监督基线；目前不提供生成任务成功、语义正确性或多路线类型评价。

## 来源、分母和封口

服务器数据 `/home/wzy/dpvlm/route_set_v1/data/observation_multitask_prefix24_v1`，来源是新 seed 282000 批次；每任务固定 TRAIN 索引 0、1 与 DEV_MODEL 索引 16、17。按预注册父集合等待全部闭合，没有按成功率选择父或补齐参考。旧 6c 批次仅参与机械跨批重复检测，没有混入训练参考。

- 24 个请求父（12 TRAIN、12 DEV_MODEL），72 个请求/实际提案，72 条成功正参考，0 初始化失败、0 零参考父、0 中断提案。
- 96 条语言记录为同父语言改写；同一父共享 3 条参考。它们不是 96 个独立场景，也不是同图不同目标。
- 所选父最终 worker 累计实际耗时 1383.483804 s，包含整个采集 worker，未用语言数放大数据量。
- 初态、正参考 SHA、严格恢复、camera policy/flags、来源 hash 全部通过 exporter；四个模型/预算输入文件 SHA 再次与本地归档字节一致。
- 当前机械 gate 检查 70 份初态指纹：28 份新物理 hash、42 份旧图像 hash；没有检出跨角色重复。旧物理信息缺失仍属未核验，hash 不同不保证独立。每次模型使用前仍需重新检查当前 gate。
- 只读取所选 TRAIN/DEV_MODEL 原始文件；其它角色仅比对已有机械 hash。TEST_LOCKED 图像、路径和验收结果未用于此审计。

快照 manifest SHA256：`0d62b1e52b8729a4f129b1dbc0207307f5373280604f6cdb0c6b22069be7bffa`。所有实际命令、退出码、日志和文件索引在 [归档](observation_multitask_prefix24_v1/artifact_index.json)。服务器 run 为 `/home/wzy/dpvlm/route_set_v1/runs/observation_multitask_prefix24_snapshot_v1`；原始二进制留在服务器来源路径，不进入普通 Git。

## 固定 12 TRAIN 父的表示证据

审计固定 H24、stride-2 实际 RGB-D 点云、每坐标 5 cm 残差界限。36 条原始正参考全部完成：36/36 压缩开闭事件序列一致、全部阶段边界坐标误差为 0、0 重采样表示失败。原始点到 H24 折线最大偏差最高为 2.047 cm，路径长度比最小为 0.9658。这只是表示损失审计，未据此筛除或修改标签。

| 任务 | 参考数 | hard point anchor 无法覆盖的末点 | 末点到最近实际点的 L2 距离 | 最大 raw→H24 折线距离 |
|---|---:|---:|---:|---:|
| reach_target | 6 | 0 | 2.18–2.27 cm | 0.53 cm |
| pick_and_lift | 6 | 0 | 2.47–2.61 cm | 2.05 cm |
| push_button | 6 | 0 | 0.74–0.77 cm | 0.56 cm |
| take_lid_off_saucepan | 6 | 6 | 10.08–10.13 cm | 0.66 cm |
| pick_up_cup | 6 | 6 | 23.95–24.16 cm | 1.37 cm |
| slide_block_to_target | 6 | 0 | 0.52–0.76 cm | 0.77 cm |

“无法覆盖”实际按最近 L∞ 距离大于 5 cm 判断，不按 L2 判断。cup 的 L∞ 为 20.36–23.76 cm，lid 为 10.07–10.13 cm。该结论只针对实际选中观测点加有限残差的硬锚点表示；soft weighted anchor 可能处于点间，因此本审计不是其不可表示证明。

判断：首个多任务普通头应预测不受可见表面锚点 ±5 cm 限制的最终三维坐标，并关闭把最终点当初始表面关注监督的辅助损失。保留全部原始末点与抓放阶段，不把空中末点替换为接触点。普通头的这一适配不是新机制优势；后续训练仍需用参考 ADE、末点误差与事件指标验证。

## 已执行复现命令

工作目录为上述固定 release：`/home/wzy/dpvlm/route_set_v1/research_v2/releases/5fb74b15ee0b4b89f2900c63363e24434ec90ac5`，设置 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`CUDA_VISIBLE_DEVICES=`、`PYTHONPATH=.`。每项通过 `record_job.py` 记录，原命令在相应 `*.status.json` 中。

```bash
/home/wzy/dpvlm/route_set_v1/.venv/bin/python -m pytest -q tests/test_multitask_snapshot.py tests/test_multitask_representation_audit.py
/home/wzy/dpvlm/route_set_v1/.venv/bin/python scripts/snapshot_multitask_observations.py --source /home/wzy/dpvlm/route_set_v1/data/observation_multitask_validated_five_v1 --compare-source /home/wzy/dpvlm/route_set_v1/data/observation_multitask_resume_probe_v1 --output /home/wzy/dpvlm/route_set_v1/data/observation_multitask_prefix24_v1
/home/wzy/dpvlm/route_set_v1/.venv/bin/python scripts/audit_multitask_train_representation.py --snapshot /home/wzy/dpvlm/route_set_v1/data/observation_multitask_prefix24_v1 --output /home/wzy/dpvlm/route_set_v1/runs/observation_multitask_prefix24_snapshot_v1/train_representation
```

测试为 9 passed / 0.60 s；导出与表示审计均退出 0。Exporter 拒绝覆盖已有输出，复现需使用新的空目录并保留既有快照。数据仍属开发使用，semantic_targets 与路线类型保留 null；没有完整生成机器人任务有效性证据。
