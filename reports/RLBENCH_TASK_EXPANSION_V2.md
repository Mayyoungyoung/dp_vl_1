# 四个新增原任务的真实采集核验

固定源 `3b597f4f2497856c5316aefd9863d50d6812a081`，run `observation_task_expansion_v2/four_new_tasks`，PID 244111 / child 244112，2026-10-02 10:43:30–10:59:44 UTC，exit 0。每任务一个独立父场景，同父三次真实恢复后执行预声明自由接近提案加原任务演示；全部为 DEV_COLLECTION。

| 原任务 | 尝试 / 成功 | 三次尝试合计秒 | 记录事件切换次数 |
|---|---:|---:|---|
| pick_up_cup | 3 / 3 | 137.540 | 每条 1 次 |
| slide_block_to_target | 3 / 3 | 122.473 | 每条 1 次 |
| insert_onto_square_peg | 3 / 3 | 215.607 | 每条 2 次 |
| stack_blocks | 3 / 3 | 473.091 | 每条 4 次 |

共请求 4 父、12 次提案，12/12 通过原任务 success 条件，无 setup 失败、无已记录失败或近重复；总采集 972.882 秒，包含初始化／任务切换等开销。状态审计记录的物体清单、任务／机器人状态与 RGB、语言均 12/12 精确相等。逐条读取实际保存的 pose/open 数组核验了长度、有限性、轨迹 hash 和首 pose 到当前输入误差（全部为 0）；全部相机矩阵与初始 RGB-D 均在索引内。事件序列按原始时序保存，没有把抓放阶段拉平。

这些结果证明小批采集可行，不能证明学习方法、开放语言泛化或三种不同路线。路线类型仍为 null。自由前缀阶段有臂碰撞检查；整个原演示未做完整连续机器人碰撞认证，不能将原任务成功等同于全过程安全。

可复现依据：`observation_task_expansion_v2/four_new_tasks.status.json` 内保存完整执行命令及源目录；`audit_v1/artifact_index.json` 保存所有原始文件 SHA256 与服务器路径，`local_archive_verification.json` 验证本地归档一致。实际二进制位于本地 `.bootstrap/observation_task_expansion_v2` 和服务器原数据目录，不进入普通 Git。`scripts/audit_rlbench_pilot.py` 拒绝审计非 DEV_COLLECTION 数据；本轮实际 audit exit 0。

全部父图、路线和事件均展示于 [完整小批图](observation_task_expansion_v2/audit_v1/all_parent_routes_events.png)，无筛选示例。基于实测成本，后续六任务正式采集优先纳入 cup 和 slide，peg 与 stack 保留可行性证据，暂不扩大。
