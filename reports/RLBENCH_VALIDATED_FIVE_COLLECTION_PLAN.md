# 独立六任务批次：五任务运动图像白名单策略

状态：固定 `84ee7a6bb00a151bdbb2ebc34d2ba3b60b407320` 已通过服务器 19 项测试和真实 1→2 跨进程恢复验证；root 已接续同一新语料启动正式批次，实际进度以作业为准。正在运行的 seed 281000、固定 6c 源批次完全保留。新批次使用独立 seed 282000，每任务仍 24 父、每父 3 个真实恢复提案，共 144 父 / 432 请求槽；任务名、父种子及角色已写入 `configs/observation_multitask_validated_five_v1_registration.json`。

相机策略必须显式传入 `--camera-policy validated-five-v1`；默认仍为 `on`。白名单仅包含 reach_target、pick_and_lift、take_lid_off_saucepan、pick_up_cup、slide_block_to_target，它们各一个已审计 TRAIN 父的 on/off 原始轨迹逐值等价。push_button 保持 on：该父的 on/on 控制也不确定，不能把 on/off 差异归因为渲染，亦不将其计作等价通过。策略只作用于 get_demo，初态、严格恢复及自由前缀观测保持 RGB-D 开启，正常及异常路径均在 finally 恢复配置。每条实际尝试保存请求策略、实际生效范围、配置恢复状态、源码 hash、原任务成功与失败；前缀先失败而未调用 demo 时明确没有 render audit，不谎称已经抑制渲染。

父角色仍固定为每任务 TRAIN 16、DEV_MODEL 2、DEV_SCORE 2、CALIBRATION 2、TEST_LOCKED 2。调度顺序在采集前固定为父索引 `[0,16,1,17,2..15,18..23]`，每个索引依次遍历六任务，让每项任务的 TRAIN 和 DEV 较早可用；不依据成功结果选父。注册角色不随执行顺序改变。variation 仍为 parent_index modulo variation_count，角色间 variation 分布可能不同，不声称 IID。

主进程与子 worker 显式传同一策略和顺序；已有注册清单不允许恢复时更换任一项。模型输入边界、成功标准、事件记录、完整失败分母和恢复条件保持不变。新批次没有把旧数据复制成新父。开启白名单不等于所有随机父已证明轨迹等价，更不等于总采集提速 48 倍；性能证据及 13 个额外实际审计提案见 `RLBENCH_DEMO_RENDER_PERFORMANCE.md`。

不同 seed 不能直接证明物理场景独立。新 worker 在严格初态生成 `mechanical_fingerprint.json`：规范化全 world、当前 pose、RGB 数组/文件与 depth 的 hash；另以 task shape/joint 的物理 pose 生成精确及量化 layout hash。layout 分组排除颜色、速度、机器人、waypoint/成功 sensor 等 dummy，避免仅因语言/颜色/检查对象变化就声称新布局。量化位置为 1 mm、四元数分量及 joint 为 1e-4，并统一 quaternion 正负号。该摘要不含完整 mesh/bbox，量化命中只是保守重复候选，不同 hash 也不是独立性证明。

`layout_usage_gate.json` 仅读取这些机械摘要；任何任务的相同 layout/图像跨角色出现，均标记 `blocked_cross_split_layout_duplicate`。所有父、失败与原角色继续原样保存；不事后移除不利个体来修复划分。**下游快照/训练前必须读取 gate，阻止被标记任务跨角色使用，等待新的父组划分版本。** 采集器的原始角色 manifest 是归档，不是绕过 gate 的训练许可；本次没有修改既有训练器来自动消费新语料。

`scripts/audit_multitask_layout_hashes.py` 可跨两个批次做只读机械审计。对旧 6c，只有已存 reference_pointer 图像文件 hash 时明确记 `legacy_missing_hash_unverified`；绝不为补齐 hash 打开 locked 的 reference world、图像、路径或验收结果。启动器末尾执行跨批次审计；若数据尚在采集时提前构造训练快照，也必须先用同一脚本做当时的机械审计。旧批次角色不会改写。

本地及服务器 19 项测试均通过，覆盖默认 on、白名单 push 例外、异常恢复、policy/顺序变更拒绝、子进程参数、预声明交错角色、新 seed 不重叠、颜色/四元数符号变化仍归同物理组、跨 split 保守 gate、legacy locked 只读 pointer。服务器执行 0.35 秒，启动器 shell 语法检查通过。

真实验证直接使用新语料 `TRAIN/reach_target_282000`。首进程 PID 316256 提交 slot 0 后按预声明 exit 3，新进程继续 slot 1、2，outer proof exit 0；3/3 原任务成功，每次恢复的世界、物体清单、RGB-D、相机、当前状态与语言都精确相等。三次 demo 均实际关闭 RGB-D 并在 finally 恢复；首 slot 记录、机械 fingerprint、注册清单与源码清单 SHA 在恢复前后保持不变。两个 worker 累计 46.397 秒，数据留在新 corpus，未复制或重采。证据见 `observation_multitask_validated_resume_probe_v1/`。root 后续作业 `observation_multitask_validated_five_formal_v1` 沿同 corpus 跳过闭合首父继续；本 agent 没有重复启动。

第一次只读跨批次机械审计实际读取 35 个已保存摘要：新批次已有 2 个物理 layout hash，旧 6c 为 33 个 legacy 图像文件 hash。未发现候选重复，但旧批次物理布局仍为未核验，不能将该结果当独立性保证；审计未打开 locked 图像、reference world、路径或任务结果。随着采集新增父，任何模型快照必须重新读取当时 gate，不可沿用这个初始审计结论。
