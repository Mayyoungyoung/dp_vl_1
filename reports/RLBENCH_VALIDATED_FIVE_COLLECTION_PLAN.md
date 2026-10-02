# 独立六任务批次：五任务运动图像白名单策略

状态：源码、测试、预注册清单和启动器已准备，尚未启动新采集。正在运行的 seed 281000、固定 6c 源批次完全保留。新批次使用独立 seed 282000，每任务仍 24 父、每父 3 个真实恢复提案，共 144 父 / 432 请求槽；任务名、父种子及角色已写入 `configs/observation_multitask_validated_five_v1_registration.json`。

相机策略必须显式传入 `--camera-policy validated-five-v1`；默认仍为 `on`。白名单仅包含 reach_target、pick_and_lift、take_lid_off_saucepan、pick_up_cup、slide_block_to_target，它们各一个已审计 TRAIN 父的 on/off 原始轨迹逐值等价。push_button 保持 on：该父的 on/on 控制也不确定，不能把 on/off 差异归因为渲染，亦不将其计作等价通过。策略只作用于 get_demo，初态、严格恢复及自由前缀观测保持 RGB-D 开启，正常及异常路径均在 finally 恢复配置。每条实际尝试保存请求策略、实际生效范围、配置恢复状态、源码 hash、原任务成功与失败；前缀先失败而未调用 demo 时明确没有 render audit，不谎称已经抑制渲染。

父角色仍固定为每任务 TRAIN 16、DEV_MODEL 2、DEV_SCORE 2、CALIBRATION 2、TEST_LOCKED 2。调度顺序在采集前固定为父索引 `[0,16,1,17,2..15,18..23]`，每个索引依次遍历六任务，让每项任务的 TRAIN 和 DEV 较早可用；不依据成功结果选父。注册角色不随执行顺序改变。variation 仍为 parent_index modulo variation_count，角色间 variation 分布可能不同，不声称 IID。

主进程与子 worker 显式传同一策略和顺序；已有注册清单不允许恢复时更换任一项。模型输入边界、成功标准、事件记录、完整失败分母和恢复条件保持不变。新批次没有把旧数据复制成新父。开启白名单不等于所有随机父已证明轨迹等价，更不等于总采集提速 48 倍；性能证据及 13 个额外实际审计提案见 `RLBENCH_DEMO_RENDER_PERFORMANCE.md`。

不同 seed 不能直接证明物理场景独立。新 worker 在严格初态生成 `mechanical_fingerprint.json`：规范化全 world、当前 pose、RGB 数组/文件与 depth 的 hash；另以 task shape/joint 的物理 pose 生成精确及量化 layout hash。layout 分组排除颜色、速度、机器人、waypoint/成功 sensor 等 dummy，避免仅因语言/颜色/检查对象变化就声称新布局。量化位置为 1 mm、四元数分量及 joint 为 1e-4，并统一 quaternion 正负号。该摘要不含完整 mesh/bbox，量化命中只是保守重复候选，不同 hash 也不是独立性证明。

`layout_usage_gate.json` 仅读取这些机械摘要；任何任务的相同 layout/图像跨角色出现，均标记 `blocked_cross_split_layout_duplicate`。所有父、失败与原角色继续原样保存；不事后移除不利个体来修复划分。**下游快照/训练前必须读取 gate，阻止被标记任务跨角色使用，等待新的父组划分版本。** 采集器的原始角色 manifest 是归档，不是绕过 gate 的训练许可；本次没有修改既有训练器来自动消费新语料。

`scripts/audit_multitask_layout_hashes.py` 可跨两个批次做只读机械审计。对旧 6c，只有已存 reference_pointer 图像文件 hash 时明确记 `legacy_missing_hash_unverified`；绝不为补齐 hash 打开 locked 的 reference world、图像、路径或验收结果。启动器末尾执行跨批次审计；若数据尚在采集时提前构造训练快照，也必须先用同一脚本做当时的机械审计。旧批次角色不会改写。

本地 19 项测试已通过，覆盖默认 on、白名单 push 例外、异常恢复、policy/顺序变更拒绝、子进程参数、预声明交错角色、新 seed 不重叠、颜色/四元数符号变化仍归同物理组、跨 split 保守 gate、legacy locked 只读 pointer。启动器 shell 语法检查通过。实际模拟器新版本验证和服务器实测仍需固定 release 后执行，不能以这些纯文件测试替代。
