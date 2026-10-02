# 两排低柱：有限端点 IK 诊断预登记

状态：只读证据、独立实现和本地测试已完成；未启动本诊断模拟器或额外 IK。原 v1/v2 路线及 v3 重建失败全部保留。新脚本 `scripts/diagnose_two_row_endpoint_ik.py` 与配置 `configs/observed_two_row_endpoint_ik_v1.json` 需要固定 source 后才能执行，不修改仍在运行的采集器。

## 已有日志缩小了问题范围

v2 全部 20 个 `ConfigurationPathError` 的 chained traceback 都包含 `solve_ik_via_sampling` 的 `ConfigurationError`：在碰撞过滤开启时，给定朝向的端点关节配置未在预算内找到，尚未调用后续 `getNonlinearPath`。这比笼统归为 OMPL 路径搜索失败更具体，但仍不是“不存在解”的证明。原 v1 成功/失败路径内部的分支没有同样完整插桩，不能把本结论外推为全部 v1 绕行的原因。[逐失败索引与输入 hash](observed_two_row_pilot_v2/existing_endpoint_ik_evidence.json)

| row-plane 目标 xyz（m） | `get_path` 成功 | 端点 IK 失败 | 模拟到达该 guide |
|---|---:|---:|---:|
| (0.14, −0.25, 0.865) | 9 | 0 | 9 |
| (0.14, 0, 0.865) | 0 | 9 | 0 |
| (0.14, 0.25, 0.865) | 7 | 0 | 7 |
| (0.34, −0.25, 0.865) | 0 | 6 | 0 |
| (0.34, 0, 0.865) | 5 | 0 | 5 |
| (0.34, 0.25, 0.865) | 0 | 5 | 0 |

这些不是相互独立的重复实验，也不是完整路线有效率；前序失败会使后续 guide 未调用。实际 Panda 模型根位于 x≈−0.309 m、joint1 原点 x≈−0.268 m。第一排 middle 距 joint1 原点约 0.463 m，第二排 −y 点约 0.690 m。距离不能证明可达性，但已否定“世界 x 小所以一定靠近机器人基座”的推断；没有依据将整体 +x 平移作为优先修复。

## 官方实现与真实坐标核验

固定 PyRep revision `8f420be8064b1970aae18a9cfbc978dfb15747ef` 的 [Arm 源码](https://github.com/stepjam/PyRep/blob/8f420be8064b1970aae18a9cfbc978dfb15747ef/pyrep/robots/arms/arm.py) 明确支持同一采样 IK API 的碰撞开关、trial 数、返回配置数和单次配置搜索时间参数。`get_path` 先试线性路径，其后先求端点配置，再调用非线性规划。本机实际 `arm.py` SHA256 为 `4c0d5e09aa777de145c184f05a24ac74bb770aedb534dc0c9501d726c17ebe5d`。因此这次直接调用端点 API，可将姿态和碰撞过滤分开审查，完全不生成 setup 之后的路线。

官方 [Panda 模型入口](https://github.com/stepjam/PyRep/blob/8f420be8064b1970aae18a9cfbc978dfb15747ef/pyrep/robots/arms/panda.py) 使用七关节 Arm；实际关节区间由当前模拟器读取，而非把外部 URDF 数字强行当本场景真值。Franka 官方 [Panda 描述](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/panda.urdf.xacro) 默认 TCP 沿 hand 的 +Z 偏移；[关节几何](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/common/franka_arm.xacro) 和 [关节限制](https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml) 说明位置范数之外仍受多关节几何和限制约束。它们是轴语义与机制参考，未声称已逐网格复现官方机器人。

更直接的依据是已存真实 v2 场景：tip 的 local +Z 在世界系约为 `(0.239731, 0.001448, −0.970838)`，与 `Panda_attachment → tip` 的单位向量点积 **0.9998345**，相对世界竖直向下倾斜 **13.87094°**。新诊断在自己的共同初态重新测这两个向量；点积低于固定 0.99 则停止，不猜轴或继续尝试另一个朝向。

唯一替代朝向由真实 tip 四元数计算：将 local +X 投影到世界 XY 后归一化作为水平 heading，指定 local +Z 为 `(0,0,−1)`，由右手正交基构造 xyzw 四元数。因此不硬编码未经验证的 `[0,1,0,0]`。v2 记录的数值示例为约 `[-0.00011607, 0.999999993, 0, 0]`；实际运行会保存自己测得的原/新四元数及完整轴矩阵。投影退化则失败，不选备用 heading。

## 固定 24 调用与共同初态

只保留现有六个 row-plane 点。每点依次执行四个条件：原始真实四元数/开启碰撞过滤、原始/关闭、竖直/开启、竖直/关闭。每条件唯一一次 `solve_ik_via_sampling`，相同 `trials=128`、`max_configs=1`、`max_time_ms=10`、`distance_threshold=0.65`；没有更多朝向、自动重试或自适应增预算。128 是本次廉价诊断预先固定的共同上限，低于原 `get_path` 默认 300；因此两者阴性结果不能直接比较，任何未找到均只解释为本次有限预算未找到。

1. 按原物理布局执行一次单独计费的两段 setup，并保存逐步 pose/open/joints、全部规划调用/模拟步和失败。不是加载 v1 隐藏状态，也不声称跨进程 same-seed 配对。
2. 在同一个进程建立新的活体 native snapshot，固定双恢复/显式渲染；保存共同初态 world/inventory/RGB-D/current/calibration、joint target position/velocity 与真实区间。
3. 每个查询之前从此同一 snapshot 严格恢复，记录全部原有 world/图像检查及 target readback；任何非零差异即停止。不能只核验末端。
4. 记录请求 pose、实际起始 joints、所有返回 joints、完整异常、实际低层 `simGetConfigForTipPose` 调用数和时间。调用计数仅在本诊断进程临时包装，finally 恢复函数，未修改共享库源码。
5. 若返回配置，仅用无 physics step 的 joint-position 设置做静态 FK/readback，记录请求与实际 joints 差、末端位置/角度误差、原 arm/gripper 碰撞检查与 tip 2 cm 点余量。返回配置不等于有效路径；若设置未生效，readback 会明确暴露差值。
6. 无论求解成功或失败，均在同一 snapshot 再严格恢复后才能继续。每次前/后恢复的 20 个 canonicalization 步单列，不冒称零总模拟成本。忽略碰撞得到的配置绝不加入正参考或训练数据。

上限为 24 个端点 API 调用、3072 个底层配置搜索、名义单调用时间参数之和 30.72 s；这不是端到端墙钟保证。setup 最多另有 2 个 `get_path` 调用。总墙钟 180 s 截止采用逐查询前检查，已开始的有界调用及必要恢复允许完成，之后全部余项保留为未尝试。保存实际时间而非把名义限额称为实测。没有 setup 后路线提案，没有数据集正参考。

模拟器内部随机流未证明可恢复，故 `simulator_random_stream_paired=false`；共同 snapshot 保证检查过的物理与观测初态，不能声称逐次随机搜索流相同或用一次二元结果估计成功率。

## 可证伪判断与后续唯一候选

当前优先候选是**同几何下使用竖直 tip 朝向**，理由是原朝向继承了默认初始机器人姿态，球体 reach 任务并未要求约 14° 倾斜，而失败已经发生在端点姿态配置阶段。这仍只是待检验假设。

- 如果原先三类失败点的竖直条件返回可核验、arm/gripper 均不碰撞的配置，而原条件仍未找到，且其余三点没有退化，则支持进入一个另行冻结的竖直朝向路线可采性 pilot。它仍须完整原标准、27 槽预算与实际轨迹分类，端点证据不能替代 >K 不同有效路线证据。
- 若只有 ignore-collision 条件返回配置，说明碰撞过滤可能参与瓶颈；不视为可采性改善，也不据此放宽碰撞。
- 两朝向均未找到只说明这次预算阴性；两者均找到则应重点检查连接/规划和初态分支。都不自动增加 trial、扫朝向或移动柱体。

任何后续跨规划条件因果比较必须在一个进程共享同一真实 snapshot，保留不同条件预算；不使用历史 v1/v2 same-seed 作为严格配对。数据升级是目标，本诊断及执行器改动不作为方法创新。

本地新增 10 项测试与原 17 项合计 **27 passed / 0.46 s**：24 条件完整性、固定预算、真实姿态推导/符号不变性/退化拒绝、轴读值、底层调用计数及异常后恢复、零容差共同状态门禁。尚未运行服务器模拟，成功与失败均未预判。
