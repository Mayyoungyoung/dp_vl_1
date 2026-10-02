# 六个冻结碰撞配置：静态对象定位预登记

状态：原固定 release 的一次实跑在初始化父节点接口处失败，0/6 配置应用；[失败结果完整保留](OBSERVED_TWO_ROW_STATIC_CONTACTS_RESULTS.md)。只修根节点接口的版本已完成本地检查，尚未重跑。本诊断只处理前次 24-query 中实际碰撞的 query **5/7/13/15/21/23**；新 IK、路线提案和训练参考预算全部为 **0**。配置文件冻结原 manifest、24-query ledger、common world、NPZ 和 PNG 的五个 SHA256。记录被修改、缺少查询、返回配置不再对应原碰撞标签或 joint readback 非零时直接拒绝。

## 能复现什么，不能复现什么

原 endpoint 诊断没有落盘 native configuration trees 或 scene `.ttt`，没有保存查询瞬间的完整 solver/controller/contact 状态。不能在新进程声称原查询瞬间复现，也不再重复 v3 式隐藏状态重建。所有产物固定标记 `original_instant_reproduced=false`，原 24-query 标签保持只读。

本次是**同一固定模型、记录的静态场景布局与六个冻结关节向量下的碰撞对象比较**。模型来源由原 RLBench pinned archive SHA 检查，再逐字节核验实际 `task_design.ttt`、`task_ttms/reach_target.ttm` 与 archive 对应成员；默认 Panda 嵌在 `task_design.ttt`，不把未加载的独立 Panda model 当实际来源。当前 PyRep Arm 源 hash 必须与原诊断保存值一致。每个实际 shape 记录 bbox、collidable flag，以及 vertex/index/normal 数组的形状、dtype 和 SHA。旧查询没有运行时 mesh hash，这项证据不被追溯声称为当时已经核验。

按原记录恢复非机器人对象的世界 pose、目标颜色/布局、四柱原尺寸与 flags、机器人 model root，以及原共同 arm/gripper joints 和 target readback。对象 name/type inventory 和已保存的柱体 bbox 必须精确匹配。依赖/名称/必要字段不匹配则关闭全部比较。记录新旧 common world、RGB-D/current/camera 全字段差异；它们是复现边界的公开证据，不更换旧 reference、不归零或放宽旧严格 gate。即使某些字段相同，仍不声称原动态瞬间复现。

## 不启动物理，不重新采路线

直接加载固定 Scene 与 ReachTarget model。常规 `Environment.get_task` 会自动开始模拟，`task.reset` 可能进入 episode 规划核验，因此本入口不调用二者，而只用 `Scene.load` 加载模型。没有 setup IK 或准备运动。

诊断进程内部拦截 IK、路径规划和 simulation-start API：任何调用即记失败并停止。PyRep 启动/关闭所需的 UI 更新统一显式使用 `simExtStep(False)`；仅刷新界面，不走 physics step。每个配置前后检查 simulation time 不变。不会修改共享 PyRep 源码或影响其它作业。

在重建的静态场景内捕获一个新的活体 native snapshot。六个配置各自之前和之后均恢复此新 snapshot、固定丢弃一次渲染再取观测，world/inventory/joints/control/RGB-D/camera 必须严格零差；任一非零停止剩余配置。此门禁针对新共同静态状态，不替代对旧动态状态的诚实差异报告。

唯一应用操作为 `set_joint_positions(..., disable_dynamics=False)`，不插值、不步进；actual joints 必须与冻结向量精确相同。逐条保存新的完整 FK tip/world，另报告相对原查询 tip 的位置与角度差。不把新静态配置当成原动态轨迹样本。

## 碰撞归因边界

首先使用实际 arm collision collection 分别检查每个 collidable body；对命中的 body，再枚举 collidable 机器人 shape 的交叉检查，保存完整矩阵、调用次数、body 与 link 名称/handle、所有命中及实际 world pose。body 分为注册柱体、任务对象、机器人 body、其它 world body。单个 link 是否属于原 collection 未单独从内部取得，故 link/body 交叉检查是补充定位；collection→body 命中才是相应 aggregate 检查的直接分解。不会拿非 collidable 的可视 mesh 重叠充作原碰撞原因。

成功的意义只是在新静态对应场景中定位潜在碰撞对象。例如柱体命中支持提出一项新布局假设；桌面或机器人 body 命中则需要不同判断。没有命中或新旧 FK 差异明显时，同样保留并报告，不能硬凑柱体原因。无论结果如何，后续任何布局修改都必须重新做真实动态采集、原机器人碰撞/tip/H24 检查与实际路线类型验证。

## 实现与检查

新增独立脚本 `scripts/diagnose_two_row_static_contacts.py`、配置 `configs/observed_two_row_static_contacts_v1.json`、测试 `tests/test_two_row_static_contacts.py`；原 collector 和原 endpoint 脚本不变。新增 9 测试覆盖固定六配置与 hash、model archive 一致性、零容差 restore、非有限状态拒绝、对象分类、collection 与 link 两层矩阵、禁止 IK/路线/开始物理以及 UI-only 更新。合并旧检查实际 **36 passed / 0.62 s**。运行必须等固定 release 与 CPU1 授权，使用全新输出和不可变 launcher。

首次运行后仅增加根节点兼容修复：已验证的 inventory 句柄通过 `simGetObjectType` 核验，再用 pinned backend 原始 `lib.simGetObjectParent` 避开包装器对合法 `-1` 的错误检查。非根父节点必须属于完整 inventory，循环与重复句柄拒绝。新增 4 项层级测试，合并 **40 passed / 0.65 s**；没有放宽世界/图像恢复门槛或调用新的 IK。
