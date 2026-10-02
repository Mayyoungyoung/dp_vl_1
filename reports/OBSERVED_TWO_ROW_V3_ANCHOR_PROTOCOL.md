# 两排低柱 v3：重建 v1 已记录状态，逐位审计后才允许采集

状态：以下为执行前预登记；现已按固定源执行一次，严格门禁失败、27 槽未尝试，见 [实际结果](OBSERVED_TWO_ROW_PILOT_V3_RESULTS.md)。v1/v2 的全部负结果与“不属于严格配对”的判断保持不变。v3 配置为 `configs/observed_two_row_pilot_v3.json`，角色仍为同父 DEV_COLLECTION；物理布局、seed、九个通道提案、v2 九点引导和全部评价阈值不变。

## 可用与缺失的恢复证据

v1 保存了准备全过程 pose/open、7 维 arm joints、2 维 gripper joints，严格初态的 task/全局对象 pose、arm joints/velocity、gripper joints/pose、task/post velocity、box 尺寸/flags、颜色、全 inventory、RGB-D、相机和当前状态。配置冻结原 `manifest.json`、准备轨迹、世界审计、observation 与 front image 的五个实际 SHA256，不能改用另一次成功状态。

v1 不是任意完整模拟器动力学快照。缺少落盘 native task/arm/gripper configuration trees，以及初态的 joint-target readback、gripper 速度、joint forces/torques、每个 robot link 速度、接触求解器 warm-start 与控制器内部状态。原 canonical restore 会重置物理引擎，将位置/速度 targets 设为保存 joint 配置/0，再固定 10 步；因此本轮使用该已知初始化协议重建，而不声称恢复未保存的隐藏状态。

实际记录的 arm 速度接近零但**不等于零**：最大绝对值 0.0005292892456054688 rad/s，七维为 `[-0.0001811981, 0.0005292892, 0.0000190735, 0.0003957748, -0.0000476837, -0.0000143051, 0.0001049042]`（此处显示舍入，实际检查使用完整 JSON 值）。所有记录的 task/post shape velocity 为零。不能将小速度归零或改成容差检查以通过恢复。

## 唯一重建流程与停止门禁

1. 核验冻结的五个 v1 文件 hash，物理、seed、阈值与 v1 一致。读取准备轨迹的最后一个 arm/gripper joint 配置，使用的是 v1 native snapshot 捕获前的配置，避免把已完成 10 步的观测状态再次当成原始配置而重复演化。
2. 新场景按同一顺序创建同样 task、robot、四柱和颜色。恢复 v1 记录的 task/post pose/color/flags，生成当前活体 native 配置树，将原恢复协议中的 arm/gripper joint 配置替换为 v1 准备末帧。**不调用 setup IK，不再执行随机规划准备动作。**
3. 沿固定双 native restore、间隔显式 render、每次 10 步的原协议重建，再读取完整当前审计和 RGB-D/camera/state。比较所有记录字段与 v1，包含微小非零 arm velocity、全局 pose 和完整 inventory；所有差异必须恰为 0，缺失字段/形状不符/非有限值均失败。
4. 任一差异则 `setup_failed`，写真实 readback/失败图像/全世界状态；保留 1 次重建 action、0 路线尝试、27 未尝试槽，并结束。本版本不自动再调 joint、不更换 reference、不放宽阈值。
5. 只有严格 readback 通过且原碰撞/可见性/深度几何检查也通过，才执行 v2 原九个通道序列 × 三个目标的唯一 27 槽；每槽九次规划上限，setup 规划调用为 0。后续每个候选仍从同一活体 native snapshot 严格恢复，不用提案 ID 替代实际分类。

门禁名称为 `strict_v1_recorded_state_readback_v1`。通过只保证全部已记录字段严格一致，`arbitrary_hidden_dynamic_state_equivalence` 固定 null。若通过，仍应在同源离线分析中复核 v1/v3 初态，而非只看末端坐标。

## 后续可恢复产物

核验 pinned PyRep 源码发现：`get_configuration_tree()` 虽标注 bytes，底层实际返回没有长度信息的 CFFI 指针。不能猜长度、按 NUL 截断或把内存地址当恢复文件。v3 若门禁通过，改用官方已提供的 `PyRep.export_scene()` 保存原生 `verified_initial_scene.ttt` 和 SHA；仍明确 `roundtrip_verified=false`，未验证重新载入时不声称其含所有隐藏动力学状态。原生二进制仅放数据目录，不进普通 Git。

## 测试与预算

当前本地 **17 passed / 0.38 s**：原九点与几何测试、记录状态精确相等、tuple/list inventory 规范化，任一速度/深度/RGB/相机/库存/字段差异关闭门禁，hash 损坏拒绝，以及读取准备末帧而非观测初态 joint 的检查。尚未运行真实重建；可能严格失败，此分支是预期且必须如实保留。

请求分母仍 1 父、1 重建 action、27 路线槽。若门禁失败不“凑够”27 条；若通过，最大完整路线规划调用 27×9=243。记录实际 setup/restore 时间、规划调用/耗时、模拟耗时、失败和全部候选预算。是否跨进程能完全再现是本轮待验证问题，不能预先称 v1 状态已经恢复成功。
