# 两排窄ID正式机制语料：116父一次性登记

状态：实现/纯测试完成，尚未启动正式采集；仅在固定源码由主线程启动后才有实测。前置v6证据为四个新静态初态门禁全部通过、原失败布局27槽取得23参考，其中目标类型数5/7/2；它没有改变旧v5失败结论。

## 父角色与唯一预算

新seed固定283200–283315，服务器2026-10-02 17:59:06 UTC只读目录/文件名核验没有占用，见[机械seed核验](formal116_seed_presence.json)。seed不等于独立布局，另外对全部实测物理几何做指纹与分组门禁。

| index | 角色 | 父数 | 目标指令数 | 请求路线槽 |
|---|---|---:|---:|---:|
| 0–63 | TRAIN | 64 | 192 | 1728 |
| 64–75 | DEV_MODEL | 12 | 36 | 324 |
| 76–87 | DEV_SCORE | 12 | 36 | 324 |
| 88–99 | CALIBRATION | 12 | 36 | 324 |
| 100–115 | TEST_LOCKED | 16 | 48 | 432 |
| 合计 | | 116 | 348 | 3132 |

同父全部目标、路线、视角/语言变化都属于同角色。旧v4父283001及v5/v6组283100–283103永久留DEV_COLLECTION，完整九条历史机械记录列在 `configs/observed_two_row_formal116_exclusions_v1.json`，按geometry exact与1mm比较，排除颜色、seed、entry FK、robot drift、dummy和velocity。新预登记没有历史或内部重复，理想tip几何预检116/116通过；这不是机器人可达性证明。

配置 `configs/observed_two_row_formal116_v1.json`、一次性完整清单 `configs/observed_two_row_formal116_registered_v1.json`。清单parent_plan logical SHA `2f8deb2841e6c914601e0c337ffc9651339648b24de91f1411c506bd2a17b3a6`。一次RandomState(283199)，每父先12个geometry uniform再官方20色中不放回取3色，之后才分角色与固定分片。无失败替换、无重抽、无从结果反向选参数。

## 范围及边界

柱高固定0.14m，截面0.035m、base0.755m；两排x分别[.135,.145]/[.335,.345]，第一排两柱y[−.115,−.105]/[.105,.115]，第二排[−.155,−.145]/[.145,.155]。目标x[.45,.47]，三个y分别[−.175,−.145]/[−.015,.015]/[.145,.175]，z=.84。其余guide/check参数保持v4/v5。

初态为v6同一固定canonical q，取代v5随机entry；目标entry `[0,0,.865]`，实际FK和观测状态原样保存。不执行setup路径，不声称旧动态状态复现。初始化/恢复禁止IK/path，固定双restore/10step+warm-render并逐step碰撞检查；每条路线再完整世界/inventory/RGBD/camera/current/motor target零差核验。三个目标可见性与柱面depth审计不变。

这是**几何范围很窄的ID机制实验**，不是广泛机器人泛化。更宽几何OOD未登记或执行。四类旧开发采集不并入正式数据。目标球颜色从官方20色表取，目标identity与RGB必须一起用于识别；颜色索引、目标坐标、全几何、guide和验收标签都不会进入forward。

每父每目标九提案，实际raw分类允许lateral/over/unknown，已知类型数可以小于或大于K=4。接受仍是逐仿真步arm/gripper碰撞、完整raw 2cm tip、3cm端点、H24净距/类型稳定性；不是连续整机安全证书。有效unknown保留，重复/失败/partial全记预算。长度不作事后过滤，后续同时评价路径长度和覆盖。

## 顺序、两CPU与恢复

严格阶段：前16TRAIN全部闭合 → 全12DEV_MODEL闭合 → 余48TRAIN闭合 → score/cal/locked。两个固定分片各58父，在每一阶段使用全局closure屏障，快进程不会提前跨阶段；各CPU1，分别core2/3，GPU隐藏。总CPU保持不超过4，为后续GPU训练留CPU。每父预计约160–180s，仅线性预估116父两worker约2.6–2.9h；实际失败、等待、初始化/渲染成本全部计入，不是吞吐承诺。

统一入口 `scripts/collect_two_row_formal.py` 复用v6物理worker，没有复制或替换检查算法。输出 `parents/<ROLE>/<parent_id>/`、机械闭合 `closures/<index>.json`；目标初态/路线只在自己role目录，完整日志密封在每父record_job日志。后台进程/PID/source/hash/退出码保留。

slot_ledger只含started/completed、parent/input id和slot，不含成功标签。异常退出后原已发提案不重放：闭合partial，给已尝试/未尝试上下界，再恢复其他登记父。最后半行最多带来一个槽不确定性。活动PID/重复slot/源变更/机械hash改变会阻止恢复。单父运行错误闭合后停止该分片，避免盲目重复同一故障；已完成的负门禁仍是有效闭合。恢复的是语料进度，不声称跨进程复现已中断native状态。

## 下游输入和封存

observations.jsonl严格五键id/parent_id/split/image/instruction；supervision含实际路线、route_types（可null）、semantic_targets和 `route_config`（labels-only）。`route_config`路径位于该worker根 `route_configs/<parent>.json`，导出时必须正确加相对前缀；RGB-D/camera/current从同父observation NPZ读取，不能从route_config提取生成条件。

先导出固定16TRAIN+12DEV闭合前缀；不按成功数量换父。失败父/缺图/零参考保留请求分母，无输入就不能虚造推理样本。未采到的类型不是负例。缺sup但真实输入存在的中断父，仅可从同父已验证成功attempt及hash重建已知正例，不补未出现目标的图像/语言。

live_layout_gate只核注册及机械closure/receipt/ledger/artifact-index hash，包括全role机械指纹；不打开locked图像、路径或成功结果。实际1mm不同于注册、历史命中或跨role重复组均block模型使用；缺初态单列missing输入。导出、训练、评价之前必须重新运行gate。DEV_SCORE/CALIBRATION/TEST_LOCKED不进入持续方法选择。旧v5/v6 raw与失败原样保留。

启动入口为固定release中的 `scripts/launch_two_row_formal116_v1.sh <commit> fresh`；恢复用相同commit和 `resume`，每次独立session目录，绝不覆盖正在执行的wrapper。本卡不是已启动声明，实际状态以record_job为准。
