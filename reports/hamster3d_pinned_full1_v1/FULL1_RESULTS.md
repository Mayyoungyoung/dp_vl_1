# HAMSTER 单个 TRAIN 请求：完整格式与资源实测

固定 source `04ed74fbc2912af58d65c487ce9947ea5cc71dc1` 的单个 `two_row_reach_283200_target0` 请求实际完成：1次generate、1个候选、151次forward、151生成token，正常EOS。官方v5 parser获得4点，独立严格校验确认整段合法JSON fence、数组/点/事件结构完整；不是fallback提取后补全JSON。**本阶段任务质量、碰撞、目标和执行成功均为null，未读supervision或验收几何。**

原始请求为“Move the gripper to touch the black sphere while avoiding the gray posts.”，沿已冻结官方模板追加完整waypoint/深度/夹爪JSON格式要求。官方代码 `97216a`、模型 `ddc5987` 及bf16算子不变；固定0..3层驻留、余32层同步pinned传输且不逐参数清缓存，未量化。输入只有既有RGB、真实观测depth与语言；相机参数只用于输出坐标转换，没有测试目标坐标或障碍真值进入模型。

完整输出如下，原文含合法JSON fence，未修补或筛点：

| 点 | 官方归一化u,v | 预测光轴深度m | 官方事件 |
|---|---|---:|---|
| 1 | 316, 863 | 1.22 | Close Gripper |
| 2 | 312, 737 | 1.17 | None |
| 3 | 676, 630 | 1.21 | None |
| 4 | 663, 705 | 1.25 | Open Gripper |

四点不等同于H24路线或额外四候选。Close→Open事件并不符合“仅移动到目标”的默认reach描述，不能因解析成功称为reach成功。原输入current处于open；首返回world点与观测current位置距离0.424272m，仅作输入/输出关系描述，未插入当前点、连接段或修复路径。HAMSTER输出可能表示关键操作waypoint，不能把缺失起始运动悄当已生成。

前8次真实输出logits均为 `[1,1,151936]` bf16，与封存exact8逐值相同且finite；整条输出前87token与旧300秒失败输出相同，原445-token prompt也逐元素相同。旧失败不是被覆盖的新成功，原两个exact8 partial调用仍独立保留。32层卸载计数与151次真实forward核对，官方参数字节未变。原输入相机数组与保存预测NPZ一致，world→pixel回投影最大差 `7.97e-6` 原224像素；该roundtrip验证转换自洽，不证明物理目标正确。

| 成本层级 | 实际秒数 |
|---|---:|
| record_job完整外层 | 504.697245 |
| runner（含子进程和前后guard） | 504.632203 |
| probe主体至退出状态准备 | 500.456321 |
| 模型加载 | 10.554284 |
| 输入准备 | 0.646271 |
| 传输hook转换/权重验证 | 137.595700 |
| 唯一生成请求 | 183.939749 |
| 输出后参数校验 | 31.726843 |

上表均为嵌套时间，不能相加。首forward15.503775秒；后150次decode forward中位1.097004秒，含GPU同步/finite检测与前8组CPU logits捕获的匹配规则。此数字不是独立热启动端到端吞吐。全部logit检测/CPU捕获记录合计0.096606秒；外层还包括历史证据及文件hash、解析、封存等开销。原1800秒配置是主要计算阶段的剩余预算，不是覆盖全部初始化/收尾IO的硬墙钟。

外层实际 `2026-10-03T06:01:45.446344+00:00` 至 `06:10:10.143589+00:00`，PID810845/child810850；科学脚本PID810851。GPU1 UUID按授权核验，CPU0单线程；外层GPU占用时间 **0.140193679小时**，主体0.139015645小时为嵌套不另加。峰值reserved6,622,806,016B、allocated6,498,190,848B低于8,864,694,272B上限；峰值RSS21,799,170,048B低于32GiB。旧下载/上传、环境、失败probe、exact8与48测试成本各自保留，不重复加成本次费用。

实际视觉QA已查看两张完整图：`figures/RGB_FOUR_POINT_OVERLAY.png` 和 `figures/FOUR_POINT_COORDINATES.png`。224原图与640输入图完整，全部4点和Close/None/None/Open标记可见。点1投影位于黑球附近，点2向上，点3跨向图右，点4位于右侧灰柱附近；这是图像位置描述，**没有借图判断三维碰撞或目标正确率**。连线仅展示原四点顺序，不是模型额外生成的连续轨迹。三维图未画任何验收障碍或真目标。

root也已独立实际查看上述两图及原prompt，确认四点完整可见、第一点靠黑球图像位置而后续跨向右侧，事件原值为Close→None→None→Open。其QA同样只覆盖视觉呈现与原文，不是checker或机器人执行验证。原4点只形成3条已返回相邻线段；从current到首点的运动尚未由该输出覆盖。

归档38个运行原件，全部SHA核验；7个NumPy文件置于 `runs/synced_hamster3d_pinned_full1_v1` ignored路径，1个first8 logits PT仅保留服务器路径/大小/hash，不下载权重。另有输入限定的front.png/observation.npz，其严格5键无目标或未来轨迹；本阶段没有读取其他TRAIN样本。完整原失败输出随正式probe保存并归档。源、冻结manifest、实际wrapper、原状态/日志/账本及root完成快照全保留；442源字节前后校验通过。

关键SHA：预测坐标NPZ `71a58d2602f8fc5b589f8d389d921c169f68037a3e85a8564c481f3c9bb81213`；科学artifact index `7d863698d91f836c4137e30a9357ac21f1fe2fbe25a95fe3a41898e8d63d5138`；inner status `f82c5730c7d25c5106c8db10d68eec178b3417ac671b2269ffbde4fe787c61e1`；运行原件tar `e2cea5001fd2a767ab8819d91bb595d75459c934bb81e099d70bdb12eb539bd4`。来源恢复见REMOTE_ARTIFACT_INDEX，归档及NumPy本地路径见LOCAL_ARTIFACT_INDEX。

本结果只证明这个请求可在授权资源上生成完整官方格式且传输改动保值。没有多任务/泛化/机器人执行或相同端到端预算质量结论。下一步只能在独立授权的零生成技术验收中使用这个已封存K1，不能混入其他请求或把技术调试partial合并成候选池。
