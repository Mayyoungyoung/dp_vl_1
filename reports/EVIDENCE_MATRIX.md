# 主张—证据对应表

| 主张 | 状态 | 实际证据 | 边界 |
|---|---|---|---|
| 事件位置辅助改善六任务普通头 | 三种子开发证据 | best ADE9.855→9.083cm、末端17.928→15.689cm；actual采样链/初始化配对通过 | 固定1500 ADE仅2/3改善，杯子退化；常规辅助，非集合创新/语义/碰撞/执行成功 |
| SFT空间失败来自单位或因果接口错位 | 当前小批审计不支持 | 9forward、1113位置×151936词表前缀logits完全相同；单位/mask/shift通过 | 不排除欠拟合/暴露偏差；真前缀末端拟合不能证明视觉定位，下一仅greedyTRAIN8 |
| 真实Qwen已完成直接路线集合SFT | 训练和生成实测完成；当前质量失败 | 3750路线目标槽，8adapter更新；独立4/整集合4全24DEV TipValid与语义均0，格式21/96与5/96 | NLL .4367不是路线质量；有限训练曝光尚不足以否定SFT能力，不把当前失败称强基线 |
| 扩充六任务TRAIN改善相同DEV的普通头 | 单种子开发证据 | 相同1500x32，12→48→96请求TRAIN父（最后95正参考），DEV ADE17.12→12.94→10.00cm、endpoint24.50→19.70→17.43cm | 同时增加variation/颜色；第三点8/12父改善、4变差，reach/slide末端>30cm，无语义/碰撞/执行认证，不是新机制 |
| 语法约束能解决VLM路线空间错误 | 当前TRAIN诊断否证 | 固定8TRAIN K1全部格式通过、终点0/8在3cm内，平均57.37cm；0eeeecb实测 | 标准格式修复；不是DEV结果，也不是训练充分性证明。停止自动DEV扩展 |
| 两排物理通道已提供超过K4有效类型 | 尚未支持 | v1 18/27通过、每目标侧向类3/1/1；v2 4/27通过且20规划失败 | v1/v2各自恢复严格通过，但跨版本初始关节不一致，不能归因引导点；未知类型不强行算不同 |
| 历史集合回归优于现有两类扩散 | 已复核 | historical commit d0d97eb；multiseed_summary.json；audit_v2/historical_regressor_seed0_metrics.json | 固定三路、真几何/真终点、冻结CLIP；旧TEST/OOD已公开使用 |
| 已构建可变路线类型和多障碍探针 | 已实现并验证 | multigate.py；audit_v2/multigate_v1.manifest.json；1152父场景7131正例 | 两墙平面通道结构、恒定z、固定语言；不是机器人观测任务 |
| 随机K参考子集会造成路径平均 | 受控单种子支持 | v2_round1逐场景结果、配对区间、ROUND1_REVIEW.md | 已知多模态回归问题，不能单独作为创新 |
| 类型不足K时应允许有效重复 | 受控单种子支持 | v2_round2/saturation_summary.json；精确饱和匹配与穷举测试 | 当前开发探针达到可覆盖类型上限；尚无独立任务结论 |
| 补全覆盖记忆稳定优于普通attention | 未获支持 | COMPLETION_THREE_SEED.md；三种子新增有效类型差+0.1641/+0.0156/-0.0469 | seed2反转；父场景bootstrap不能代替训练种子稳定性；不提升为核心方法 |
| 自产草稿补全优于一次联合生成 | 修正后仍未获支持 | v2_completion_selfdraft：attention2+2=3.3047/joint4=3.3125；coverage3.2813/3.3438 | 模型草稿混合确实修复大量失败，但双前向没有优势，保留joint4 |
| 真正使用Qwen3-VL训练/推理 | 已实测 | 官方固定revision；96真实RGB+语言前向；observed_frozen_v1；observed_online_v1 | 不是CLIP替代；真正接入不等于路线质量已达标 |
| 真实LoRA参数更新 | 已实测 | 末两层q/v共114688参数，8张量非零梯度+改变hash；observed_online_v1/lora_seed0 | 短训练无质量收益，不将“参数更新”解释为有效微调；缓存未复用 |
| 观测语义目标精确定位 | 部分积极开发证据 | OBSERVATION_GROUNDING_THREE_SEED：同RGBD头辅助监督使严格3cm均值1.04%→28.82%，终点18.92→14.51cm | 全24指令/8父，23参考；三种子一致但小DEV反复选优，常规基线修正不是核心创新 |
| LoRA提升观测路线质量 | 尚未获支持 | 300步从零头及500步同头warm-start配对；后者两边均选step0 | 实际参数更新与有效微调分别报告；当前不能称充分训练的强LoRA基线 |
| 同初态RLBench多路线采集 | 已完成小批实采 | derived32:275/288成功、32父、restore/RGB0差；4原任务12/12采集成功 | 采集成功不是模型执行；未据轨迹距离定义不同类型，连续全身碰撞未认证 |
| 真障碍下可判别的多路线采集 | 小批通过并正在扩大 | explicit-render四父48尝试20路径17类型，48/48恢复0；实际每步arm/gripper碰撞审计 | RLBench-derived扩展；失败21规划+7碰撞保留；不是原benchmark或生成模型成功 |
| 新布局训练后定位辅助仍有收益 | 已完成配对开发实验 | 新32父三种子 strict0→28.82%，参考终点18.54→12.19cm；新64父seed0 strict0→19.79%、终点15.70→10.50cm | 同一旧8父DEV，固定训练曝光；不证明扩大数据必然提高严格正确率 |
| 任务专用传统定位已是强对照 | 已实测 | TRAIN颜色原型＋RGB-D，旧/新训练集严格19/24与20/24；保留背景误选大离群 | 仅已见精确指令、单终点，无完整路径；参考误差23与目标中心误差24不能混比 |
| 局部编辑门控提高变化后集合质量 | 续训后仍未获支持，放弃此实现 | CONSTRAINT_UPDATE_V2_RESULTS：4000步b1/b2局部较自由补全少0.06537/0.11800有效类型 | 受控真几何；同数据/信息/网络规模、累计候选4+b；停止继续堆模块 |
| 神经观测模型已能稳定绕障 | 未获支持 | obstacle16 plain/aux TipValid2.08%/4.17%，Any均8.33%；12指令全量图 | 仅箱体末端线段检查，非手臂、桌面或执行认证；实际训练失败仍大量存在 |
| 路径内部损失与定位损失冲突 | 当前诊断未支持 | 固定6个TRAIN样本、best250/last1000共享参数梯度cos为正，损失重构误差<3e-9 | 一批局部诊断，不能证明全局无冲突；没有据此增加detach模块 |
| 软坐标均值会损害当前观测定位 | 配对三种子支持常规修复 | OBSERVATION_PEAK_ANCHOR；natural64严格语义17.36±13.19%→83.68±6.62%，参考ADE6.47→4.55cm；同参数、数据、曝光 | 仍是旧8父DEV；峰值读取是传统定位修复，不能包装为集合新机制 |
| 峰值读取稳定改善障碍路线 | 尚未获支持 | obstacle32原ADE选模语义19.44→28.47%但峰值seed1为0，ADE18.38→19.30cm；seed0 TipValid4/48→8/48 | 额外固定步检查必须双方所有seed；不替换主选模；box tip检查不等于机器人执行 |
| 在线RGB-D LoRA改善共享预训练头 | 未获支持 | OBSERVATION_ONLINE_RGBD_PAIR：两臂1000×4，均选共同step0；last均退化 | 真实Qwen、8adapter张量实际更新；原始失败与公共预训练成本保留 |
| 观测A*传统基线已合理运行 | v2已实测 | OBSERVED_ASTAR_V2_PROTOCOL：固定TRAIN修复后44/48 TipValid，UniqueClassified1.333，median1.969s；v1失败保留 | 闭集指令原型、末端箱体检查；低覆盖且非整机执行，尚无固定时间公平对比 |
| 峰值定位收益迁移到新布局 | 三种子开发支持 | OBSERVATION_FRESH_DEV_TRANSFER：原64模型soft28.47±3.18%→peak72.05±3.14%，无新训练 | 新16DEV已用于分析，不是锁定TEST；原型单端点40/48仍强 |
| 新192观测训练有效 | 配对seed0实测 | OBSERVATION_RESERVED192_PAIR：原ADE-best语义66.67%→95.83%，各384000槽 | 仅常规定位修复，peak末步回落87.5%，相对64数据与曝光均三倍 |
| v参数化解决扩散路线失效 | 未获支持 | DIFFUSION_MULTIGATE_V3_RESULTS：12000步ind/set Unique1.21354/1.09375，Valid40.17%/33.72% | 单训练seed；累计曝光为回归4倍，仍在改善不能称收敛；停止此支线 |
| 错误参考或H24重采样导致观测碰撞 | 当前TRAIN诊断不支持 | OBSERVATION_TRAIN_GEOMETRY_DIAGNOSTIC：181/181原始与H24正参考通过固定检查 | seed0模型前段碰撞仍多；不以此证明未观察几何/整臂安全 |
| 模拟器跨进程同父恢复 | 已实测通过 | observation_multitask_resume_probe_v1：world/RGB-D/camera/language精确一致，首记录hash未变，3/3任务成功 | 只有恢复机制证据；不等于六任务已完成或路线类型可判别 |
| 新96障碍普通定位头有积极结果 | 单种子配对支持 | OBSERVATION_OBSTACLE_RESERVED96_PAIR：新8父/24DEV，soft/peak最佳Tip44.79%/52.08%、Unique.75/1.125 | 常规修复；peak末步Tip43.75%退化；同数据A*87.5%/1.25仍强，非新核心 |
| 缓存头质量对应真实在线Qwen推理 | 实测一致 | 同24请求的路径最大差<=1.431e-6m；全链生成中位61.73/77.20ms | 包含读取/processor/Qwen/RGB-D/head，评分、碰撞验收和执行不在计时内 |
| 观测球形边界提供可分配路线分支 | 被当前TRAIN审计否定 | OBSERVED_DEPARTURE_REGION_AUDIT：24指令×3半径全部J1、无失败或预算耗尽 | 放弃此表示，不外推场景只有一条路线；不实现分配器 |
| 早段碰撞附近有局部可见几何支持 | TRAIN诊断支持 | OBSERVED_LOCAL_SUPPORT_AUDIT：33首次精确接触的10cm邻域均有点，前半弧长覆盖30/33 | 可见点支持不是已知自由空间；680参考前缀点仍有遮挡/画外，更新界限未证明充分 |
| 研究初版核心已成立 | 尚未成立 | REMAINING_EXPERIMENTS.md | 主要缺口是有效观测规划和稳定核心优势；剩余工作不只是扩大规模 |
| 一次局部观测更新改善集合 | 当前配对未支持，停止此实现 | OBSERVATION_REFINEMENT_PAIR：best修1坏0，last修0坏3；局部耗时增加39.1%，同8完整路径状态预算 | 两臂同参数/初始化/采样/曝光；逐坐标更新未饱和；仅常规修复，不将微小best增益称核心 |
| 白名单相机策略下跨进程恢复 | 新独立父实测通过 | observation_multitask_validated_resume_probe_v1：3/3成功，全部初态/输入/首slot hash保持，46.397s | 新144父批次已实际运行但未完成；五任务单父相机等价不外推所有父，push保持开启 |

所有开发集bootstrap区间只描述已选择开发数据上的配对差异，不能作为最终锁定测试的确认性推断。未实测字段在MAIN_RESULTS中留空。
# Latest six-task/direct-VLM boundary, 2026-10-02 13:37 UTC

Sealed prefix24: six original tasks,24 parents/72 successful references,12 TRAIN and12 DEV parents; three source attempts per parent. Actual frozen Qwen96 encodings and ordinary free-endpoint head1500-step training completed from6bc2b8. TRAIN/DEV macro path ADE1.783/17.121cm shows a large generalization gap, not representative task success. Semantic, collision, UniqueValid and execution results remain null. Complete provenance: `observed_multitask_prefix24_v1/artifact_index.json`.

Direct VLM SFT boundary: real2B preflight v1 failed memory, v2 passed exact-prefix masking and4 actual LoRA steps at4.839GB. This establishes a working memory-bounded training interface only; full SFT and independent-versus-whole-set generation have not completed. No new core advantage follows from either result.


## 2026-10-02 16:50 UTC — latest method/data evidence

| Claim | Status | Measured evidence | Boundary |
|---|---|---|---|
| Same physical initial state can support more than K4 route relations | Single-parent feasibility supported | v4:8/9valid,5 distinct lateral sequences,9 exact restores | 3 valid unknowns,1 H24 type-instability rejection; one target/layout, no learned-method claim |
| Response-aware pair selection improves on geometry/DPP for single gate closures | Rejected in current controlled setting | TRAIN768:3806closures,3424solvable; all720evaluable parents tie at100% AnyValid | Reference controls privileged; stop module, never promote weak K4prefix2 comparison |
| Real K2 ordinary small-budget baseline exists | Registered and training, outcome pending | Same3000x64/old static selection,384000 path states | Half K4 path budget, different query count; no causal joint-risk claim |
| Extra SFT exposure restores route quality | Unverified; actual recovery implementation tested |13server CPU tests pass, including exact tiny continuous/continued/resumed loop | Original1500 results remain;6000 GPU outcome not yet present |
