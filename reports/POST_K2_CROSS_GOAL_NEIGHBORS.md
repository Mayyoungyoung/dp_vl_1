# 跨目标路线对应：五项近邻短核对

2026-10-03；接续 `POST_K2_CORE_DECISION.md`，只核对原文与作者来源，未复现、未运行模型或服务器任务。

**判断：当前“同父不同目标的已知同类型路线表示对齐”应先归为常规辅助监督假设，尚不足以作为方法贡献。** 目标与路线形状分离、跨目标拓扑去重、历史候选对应及条件化组合生成均有明确先例。有限检索未发现与本项目完整观测输入和正例不完备设置完全相同的实现，但这不是新颖性证明。

| 最近邻及核验状态 | 原文的具体机制 | 对当前候选的约束 |
|---|---|---|
| Osa & Ikemoto，*Goal-Conditioned Variational Autoencoder Trajectory Primitives with Continuous and Discrete Latent Codes*，[SN Computer Science 1:303，2020](https://link.springer.com/article/10.1007/s42979-020-00324-7)；读 [v2 §III-A、式5–6](https://arxiv.org/pdf/1912.04063v2)。本次未核实作者实现。 | 连续/离散潜变量表示轨迹变化和类型，额外目标坐标条件进入解码器；改变目标和类型码生成新轨迹，另有目标约束投影。 | “目标改变而保留路线类型”本身已有先例。本项目可检验观测条件下的**跨目标已知正对应监督**，但不能将目标/意图分离或对比损失单列为创新。该文给定目标坐标和投影不能直接移作同信息主基线。 |
| de Groot 等，*Topology-Driven Parallel Trajectory Optimization in Dynamic Environments*，[T-RO 2024 / 原文v2](https://arxiv.org/html/2401.06021v2)，§IV-B 式6–7、§IV-D、附录A-A；[作者 guidance_planner](https://github.com/tud-amr/guidance_planner)、[mpc_planner](https://github.com/tud-amr/mpc_planner)。 | `FilterAndSelect` 去掉通向不同目标但拓扑等价的路线；`IdentifyAndPropagate` 在新旧候选间传播同伦类型标识；局部优化加约束以免不同路线重新塌缩。附录明确连接不同目标的末端来比较 H-signature。 | 这是最直接的近邻：跨目标类型对应和有限候选拓扑去重已有算法。当前候选与其区别只能是从不完备正参考学习、观测推断目标及固定生成成本，不能声称首次跨目标对应。作者实现为2D动态环境，不能不经验证把其同伦保证移植到3D机械臂；本项目 row-crossing 类型也不是完整配置空间同伦类。 |
| Song 等，*Don't Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving*， [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Song_Dont_Shake_the_Wheel_Momentum-Aware_Planning_in_End-to-End_Autonomous_Driving_CVPR_2025_paper.html)；读 [§3.1–3.2 式1–5](https://arxiv.org/html/2503.03125v2)；[官方 MomAD](https://github.com/adept-thu/MomAD)。 | 当前多候选与历史路径做坐标对齐，用 Hausdorff 距离选对应，再读取历史查询生成新的整集合。 | 读取旧路径、匹配候选、保留意图后重生成均已有先例。其主要条件变化来自时间/观测，本项目拟干预目标指令并只监督已知正对应；这个区别须赢过相同结构的普通条件补全。文中“Topological”匹配用的是 Hausdorff 距离，不应误写成严格同伦判别。 |
| Brehmer 等，*EDGI: Equivariant Diffusion for Planning with Embodied Agents*，[NeurIPS 2023正式论文](https://papers.neurips.cc/paper_files/paper/2023/hash/c95c049637c5c549c2a08e8d6dcbca4b-Abstract-Conference.html)；读 [§2–4及附录C](https://papers.nips.cc/paper_files/paper/2023/file/c95c049637c5c549c2a08e8d6dcbca4b-Paper-Conference.pdf)。本次未核实作者实现。 | 对空间、时间平移及对象排列实现等变建模，任务通过条件和引导打破对称；导航观察包括目标及障碍状态。 | 在同一障碍场景中只改目标，通常不是把整个场景施加同一可逆群变换；可行类型也可能增减。因此当前候选应称**部分正对应**，不能直接称目标等变性，也不能对全部候选强加一一对应或坐标一致。 |
| Clark & Shkurti，*What Do You Need for Compositional Generalization in Diffusion Planning?*，[arXiv v3，2026-02-09](https://arxiv.org/html/2505.18083v3)，§III–IV，尤§IV-C/E；[作者代码](https://github.com/rvl-lab-utoronto/diffusion-stitching)。仅确认预印本，未核实会议录用；仓库仍使用旧标题。 | Eq-Net 用局部感受野和时间移位等变性促进轨迹组合，比较再规划/数据规模，并用起终点或中间状态 inpainting 测试目标泛化。 | 不应把“旧意图可组成新目标轨迹”当作未被研究的问题。它也说明普通归纳偏置或数据量可能解释收益。本文没有证明本项目的正例对应监督；本项目也不能用其给定状态 inpainting 越过观测输入限制。无需因此重开已经停止的扩散支线。 |

真正尚可证伪的差异很窄：在两臂都取得同样目标/类型正例、同样参数和曝光时，**跨目标的配对关系**是否比同目标类型辅助监督更能避免语义正确且几何有效路线的覆盖丢失。普通目标条件生成、路径 attention、类型分类、Hungarian matching 和一致性/对比损失都不构成这一差异。

下一决定仍等待新正式数据的普通 K4 基线。低成本 TRAIN 审计应先统计“两个目标均有已验同类型正参考、普通生成也已有正确有效候选、但目标改变后丢失该类型”的父场景数。还须按目标对保留 unknown/无对应，不能由共同标签推断坐标前缀可拼接，不能把未采到参考作负例。若主要错误仍是端点或碰撞，或者可迁移类型缺口很小，则无需实现该辅助机制。

若上述条件成立，原卡的同目标配对控制必须保留；它达到相同收益时，应将结果定性为普通类型监督收益并停止跨目标核心主张。旧2条上下文加新4条输出仍计6个完整路径状态，不能与无上下文 joint4 声称同候选/同成本。当前不新增模型模块或训练授权。
