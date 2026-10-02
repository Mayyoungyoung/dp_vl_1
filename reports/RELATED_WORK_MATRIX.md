# 任务层路线集合：近邻工作核验矩阵

核验日期：2026-10-02（Asia/Shanghai）。范围为原论文、会议论文集、作者项目页及由原论文/项目页指向的代码。**本文件是文献与源码核验，不是复现结果表；下列外部方法均未在本项目完成复现。** 代码可访问、检查过推理入口、下载过示例和完成训练复现是不同状态。

首轮决定：保持普通多查询集合回归为强基线。优先检验“已有路线是否确实覆盖了一个有效方案”这一补全机制；不能把 Qwen、深度编码、三维路径、查询、匹配、候选 attention、相关采样或集合奖励本身写成创新。两张可证伪方法卡见 [METHOD_CANDIDATES.md](METHOD_CANDIDATES.md)。这是基于已读材料的研究判断，不是穷尽检索后证明的新颖性结论。

## 高层路径与机器人系统

| 工作与一手来源 | 核验到的日期/发表状态 | 输入 → 输出；候选机制 | 监督与预算 | 本项目的相同点、差异与必要对照 |
|---|---|---|---|---|
| [HAMSTER 原论文](https://arxiv.org/abs/2502.05485)、[ICLR 论文集](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3bfee3bc6639c36e6e7b058db909f760-Abstract-Conference.html)、[官方代码入口](https://github.com/liyi14/HAMSTER_beta) | arXiv 首发 2025-02-08；ICLR 2025 已由论文集确认 | RGB+指令 → 一条粗粒度二维末端路线及夹爪事件，交给三维低层策略。多点是一条路线的路点，不是多条替代路线 | 混合域外路径/点监督，随后训练路径条件低层策略；没有本项目的固定 K 有效独特路线目标 | 高层/低层分工是直接近邻。需要共享固定执行器的系统级对照；预训练数据和主干不同，不能称同数据实验。已读 `server.py`：单次生成可按 temperature 采样；仓库是 beta 服务入口，不能据此声称完整训练复现链已齐备 |
| [3D HAMSTER 原论文](https://arxiv.org/html/2606.31329v1)、[官方代码](https://github.com/DAVIAN-Robotics/3D_HAMSTER)、[官方权重](https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER) | arXiv 2026-06-30；作者仓库/模型卡标 IROS 2026，本次未取得独立会议论文集记录 | 标定 RGB-D+语言 → 自回归 `(u,v,d)` 路点/事件，再转世界坐标；Qwen3-VL-8B+LingBot-Depth，融合后以三维点云条件策略执行 | 深度对齐和任务微调两阶段，密集深度重建；论文有大规模混合监督。不是固定 K 集合覆盖训练 | “Qwen+深度+3D”已有直接竞争者。已读 `hamster3d/inference/api.py`，默认 `do_sample=False` 生成一条路径；当前代码树可见推理/模型组件，未见完整训练入口。可先复用其公开权重做系统级单路线/重复采样检查，不能把本项目 2B 同数据训练与其大规模 8B 训练直接归因于机制 |
| [3DWay 原论文](https://arxiv.org/html/2609.08224v1)、[项目页](https://ziqin-h.github.io/3DWay/)、[官方代码](https://github.com/ziqin-h/3DWay) | arXiv 首发 2026-09-08；作者项目页和代码标 ECCV 2026，本次未取得独立 ECVA 论文集条目 | 标定多视角 RGB+语言 → 对应二维路点 → 几何三角化的单条三维轨迹 | 多视角一致监督；路径执行或引导 VLA。几何重建与训练资源须单列 | 本项目主线是有限预算的替代路线集合，不以投影/三角化作为贡献。已读 `policy.py`，`NVILAPolicy` 调用 NVILA 服务并解析两视角路线；不应误写成 Qwen 实现。现有项目的 3DWay 元数据/标定烟测不是该系统复现 |
| [Track2Act 原论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09800.pdf)、[作者项目页](https://homangab.github.io/track2act/)、[官方代码](https://github.com/homangab/Track-2-Act) | arXiv 首发 2024-05-02；ECCV 2024 已由 ECVA 论文确认 | 当前/目标图像 → 场景点未来 tracks → 刚体变换与末端计划；残差策略闭环修正 | 网络视频学习 track 预测，少量机体数据训练残差策略；代码有 `train_track_pred.py` 和 `inference_track_pred.py` | 多个被跟踪点不是多个候选末端路径。“diverse manipulation”不能解释成固定 K 的路线类型覆盖。使用目标图像时信息条件与语言目标版本不同，需单列 |
| [Value-Guided Multi-Path Reflection 原论文](https://arxiv.org/html/2602.19372v1)、[arXiv 元数据](https://arxiv.org/abs/2602.19372) | 首发 2026-02-22；作者 arXiv comments 标 ICRA 2026；未独立核实论文集/官方代码 | 当前/目标图像 → 高层动作；动态模型+beam search 产生多种未来，以价值反馈和解码聚合改进动作，支持置信早退 | 交互模仿与专家状态监督，推理有想象轨迹、价值估计和反思成本 | “读取其他路径并改进当前预测”也已有机器人近邻。它主要选择高层动作和未来后果；本项目应验证空间路线集合的有效覆盖与重复幂等性，而非仅主张多路径互相交流 |

## 集合预测、多假设与多样化

| 工作与一手来源 | 状态 | 输入 → 输出；核心机制 | 监督/预算与本项目应做的比较 |
|---|---|---|---|
| [DETR](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/832_ECCV_2020_paper.php)、[代码](https://github.com/facebookresearch/detr) | ECCV 2020；arXiv 2005.12872 | 图像 → 对象集合；学习查询+二分匹配 | 完整对象标注下的空槽有意义。路径参考通常不穷尽，不能照搬 no-object 将未示范解判负。当前 `SetRegressor` 属于成熟集合预测范式的路线适配，必须保留为基线 |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | ICML 2019，PMLR 已确认 | 输入集合 → permutation-invariant 表示/输出；attention 建模元素间相互作用 | 普通集合 attention/池化不构成新贡献。应以同训练、同路径上下文的普通 cross-attention 补全器消融显式覆盖记忆 |
| [MTR](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)、[论文](https://arxiv.org/abs/2209.13508)、[代码](https://github.com/sshaoshuai/MTR) | NeurIPS 2022，论文集确认；arXiv v2 2023-03-18 | 交通参与者历史/地图 → 多模态未来；意图查询对、局部地图提取和运动细化 | 单个观测未来的预测与同一机器人任务的多个可行解不同。学习意图查询、粗到细、轨迹条件地图读取均非本项目独创；应考虑同条件意图/连续路径分解基线 |
| [Diverse Trajectory Forecasting with DPP](https://arxiv.org/abs/1907.04967)、[作者机构论文](https://www.ri.cmu.edu/app/uploads/2020/04/Yuan_DPP_forecast.pdf)、[作者出版列表](https://ye-yuan.com/) | ICLR 2020；作者列表和论文首页确认 | 场景上下文 → 一组相关潜变量，经既有生成器获得多条未来；DPP 多样性损失训练采样函数 | 质量和多样性联合建模是强近邻。作者论文入口本次未核验到官方代码，不把第三方 DPP 项目当作者实现。可做 DPP-inspired 适配，但须如此命名并验证公式/数值，不称完整复现 |
| [LED](https://openaccess.thecvf.com/content/CVPR2023/html/Mao_Leapfrog_Diffusion_Model_for_Stochastic_Trajectory_Prediction_CVPR_2023_paper.html)、[代码](https://github.com/MediaBrain-SJTU/LED) | CVPR 2023，CVF 确认 | 历史轨迹/社会条件 → 相关轨迹初值+少步扩散；可训练 leapfrog initializer 预测均值、尺度和相关样本 | 已读 `model_led_initializer.py`：默认 `k_pred=20`，并行生成 K 条样本偏移。应将同类 initializer 适配到相同条件和 K，并计入初始化器训练及少步细化；相关潜变量/少步扩散本身不新 |
| [Particle Guidance](https://proceedings.iclr.cc/paper_files/paper/2024/hash/612a7948f3294a02a63d970566ca8536-Abstract-Conference.html)、[论文](https://arxiv.org/abs/2310.13102)、[代码](https://github.com/gcorso/particle-guidance) | ICLR 2024，论文集确认；arXiv v2 2023-11-24 | 同条件扩散粒子通过随时间变化的联合势相互作用；以非独立采样提高多样性 | 本次检查了 `generate_particle.py`，有势强度/步数和 DINO 等表示配置，仓库还含 synthetic notebook。路线适配必须明确核/势、梯度和采样步，调参只用 DEV；仅加路径欧氏排斥是适配基线，不等于复现所有原文结果 |
| [EDDY v2](https://arxiv.org/html/2605.06553v2)、[作者项目页](https://galvinograd.github.io/eddy/) | 首发 2026-05-07；v2 2026-09-28；本次只确认预印本，项目页仍写 Code (soon) | 负相关初始化+经 Stein 算子的反对称场耦合；旨在避免普通排斥改变单粒子分布 | v2 对独立粒子下无散度耦合与初始化的作用有明确区分。不能只实现排斥就标 EDDY；理论的精确 score/连续动力学条件不能自动转移到离散近似路线模型。暂无本项目复现，不作为已跑强基线 |
| [DPP 基础与条件化](https://arxiv.org/abs/1207.6083) | 2012 专著/预印本入口 | 有限元素集上的质量—相似度核，支持子集建模及条件操作 | 以已有集合为条件的多样化不天然新颖。若拟议覆盖补全只是 conditional-DPP 或贪心子模覆盖的神经近似，应主动降级新颖性表述并加入相应机制对照 |

## VLM 多候选、集合训练与传统规划

| 工作/对照 | 一手核验与机制 | 对本项目的约束与可复现状态 |
|---|---|---|
| 同 Qwen 独立采样 / 整集合 SFT / 路线补全 SFT | 这是本项目应构造的三种**实验配置**，不是在本次检索中核验到的一篇同名论文。独立生成 K 个序列；一次序列含 K 条路线；给部分路径后生成余项。高层单路线 SFT 的直接来源见 HAMSTER/3D HAMSTER | 固定相同图像、深度表示、指令、基座 revision、训练路线曝光和 token 预算。随机排列集合序列，避免固定左/右/上顺序成为捷径。重复、格式错误都计预算；未实施不能填结果。若坐标回归路线更强，整集合 SFT 仍是对照 |
| [Diverse Beam Search](https://arxiv.org/abs/1610.02424)、[作者代码](https://github.com/Cloud-CV/diverse-beam-search) | 2016 预印本入口；分组 beam 以多样性目标解码多个序列。此次未核实正式会议录用，不沿用第三方会议标签 | “一次解码多序列”不是新颖性。扩展/剪枝搜索成本与已完成路线数分别报告；文字差异不是通道/任务有效性差异。本项目未复现 |
| [DivPO](https://arxiv.org/abs/2501.18101)、[作者出版列表](https://angie-chen55.github.io/) | 2025 预印本已核验；从响应池按质量和稀有程度构造偏好对。网上有相互矛盾的会议标签，本次未确认正式发表/官方实现 | 可用于具有序列概率的 Qwen 路线文本分支。样本池、评审成本和训练候选曝光必须记录；确定性回归坐标头不能直接套 DPO。未复现 |
| [SetPO](https://proceedings.mlr.press/v306/li26jk.html)、[原论文](https://arxiv.org/abs/2602.01062)、[作者代码](https://github.com/chenyili0818/SetPO) | ICML 2026，PMLR 306:72496–72522 已确认；kernel 集合效用的 leave-one-out 多样性边际作为策略优势修正 | 集合边际奖励/稀有路线奖励已存在明确近邻；移植到空间路径须同时约束真实任务质量。是语言推理 policy optimization，不等于普通回归器的损失。未复现 |
| [Yen K 最短无环路径](https://pubsonline.informs.org/doi/10.1287/mnsc.17.11.712) | Management Science 1971，出版社确认；已知图、边代价、起终点 → 按长度排列 K 条无环图路径 | 几何受控层可作传统强基线；几条图路径可能属于同一路线类型。生成后再丢弃近重复路径也计候选/搜索预算，不能把搜索大池只报 K。观测层须共享预测目标与观测地图 |
| [MPNet 扩展原文](https://arxiv.org/html/1907.06013)、[作者实现](https://github.com/ahq1993/MPNet)、[实际读取的 replan_path 源码](https://raw.githubusercontent.com/ahq1993/MPNet/master/MPNet/neuralplanner.py) | 2019预印本；2020-06-27 v3页首注明T-RO接收；另有ICRA2019版本。原文算法4及源码对可连接段保留，对不能连接的相邻节点执行神经双向局部重规划；原文还讨论经典规划fallback | “保留有效段+神经局部修复”已存在。当前 constraint_update 仅检验相同旧集合、同新增完整路线预算下预测复制门控是否胜普通条件补全；不能把局部编辑概念称为新颖。已读作者源码，未安装/运行/复现 |
| [MARS 出版社版本原文（作者机构存档）](https://re.public.polimi.it/retrieve/4bb6d46f-51b9-48d7-89b7-c5be61421687/11311-1255951_Faroni.pdf)、[机构记录](https://re.public.polimi.it/handle/11311/1255951)、[原文[41]代码链接](https://github.com/JRL-CARI-CNR-UNIBS/replanning_strategies) | IEEE Access 11:4105–4116，2023-01-10发表，DOI10.1109/ACCESS.2023.3235652；多条预计算路线构成有向图，障碍使当前路失效时连接到其他可用子路径，并用informed sampling、图复用及anytime改进 | 多路线复用与变化后替换已有明确传统强近邻。论文代码链接现重定向作者OpenMORE；已核验README/子模块来源，未执行MARS核心实现。不能把本项目受控神经筛选或解析开口规划器写成对MARS的复现/胜出 |
| [同伦约束搜索](https://ojs.aaai.org/index.php/AAAI/article/view/7735)、[作者机构记录](https://publications.ri.cmu.edu/search-based-path-planning-with-homotopy-class-constraints)、[3D 后续作者论文](https://www.cs.cmu.edu/afs/cs/Web/People/maxim/files/planwithhomotopy_aaai12_invitedpaper.pdf) | 平面方法 AAAI 2010；3D 表示 RSS 2011/AAAI 2012 后续。AAAI 网站 3D 条目的重新上网日期不应误写成论文年份 | 已知障碍下将路径类别纳入搜索状态，排除已找到类别后继续找有效路径；这是显式类别记忆的强概念近邻。普通三维箱体左右绕行不应未经证明叫不同同伦类。项目当前未复现 |

## 源码检查记录及复现边界

以下为 2026-10-02 经 GitHub API 读取的默认分支 **tree SHA**（不是项目训练 commit，也不是本项目实验复现证据）。只进行了只读检查，未安装或执行外部代码。

| 仓库 | tree SHA | 实际检查 |
|---|---|---|
| DAVIAN-Robotics/3D_HAMSTER | `97216a8493f46301bf569d398462b8bb21c458c5` | 文件树、推理 API；默认确定性单路线；模型卡 |
| liyi14/HAMSTER_beta | `526a37f59f97c445005fcdf28f2cfb81ea742e4b` | 文件树、`server.py` 的生成调用；依赖外部 VILA |
| homangab/Track-2-Act | `3e4ccd4ffb9aff39daf2c00d4732545bcd56f9cf` | 文件树/训练和推理入口存在性；未运行 |
| ziqin-h/3DWay | `07fa390520d99fdd9c7bd94aa8e4f03539e6f50f` | 文件树、`NVILAPolicy`、二维路径解析/三角化调用 |
| MediaBrain-SJTU/LED | `aae048a85292a24c2218de5fe9b4d20d3542bf04` | initializer 全文和 trainer 入口；未适配/训练 |
| gcorso/particle-guidance | `ce7de745191168c10b2c125b78d94ad8de119488` | 文件树、生成脚本、synthetic notebook 存在性；未执行 |

真正启动外部基线前还需锁定 commit、license、依赖、权重 revision/hash 和本项目适配差异。未获得完整训练链的系统，可以公开权重推理对照起步，但不得称同数据从头训练复现。

约束更新补充核验（2026-10-02）：MPNet 默认分支 tree SHA `f38623bff34626b649d00f3f6535c1954852907e`，已读 `MPNet/neuralplanner.py` 中 `replan_path`、`steerTo` 与保留分支；MARS已读原文方法及参考[41]，其代码地址重定向OpenMORE，页面所列 `replanners_lib` 子模块为 `074c7e875a0a800614cbc9244dbff1a7101c498c`。后者仅核验来源和框架说明，未声称读取或执行全部核心实现。两者均无本项目训练/规划复现结果。

## 对当前创新叙述的直接影响

1. HAMSTER/3D HAMSTER/3DWay 已覆盖高层 VLM 路线、几何输入和三维指导；这些是研究接口，不是本项目的新机制。
2. DETR/MTR/LED/Particle Guidance/SetPO 已覆盖集合匹配、意图查询、相关候选、相互作用和集合多样性目标。仅组合这些模块不足以建立方法贡献。
3. 可以检验的窄问题是：在严格候选和更新时间预算下，补全器能否区分“已有路线条数”和“当前任务下实际已覆盖的有效方案”，并在重复、无效草稿、条件变化时稳定分配剩余预算。
4. 首轮必须同时给普通补全 attention、无有效性门控、非幂等聚合与从空集合生成结果。若普通基线已经解决，则保留强基线并换机制，不能靠换名保留创新结论。
