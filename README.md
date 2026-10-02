# Task-layer route-set generation

## V2研究分支（2026-10-02）

当前实施入口是 [研究状态](STATE.md)、[中文证据报告](reports/RESEARCH_V2_ZH.md) 和 [恢复说明](RESUME.md)。代码在 `codex/multiroute-v2`，历史版本保存在 `d0d97eb`。下文旧实验说明继续保留，不代表V2必须固定三路、使用扩散或输入真实终点。

V2已经实测可变1–16种开口序列、集合补全三训练种子、真实Qwen3-VL-2B、同初态RLBench-derived采集、普通观测集合头与真实LoRA更新。核心方法优势尚未成立：max覆盖记忆跨种子不稳定，观测精确定位仍在改进。失败及修正记录见 [研究日志](RESEARCH_LOG.md)，完整结果见 [MAIN_RESULTS](reports/MAIN_RESULTS.csv)。

观测层只允许RGB/RGB-D、语言、相机和当前状态；真实目标/障碍几何仅作标签与评价。`observation_eval_v2`保留无成功示范的开发指令评价语义，缺参考的ADE为null，原23条有参考子集结果另存历史。不同训练曝光、线程、计时范围与重复成本行在结果表中标注，不能直接按表行相加GPU小时。

服务器从明确commit导出运行，原目录和历史环境保留。当前仍按GPU1、35%显存和CPU4线程总范围执行，禁止更改共享环境或占用其他GPU。训练权重/数据不入普通Git，保留实际文件、hash、来源和断点恢复状态。

## 历史基础版说明

这是针对“任务与场景 → 一组可选任务路径”的可运行研究基础版。执行器、逆运动学、关节动作与机器人控制不在本项目的实现范围。

目标是同一条件下，在相同数据和候选预算 K 内，提高有效且不同的路线数量。模型不把“轨迹抖动更大”视为贡献，也不通过先生成大量候选再挑 K 条获得隐藏优势。

实际三训练种子结果、置信度质量与边界见 [实验报告](reports/experiment.md)。

## 推荐结构

```text
图像 + 任务文本 ── 冻结的视觉语言编码器 ── 场景/任务特征
                                                │
几何观测 + 起终点/交互锚点 ────────────────────────┤
                                                ↓
                            共享条件编码 + K 个路线候选
                                                ↓
                     候选间通信 + 连续路径解码（扩散可选）
                                                ↓
                       K × H × XYZ + 逐路线校准置信度
                                                ↓
                           固定几何检查器 → 下层选路接口
```

VLM 应提供任务目标和必须满足的交互约束，避免先确定一条完整中间路线。路径头负责保留可替代的路线；同一候选的起点、终点和必要事件可以共享。候选是替代方案，不会同时执行，因此不应施加“候选之间不能相交”的约束。

直接 XYZ 是第一版表示。双视角二维表示的投影、三角化工具已经提供；后续可以替换生成头，保持条件、示范、候选数、检查器一致。投影一个三维预测再重建不是训练双视角生成模型，不能作为表示优劣实验。

## 已实现方法

- `independent`：按路线模式平衡训练的普通扩散，每组独立生成 K 条路线。
- `set_diffusion`：联合去噪 K 条路线，每层通过候选注意力交换整条路线摘要；没有候选编号编码，具有排列等变性。基础配置使用标准噪声预测损失。
- `regressor`：K 个学习查询一次生成完整路线集合，用穷举匹配（K=3 时等价于最优集合分配）训练。候选槽位没有预设左/右/上语义。
- `Critic`：独立预测每条路线通过任务层几何检查器的概率；TRAIN 训练，VAL 校准，TEST/OOD 报告 Brier/ECE。它不是机器人执行成功率。

第一版路径编码是相对于起终点直线的内部路点残差，两端固定。一个路线展平为一个 token，经条件 MLP 和候选注意力处理。这是便于快速验证的网络，没有把它包装为大型时序 Transformer。可选低噪声集合匹配辅助项通过 `--cover-weight` 开启；默认实验为 0。

扩散采用 100 个训练噪声级别和 40 次 DDIM 前向；回归只需一次前向。相同 K 不代表相同计算量，应同时看时延。

## 数据与边界

已实际读取 3DWay 公开元数据前缀，并下载官方双视角图像与标定示例。公开数据适配与几何烟测见 [sources.md](reports/sources.md) 和 [3dway_geometry_smoke.json](reports/3dway_geometry_smoke.json)。未把不对应的示例标定套用到元数据轨迹上。

覆盖实验使用程序化小数据：768 TRAIN、128 VAL、128 TEST、128 OOD 场景，每场景 3 种通道 × 4 条路线变体，共 13,824 条参考路线。所有路线和文本随完整场景划分。落地箱体阻挡直线，有负 y、正 y 和上方三种**参考路线模式**；这些不是宣称的三维同伦类。OOD 使用更宽、更高、沿 x 偏移的箱体。

CLIP ViT-B/32 是真实冻结的预训练图文编码器，固定 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。每个场景缓存双视角平均图像特征与文本特征，共 1024 维，拼接 12 维真实几何与起终点。**当前几何与任务锚点是真值输入，CLIP 特征是附加条件；尚未学习自然语言目标定位，也未证明 CLIP 对指标或语义泛化的贡献。**

所有场景只有一个障碍、相同技能结构、固定 3 个参考模式。基础结果只验证路线集合机制；不能代替真实操作任务、变数量可行模式、未知物体或机器人执行实验。

## 指标

- `unique_valid` / UniqueValid@3：三条原始输出中，通过检查且模式不同的路线数，取值 0–3。
- `valid_rate`：候选有效率。检查完整折线段与扩张 AABB 的碰撞，端点、有限值、工作空间以及长度比；不只检查离散路点。
- `coverage`：覆盖三个参考模式的比例。
- `success`：至少一条通过该检查器的比例，不是实际机器人成功率。
- `mode_exclusion_survival`：预先规定分别禁止一种模式，已有集合是否还有符合规则的候选。没有插入实体障碍，不代表物理通道封闭实验。

扩散测试的 4 个随机重复各自只有 K=3 条，分别评估后平均；没有合并成 12 条。主要方法额外运行训练 seed 0/1/2，汇总均值与跨种子标准差。`comparison.png` 和场景 bootstrap 区间展示 seed 0；跨种子结果应看 `multiseed_summary.json`。

同一 seed 下各方法具有相同目标路线曝光量。独立扩散禁用候选注意力，其不活跃参数不计入 `active_parameters`；还应查看容量更接近的加宽独立模型。共享 CLIP 缓存、数据文件和 checkpoint 都有 SHA 检查。

## 复现

服务器项目位于 `/home/wzy/dpvlm/route_set_v1`，Python 环境仅位于该项目 `.venv`。没有 sudo、apt、删除他人文件、修改共享环境或中断他人进程。可在项目目录运行：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests -q
.venv/bin/python -m routeset.data --output data/routes.npz
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/cache_clip_features.py \
  --model data/clip-vit-base-patch32 --data data/routes.npz \
  --output data/clip_features.npz
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_baselines.py \
  --data data/routes.npz --features data/clip_features.npz \
  --output runs/main --steps 4000 --seeds 0 1 2
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_confidence.py \
  --data data/routes.npz --features data/clip_features.npz \
  --prediction-dirs runs/main/independent_seed0/evaluation \
    runs/main/set_diffusion_seed0/evaluation runs/main/regressor_seed0/evaluation \
  --output runs/confidence --steps 2000
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m routeset.infer \
  --data data/routes.npz --features data/clip_features.npz \
  --checkpoint runs/main/regressor_seed0/best.pt \
  --critic runs/confidence/calibrated_critic.pt \
  --output reports/demo_candidates.json
.venv/bin/python scripts/summarize_seeds.py --runs runs/main
.venv/bin/python scripts/make_report.py --runs runs/main --output reports
```

不要在复用 checkpoint 时重新生成数据文件；数据和特征缓存的 SHA 必须一致。新安装时先创建独立环境，安装 CUDA 版 `torch==2.4.1+cu121`，再安装 `requirements.txt`。现有服务器环境已按这一方式准备。

`scripts/inspect_3dway_data.py` 可限量检查公开配对二维标签；`scripts/check_3dway_example.py` 用官方相机标定验证纯几何重建。公开几何测试中的点是合成投影点，不是模型预测。

Critic 在 seed 0 三种生成器的验证预测上共用 Platt 校准，拟合样本自然形成 4:4:1 的方法权重。后半 VAL 只对 Critic/校准器留出，曾用于路径模型选 checkpoint；最终 TEST/OOD 没有用于选择或校准。每个方法单独报告 Brier/ECE，分布外校准仍有偏差。Critic 只适用于两端已硬约束正确的生成路线。生成时延是 batch 32 的均摊值，使用已缓存条件，不包含图文编码与模型加载。

## 下一轮最有价值的工作

1. 固定可重置 RLBench 场景和任务，收集/规划同场景多路线并验证；让可行模式数变化，加入多个障碍、窄通道和必经交互锚点。
2. 把真实 VLM 的对象/交互锚点 token 接入同一个路径集合头；分别测目标识别、语言改写、未见布局和路线覆盖，检验预训练特征的实际作用。
3. 加入 VLM 独立采样、一次输出多条、Particle Guidance、多起点传统规划和容量匹配对照。评估 K=1/2/3/4/8 与固定时延预算。
4. 比较直接 XYZ 与真正学习的成对二维输出。二维版要记录原始重投影误差、失败率和校正幅度；不能用后处理掩盖模型不一致。
5. 在固定执行器上检验筛选后的一条路线。所有方法共享检查器；置信度按实际控制器成功标签重新校准。

如果一次前向的集合回归已经可靠覆盖模式，优先使用它。扩散可作为每个路线查询内部的连续变体生成头；是否需要扩散应由有效覆盖、质量和成本决定。
