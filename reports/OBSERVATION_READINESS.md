# 观测任务链路核验（2026-10-02）

## 已核验的环境

服务器真实根目录为 `/home/wzy/dpvlm/route_set_v1`。历史环境 `.venv` 是 Python 3.8.10、PyTorch 2.4.1+cu121、Transformers 4.44.2；原数据是固定三路合成数据、冻结 CLIP 特征和两张 3DWay 示例图。原环境没有 Qwen、RLBench 或 PyRep。此前结果不能作为 Qwen 或机器人实验。

本轮独立准备 `.venv-qwen`（Python 3.11.13、PyTorch 2.4.1+cu121、Transformers 4.57.1）和 `.venv-sim`；不升级原 `.venv`。固定公开来源：

| 组件 | 固定版本 / revision | 来源与核验 |
|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | `89644892e4d85e24eaac8bacfd4f463576704203` | [官方模型](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)，官方API逐文件blob/LFS哈希在 `scripts/observation_model_manifest.json` |
| Qwen weights | 2,127,532,032 BF16 parameters / 4,255,140,312 bytes | 官方权重 SHA256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`；下载完成后必须核验，未完成时不称已加载 |
| RLBench | `02720bba4c73fe02eb75df946b8791b806028a9d` | [官方代码](https://github.com/stepjam/RLBench/tree/02720bba4c73fe02eb75df946b8791b806028a9d) |
| PyRep | `8f420be8064b1970aae18a9cfbc978dfb15747ef` | [官方代码](https://github.com/stepjam/PyRep/tree/8f420be8064b1970aae18a9cfbc978dfb15747ef) |
| CoppeliaSim | EDU 4.1.0 Ubuntu20.04 | [RLBench官方安装所指定的二进制](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)，仅解包在项目 `.observation-deps/CoppeliaSim` |
| Xvfb | 2:1.20.13-1ubuntu1~20.04.20 | 用系统已配置 Ubuntu 官方包源下载deb，项目内解包；不安装系统包、不改共享X配置 |

服务器 Hugging Face 直连超时，下载使用 hf-mirror 传输；以官方模型 API 的 git blob/LFS SHA256 判定字节是否一致。Python 和固定源码从官方 GitHub 下载到本地后 SCP 到服务器，绕开服务器极慢的 GitHub 下载。Qwen下载可续传。独立环境的已安装全量依赖在 `scripts/observation_*_freeze.txt` 记录。

实际下载的Python/模拟器/源码压缩包SHA256与固定URL另见 `scripts/observation_dependency_manifest.json`。PyRep初次构建因standalone Python默认寻找未安装的clang失败；使用系统现有 `CC=gcc CXX=g++` 在独立环境构建成功。未安装或修改共享编译器。此轮项目新增主要占用约 `.venv-qwen` 5.4GiB、`.venv-sim` 278MiB、模拟器与源码/安装包1.6GiB、Qwen权重4.0GiB，低于授权的初步30GB。

## 四个任务的源码适配审计

| 原任务 | 可核验内容 | 多路线采集范围 | 当前限制 |
|---|---|---|---|
| reach_target | 颜色指定目标、两个不同颜色干扰球、末端被成功传感器检测 | 接近目标前的自由运动 | 原任务没有离散通道；不同弯曲不能自动计为不同路线类型 |
| pick_and_lift | 颜色指定方块、抓取和抬升事件、干扰物 | 抓取前的自由运动 | 保留原接触/抓取轨迹，不把插入引导点当推理条件 |
| push_button | 按钮基座颜色、按钮关节位移条件 | 下压前的接近运动 | 单个按钮，原指令变化主要是改写，不足以证实目标选择 |
| take_lid_off_saucepan | 抓盖与抬起传感器条件 | 抓盖前的自由运动 | 单一目标；未来需增加同图多目标/约束配对 |

试采明确称为 **RLBench-derived free-approach pilot**。原任务成功条件保持；新加自由运动方案不冒称原 benchmark 的多解标注。不因轨迹欧氏差异给唯一类型数；`unique_valid_route_types=null`。连续碰撞认证尚未完成，仅显式检查新加自由段的模拟步碰撞。

官方 `TaskEnvironment._get_live_demos` 在每条示范前调用 `reset()`，因此连续 `get_demos(K)` 不是同初态多解。实测发现，仅重设 NumPy RNG 也不足够；单独恢复 configuration tree 仍未恢复物理引擎状态。稳定实现采用 `stop → 恢复任务/机械臂/夹爪树、颜色、关节与控制目标 → start → 恢复位置控制标志 → 固定10步稳定化`，将这个可重复稳定状态作为父场景基准。每次仍逐对象核验姿态、关节、速度、颜色与机器人状态，并检查RGB；超过容差立即拒收。已成功的单父场景9次路线尝试中，完整状态最大绝对差与RGB逐像素最大差均为 **0**。所有失败、总尝试数与近重复率保留。新增任务要重新核验，不能将一次成功推广到所有任务。

四任务切换时还需在导入下一个模型前停止模拟；否则 CoppeliaSim 在之后 `stop()` 时会移除运行中导入的任务模型。此失败已记录并修复。没有为通过审计而放宽恢复容差。

## 输入与缓存契约

`scripts/observation_cache_qwen.py` 真正调用 `Qwen3VLForConditionalGeneration` 的底层模型，使用最后隐藏状态（不生成或输入真实路线答案token）。输入JSONL仅允许 `id,parent_id,split,image,instruction` 五个字段，拒绝多余标签字段，检查父场景跨划分泄漏。输出缓存包含冻结Qwen最后token与mean hidden，记录图像hash、模型/processor revision、token数与单请求耗时。

深度、相机、当前夹爪状态单独存储用于观测几何/状态头；不输入 `task_low_dim_state`、真实目标坐标、完整障碍几何或未来路线。路线监督文件和采集引导点审计文件与输入JSONL分离。LoRA/骨干更新后禁止复用冻结缓存。Qwen真实前向验证本身仍不等于观测路线训练或语义泛化证据。

## 运行与恢复

```bash
# 可恢复公开权重下载：只重用已通过官方哈希的文件
cd /home/wzy/dpvlm/route_set_v1
bash scripts/observation_download_job.sh

# 四任务，各1父场景，3次独立恢复后的路线尝试；项目Xvfb与软件渲染
bash scripts/observation_rlbench_job.sh observation_rlbench_pilot_20261002 --parents-per-task 1 --attempts 3

# 需在GPU协调后执行；单进程显存上限为35%，只用物理GPU1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv-qwen/bin/python scripts/observation_cache_qwen.py \
  --model data/qwen3-vl-2b-instruct-89644892 \
  --manifest data/observation_rlbench_pilot_20261002/observations.jsonl \
  --output data/observation_rlbench_pilot_20261002/qwen_cache --device cuda
```

每个作业在 `runs/<run_id>` 保存 PID、status、log、开始/结束时间与退出码。稳定版wrapper先将执行脚本复制到 `run/source`、记录SHA256，再从该副本执行。第一批成功的运行在启动后补存了仍未改动的实际源码，并用 `source_captured_after_start_at` 明示该时间；较早失败修复轮只有日志、traceback和工具修改历史，没有启动时源码hash，不能事后声称某个新commit就是当时运行版本。Xvfb是本作业子进程，退出时只清理自己的Xvfb。后台下载/采集并不代表会话结束后继续自主分析。

## 当前实测状态

### 已完成的真实Qwen前向

`observation_qwen_forward_20261002` 已退出0，使用真正的官方Qwen3-VL-2B-Instruct BF16权重和冻结隐藏状态；模型与processor所有文件官方hash均通过。输入为同一张真实模拟RGB的三个颜色目标指令，三者图像SHA256一致。

| 检查 | 实测 |
|---|---:|
| 真实模型参数 | 2,127,532,032 |
| 可训练骨干参数 | 0（本步仅冻结编码，未LoRA） |
| mean / last hidden维数 | 各2048 |
| 输入token数 | 84 / 请求 |
| 模型加载时间 | 12.000秒 |
| 首次冷前向请求（含该次预处理） | 2.592秒 |
| 后两次单请求编码 | 0.0555 / 0.0507秒 |
| 峰值PyTorch allocated显存 | 4,279,397,376 bytes，约3.99GiB |
| 同图不同语言mean hidden L2差 | 0.444 / 0.404 |
| 同图不同语言last hidden L2差 | 6.023 / 6.550 |

上述时延只覆盖观测编码，不包括路线头、候选更新/检查/评分，不是完整方法端到端时延。特征变化不等于目标正确率。此步骤证实真实Qwen加载、输入隔离和编码链路；本报告不借此前向声称完成LoRA、机器人路径训练或泛化。

随后 `observation_qwen_reach32_cache_20261002` 用 commit `b684b3e620012361bf309709669d6ed8eac0d5bd` 中缓存脚本的冻结副本完成正式96条指令，wrapper退出0。96/96缓存通过id、图像hash、输出hash、有限数值和维度复核；mean/last各2048维。文件页已热时加载2.507秒，总计7.687秒；单请求编码中位数46.51ms、P95 48.55ms、该轮首个请求471.72ms，峰值allocated显存4,279,413,760 bytes。同图变指令的last hidden最小L2差3.915，仅说明语言确实进入编码。正式输入manifest SHA256为 `31fb89f7a68df597d059f8c0a5be5367395ed58873766cd8b87c3000243a0bb3`；96个特征文件hash索引在 `reports/observation_reach32_audit/qwen_cache_hashes.json`。

### 真实同图三目标小批

`observation_derived_reach_pilot_20261002d`：1个父场景、同图3条不同颜色目标指令、9/9路线采集成功；每次完整状态/RGB恢复差为0。9条保存路线均有限且通过3cm终点阈值复核，平均终点误差0.706mm、最大1.181mm，0条1cm近重复。真实机器人运动由固定PyRep路径规划与模拟执行产生；此处是采集器验证，**不是学习模型性能**。没有离散通道标注，唯一有效路线类型数仍为null。

`observation_derived_reach32_20261002` 已完成：32独立父场景和32唯一初图，24 TRAIN / 8 DEV_MODEL，每父3条不同颜色目标指令，共96条。288总尝试中，275条采集成功，13条因 `ConfigurationPathError: Could not create path.` 失败；全部失败保留。全部288次恢复的完整状态与RGB差均为0；275条保存路线全部有限并通过3cm终点阈值独立复核，平均终点误差 **0.677mm**，最大 **10.844mm**。1cm近重复0条；这不是拓扑路线数。实际采集耗时707.112秒。

`observations.jsonl`（严格5字段）与 `supervision.jsonl` 分离；当前观测npz含深度、相机和夹爪状态，监督npz才含未来路径，目标坐标和目标index仅在supervision用于训练监督/评价。父场景划分同时约束图像、语言和所有路线变体。参考集不完整，未匹配路线不得当不存在。审计结果与全数据文件hash在 `reports/observation_reach32_audit/`；hash清单自身SHA256为 `b6cb09f782d8bd8d3755ec33671cf6399bf7d5c2a57c7a47b52e6b3d9c93e81b`。首父场景从不同进程、相同种子独立重建的RGB SHA256一致。

每指令成功参考条数分布：88条指令有3条参考、5条有2条、1条有1条、2条没有成功参考。TRAIN保存208条路线、DEV_MODEL保存67条。没有参考的2条不是无解标签；它们不能参与参考路径loss/ADE，但必须保留在有独立目标标签的语义评价中。96条观测指令仍全部保留并缓存Qwen，避免把规划器采集失败误写成语义目标不存在。32场景初图montage已检查，见 `reports/observation_32scene_montage.png`；实例mask可见率尚未量化。

![真实观测和采集路线](observation_real_rollouts.png)

### 四原任务可采集性复核

`observation_rlbench_pilot_20261002d` 已完成四任务各1父场景、每父3次路线尝试，**12/12通过各原任务成功条件**；全部12次完整状态/RGB恢复差为0，0条1cm近重复。实际耗时433.585秒。它验证了四任务的同状态采集链路和事件执行可行性；接近段插入引导方案仍属derived采集，规模不足以作为任务泛化或benchmark主结果。全轨迹连续碰撞认证和离散通道类型仍缺失，不能根据此表给学习模型的Valid@K或UniqueValid@K。

| 原任务 | 父场景 | 成功/尝试 | 原任务事件 |
|---|---:|---:|---|
| reach_target | 1 | 3/3 | 末端到目标检测区域 |
| pick_and_lift | 1 | 3/3 | 抓取目标并抬升至成功区域 |
| push_button | 1 | 3/3 | 按钮关节下压 |
| take_lid_off_saucepan | 1 | 3/3 | 抓盖并抬起 |

### 保留的失败与修复

| run_id | 真实结果 | 判断/修复 |
|---|---|---|
| observation_rlbench_pilot_20261002 | 审计JSON numpy数组序列化异常 | 转为明确JSON数组，未收录路线 |
| observation_rlbench_pilot_20261002b | 12/12恢复审计拒收 | RNG重设不足；开始显式恢复模拟状态 |
| observation_derived_reach_pilot_20261002 / b | config tree后仍有joint_velocity及RGB差 | 停启物理引擎并统一稳定化，不放宽标准 |
| observation_derived_reach_pilot_20261002c | 3/3恢复差0，但3条终点失败 | stop/start重置了位置控制flag；恢复位置控制 |
| observation_derived_reach_pilot_20261002d | 9/9成功且完整恢复0差 | 启动32父正式pilot |
| observation_rlbench_pilot_20261002c | reach采集成功，切换pick_and_lift时模型句柄不存在 | 在停止状态导入新任务；以d版重新核验四任务 |
| observation_derived_reach32_20261002 | Python采集与数据审计完成，但外层wrapper实际退出1 | 运行期间为补source冻结而修改了shell，Python结束后bash读剩余行时报unexpected EOF；保留为 `collection_complete_wrapper_failed`，不伪记exit0。275轨迹、summary与hash审计均已完整写出；后续使用不可变wrapper/commit release |

### 冻结Qwen普通头失败后的实测定位

`runs/observed_frozen_v1/seed0` 的best checkpoint为750步；本轮使用CPU1重新读取实际checkpoint并同时评价TRAIN和DEV_MODEL，没有改动3cm验收标准。94条有路线监督的指令参与路径评价（TRAIN71、DEV23）；无成功参考的两条仍保留但不计入路径训练/评价。

| checkpoint / split | 轨迹ADE (m) | 候选终点误差 (m) | 严格语义目标正确率 | 仅最近目标身份（诊断，不是成功率） |
|---|---:|---:|---:|---:|
| best750 TRAIN | 0.06451 | 0.12675 | 2.11% | 88.73% |
| best750 DEV_MODEL | 0.09966 | 0.19544 | 0% | 81.52% |
| last1000 TRAIN | 0.05552 | 0.10751 | 3.17% | 88.38% |
| last1000 DEV_MODEL | 0.11156 | 0.22283 | 0% | 56.52% |

结论是米制定位不足并伴随后期过拟合；不能把它简化为训练集已经掌握任务。best模型在TRAIN本身仍有12.7cm终点误差。同图仅换语言时，TRAIN/DEV平均终点响应分别28.55/28.49cm，说明语言并未被完全忽略；但每组4候选的终点分散度只有约6mm，也未产生可靠的新方案。

RGB-D与标签核验：96/96目标中心投影在图像内；目标中心到最近观测表面平均22.24mm、最大22.97mm，中心射线观测点到目标最大25.70mm，与球表面到球心距离相符。PyRep内参负焦距是其约定，按原符号反投影、外参按camera-to-world、无上下翻转，可以得到一致的世界坐标。此诊断使用真实目标坐标仅核验数据，绝不进入模型条件。它不等于实例mask可见率或完整遮挡证明，但不支持“全部目标不可见/坐标系整体错误导致约20cm误差”的解释。

诊断源为 `scripts/observation_diagnose_head.py`，汇总、逐场景与可重建预测在 `reports/observation_head_diagnostic/`；服务器同目录包含实际预测NPZ，普通Git按既有规则不收录NPZ。复现：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=<recorded-release> .venv/bin/python -m scripts.observation_diagnose_head \
  --run runs/observed_frozen_v1/seed0 --output reports/observation_head_diagnostic
```

据此实现常规观测定位修复：`routeset/observed_geometry.py` 从当前RGB-D、原相机参数产生12,544个无标签观测点；冻结Qwen条件与当前夹爪状态对可训练点特征作空间attention，产生学习的表面anchor和128维上下文。已有普通集合头共享该上下文；终点为预测anchor加逐坐标有界残差（默认5cm），起点仍为当前夹爪位置。该模块是强观测基线的输入修复，**不是论文核心创新**；未增加真实目标、实例分割标签、完整几何或路径答案输入。

`scripts/train_observed_geometry.py` 默认与原头对齐1000步、batch32、K4、128维头和2层集合块，保留相同已知正例匹配目标及DEV选择标准。总参数1,231,965，其中观测几何321,537。仅缓存原始观测点；可训练点编码每步重算。训练支持完整模型、AdamW、调度器、随机状态、采样器和曝光量恢复，保存best/last。记录原始RGB-D读盘/反投影、geometry与头的单请求时延；明确Qwen编码须另外计入总延迟，尚不冒称端到端机器人系统时延。

CPU1验证已实际完成：6项测试通过；真实两样本各12,544点前后向和参数更新0.167秒，点编码、任务query和路径输出参数确实更新；小配置4步完整训练与2步中断后恢复2步的最终模型逐位相同。详见 `reports/observation_geometry_validation.json`。这证明实现和恢复链路可运行；geometry版本的训练收益需由随后固定commit的真实GPU实验判定。

### 评价协议修复：observation_eval_v2

发现并修复了真实分母错误：最初loader因规划器未采到参考而同时删除了两条观测，导致DEV语义只评23条。参考集不完整不意味着目标不存在，因此新版保留全部96观测，TRAIN loss仍只用71条有参考指令；DEV语义评价改为24条，参考ADE/终点/事件仍只统计原23条，缺参考行这些指标为null，不当作失败或有效。

严格颜色身份和3cm阈值完全不变，DEV checkpoint选择仍基于原23条参考ADE。旧23条历史输出完整保留。CPU已用原冻结头best750权重重新评价：24条严格语义仍0%，23条参考ADE仍0.09965624m、终点误差仍0.19544326m。新增证据在 `reports/observation_eval_v2/observed_frozen_v1_seed0/`，包含metrics、逐场景、原checkpoint/config SHA、原训练release和实际评价源SHA；没有覆盖旧训练summary。该目录的TRAIN子目录保留71条有参考训练样本诊断；新geometry训练结束会另对完整72条TRAIN观测评价，参考指标分母71。

新版loader/指标与geometry共12项CPU测试通过，特别验证无参考语义失败会降低完整分母、无参考ADE保持null、输入数量不减少。真实数据确认TRAIN监督71、DEV全部24，其中23有参考。新版geometry完整4步与中断恢复再次逐位相同，见 `reports/observation_geometry_validation_v2.json`。在线冻结/LoRA对照必须使用其实际骨干按同协议重评，不能用旧冻结缓存替代已微调骨干。

### 图像必要性弱对照与固定失败可视化

`scripts/observation_language_control.py` 只用TRAIN有参考指令的末端：先对一个父场景/指令的多条参考末端取平均，再按完全相同的instruction跨父场景平均，因此参考多的场景不会额外加权。该弱诊断不读RGB、深度或当前状态，也不生成完整路线。实际TRAIN监督71条，DEV全部24条，未见指令0条；DEV严格语义正确率0%，目标中心平均误差0.281939m，23条有参考的末端误差0.281923m。逐场景、训练均值和源码/数据hash保存在 `reports/observation_language_only_control.json`。这说明该数据中的语言位置先验不足，不证明任意语言模型都无法解决，也不将其作为强路线基线。

几何版本 `runs/observed_geometry_v1/seed0/best.pt`（750步，SHA256 `a5260c2899298a2aabb8db60553805b9cb43b4250bd76eedd106b044eca5d706`）已由 `scripts/visualize_observed_geometry.py` 实际加载。固定选择按ID排序的前两个DEV父场景261024与261025，各三个目标，未依据结果选择。两张图包含全部6条指令的真实RGB/attention、预测终点投影、H24三维及俯视路线；目标坐标只用于可视化标记和既定评价。每条均为0/4严格命中，完整索引、逐场景误差、attention和预测见 `reports/observation_geometry_fixed6/`。

这些固定例子中，目标中心4cm内的观测点仅获得8.3%–33.9%的attention质量，48.8%–55.5%的质量位于三个目标邻域之外；其余在多个目标之间分散，学习anchor常落在球之间。4cm仅为解释attention的诊断邻域，不是新的成功标准；仍使用原3cm终点标准。它支持下一步检查语言条件空间选择及anchor监督，而不能把未命中的平均位置描述为已完成开放词汇定位。

### 自然布局256父数据扩展：试采通过，正式作业运行中

`scripts/collect_derived_reach_fast.py` 保留原可靠native stop/start恢复与原任务自然随机布局，不创建障碍、不重定位目标。每父场景3种目标语言、每目标3个自由运动提案；不同欧氏路线不标为不同类型。每步读tip pose和真实gripper_open，并沿用 `collect_obstacle_reach.py` 的arm/gripper外部碰撞检查。初末均与固定RLBench版本 `Scene.get_observation` 的同名字段逐值比较，任何非零差值拒收。状态、完整对象inventory、RGB恢复要求严格零差；失败保存partial轨迹。父初始化失败单独记录，并为其9个未执行提案记录原因，不自动换种子补齐。原free-derived版本已经直接读取tip，因此本轮不声称未经实测的提速收益。

固定release `bb5289c3e9a2c8bc5c60035aaf6cc666f88207e3` 的试采 `observation_reach_fast_pilot_20261002` 使用独立seed261900、1个DEV父，实际耗时27.9879秒、9/9成功；9/9完整恢复零差，初末官方字段等价均通过，零近重复。独立保存轨迹审计9/9数值有限、9/9终点通过，平均终点误差0.691mm、最大1.178mm。见 `reports/observation_reach_fast_pilot_audit/` 与 `observation_reach_fast_pilot_summary.json`。

正式作业 `observation_reach_fast256_20261002` 已于2026-10-02 08:45:32 UTC启动，PID159691，CPU线程1、软件渲染、无GPU。使用全新seed262000–262255，前192父为TRAIN、后64父为DEV_MODEL，全部开发用途；与旧32父及试采父不重叠。计划256父、768指令、2304提案不能写成已成功数量。启动后73秒的持久化快照仅有3父、27/27成功、零恢复/等价失败；最终结果待作业实际完成后审计。按单父试采粗估约2小时，不能保证此耗时。

运行证据、PID、实际命令、源码hash和下一步审计见 `reports/observation_reach_fast256_job.json`；早期快照见 `observation_reach_fast256_progress_snapshot.json`。服务器每完成一父原子更新data目录summary，并逐尝试落盘。不要重复启动同run；若中断，保留父场景/attempt日志与已写数据，当前采集器会拒绝覆盖现有目录，不能盲目同命令重跑。会话后的后台进程只执行既定采集，不代表仍会自主分析或改代码。
