# 观测扩散拟合审计：代码事实、剩余不确定性与单一修复试验

2026-10-03；只读审计完成，未修改实验实现／配置，未 SSH、加载模型、生成路径或读取保留集。本报告新增计算仅为本地 JSON／日志汇总、源码 SHA 比对和既有 teacher 噪声的 NumPy 重建；新增模型 forward、优化、搜索均为 0。

**结论：没有发现足以解释失败的 DDIM 公式、坐标单位／还原、正例对齐或条件分支断梯度错误。已有证据明确支持“已训练场景中的路线主体去噪／自由生成拟合不足”，尚不能归因于扩散家族失效、多正例平均、条件被忽略或几何泛化。** 一个有区分力的新读数是：最低噪声 t=0 时，模型重建甚至弱于不去噪的反缩放输入。建议至多做下文一次有上限的 **x0 时间步损失加权** 对照，作为普通基线修复；未实施，也不是核心方法候选。

## 1. 审计来源与字节身份

实际训练源为 `6e0203ba1335f9fa9975c523c657959f8bc9ab60`。审查了归档在 `observed_two_row_diffusion_v1/source/` 的 `.py.txt` 原件，并实际核验下列 SHA 等于两臂训练 config 的 `source_sha256`。当前工作区对应源码仅换行形式不同，统一 CRLF/LF 后逐字节相同；下文行号对应这份相同逻辑，不把工作区文件 SHA 冒充运行 SHA。

| 原运行文件 | 已核对 SHA256 |
|---|---|
| `scripts/train_observed_two_row_diffusion.py` | `fda876c5b2f2960f9434530f6cb6b4739085cf8c68487eaa872ad797cab9deaa` |
| `routeset/observed_route_diffusion.py` | `19fb4104d86094bbbd7742210369ab983b88c21108a282766307f3b5d93e7d9c` |
| `routeset/observed_diffusion_stream.py` | `2a3efa744d02b5ac71a4e8b2dee58578ee969fa9f5285cd1198204696232838d` |
| `routeset/observed_geometry.py` | `5bd25f43e780d0ec5ed1c9f336bbbe0b01361175f876eacd01fdeada1a9f8f9d` |
| `routeset/models.py` | `78e94885397d2440e5fa0761e8bf0cc596bbda3ee04c309024aff7bb0e37819b` |

关键结果原件均位于 [扩散归档](observed_two_row_diffusion_v1/DIFFUSION_RESULTS.md)：

| 原件相对该 family 的路径 | SHA256 |
|---|---|
| `independent/summary.json` | `8ecf07ccf6322f565734653a727b82750b08cccd507a5a89c98ae56ba58ab80e` |
| `set/summary.json` | `aa107388729207345a8364fb35e73434175796820321834834871c8e8aa2ad13` |
| `independent_denoising-diagnostic/teacher_diagnostic.json` | `9bdf0a28f34e3052a29014798ba645491d0fa9841181d73125ba71ed47893cbc` |
| `set_denoising-diagnostic/teacher_diagnostic.json` | `20c1a1ca3bf34a153138fdc04345e63ecc614d11b531e86c8b24232c734de3ca` |
| `analysis_run/analysis/analysis.json` | `235039f9261095cc0fd551382afa02c01357f85de9eaff0421a1f188b68464ee` |

## 2. 已确认的代码事实

**坐标和事件。** `observed_route_diffusion.py:30–53` 将 H24 后 23 点变成 `[xyz-current_xyz, 0.2*open]`，XYZ 单位仍是米；decode 精确逆变换并补回观测首点／首事件。不存在毫米／米混用，也没有标准化后漏做逆变换。该状态没有数据尺度标准化，这是配方选择，并非自动构成程序错误。普通 [Diffusion Policy 官方 normalizer](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/normalizer.py) 确实提供训练数据拟合的可逆尺度变换；本实验没有复现其完整配方，不应称其系统复现。

`forward_x0:124–144` 的 92 维含噪输入包含终点和事件。中间 XYZ 为预测 anchor 的直线加可训练残差；末端为同一预测 anchor 加逐坐标 `0.05*tanh`，事件为 `0.2*sigmoid`。anchor 来源是观测点的学习 attention／peak，没有真实终点桥。完整未来路径仅用于训练目标和 teacher 加噪，未进入自由采样条件。

**训练目标和随机流。** `train_observed_two_row_diffusion.py:377–398` 为逐元素、均匀时间步的 x0 MSE，加原 `0.02*grounding_loss`；没有对含噪候选重新 Hungarian 匹配，没有把一条的噪声配到另一条标签。stream 有父样本、positive（正参考抽样和置换共用）、时间、epsilon 四个独立可恢复 RNG 流。R≥4 时不放回抽 4，R<4 时保留全部正例再补采，unknown 未过滤；四候选共享本输入的 t、epsilon 各自独立。独立／集合的实际完整流 SHA 相同；普通回归仅父抽样链相同，其 full-positive saturation 监督与这里的四参考抽样不同。

**DDIM 数学。** `ObservedX0Schedule:150–227` 使用 100 步 cosine beta、40 个不重复降序整数时刻，eta=0：

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad
\hat\epsilon=(x_t-\sqrt{\bar\alpha_t}\hat x_0)/\sqrt{1-\bar\alpha_t},
\]
\[
x_s=\sqrt{\bar\alpha_s}\hat x_0+\sqrt{1-\bar\alpha_s}\hat\epsilon.
\]

它与 [DDIM 原论文](https://arxiv.org/abs/2010.02502)及[作者 `generalized_steps` 实现](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py)的 eta=0 更新相符；只是此处网络直接预测 x0。第 40 次 t=0 后直接返回 x0，等价于 s=-1、alpha=1，没有第 41 次补修。高噪声处没有 epsilon→x0 除以极小 `sqrt(alpha)` 的网络误差放大；此前 controlled epsilon 失败原因不能直接套到这里。原测试覆盖真实 beta 数值、oracle 中间步和最终还原、40 次调用，当前没有公式级反证。

**条件梯度。** `encode_observation:117–122` 严格白名单，context 来自 Qwen feature、current 和观测 geometry；`models.py:53–61` 逐 block 注入该 context。geometry peak 使用 straight-through soft-anchor 梯度。末步原始 `gradient_audit` 中独立臂 53/53 活跃张量、集合臂 65/65 活跃张量均 present/nonzero/finite；独立臂另外 12 个候选 attention 张量明确不活跃。这排除了“整个条件支路被无依据冻结／断开”的解释。**它不证明路线主体充分使用几何，更不证明正确的因果依赖**：当前审计未做真实数据条件置换或 Jacobian 干预；grounding loss 本身也可给 geometry 非零梯度。仅凭梯度日志不能宣称每个模块的实际任务作用已验证。

**teacher 与自由生成不同。** `denoising_diagnostic:588–630` 在首 6 条 TRAIN、每条前 4 个正参考上逐时刻单次预测 x0；没有 rollout，也不是独立生成池。自由评价 `evaluate_pool:230–319` 从纯高斯出发，几何编码一次、40 次去噪，全部 K4 封存后才检查。两条路径的单位、decode、条件实现相同，输入分布不同；teacher 误差不可直接换算成自由成功率。

## 3. 实际失败集中在哪里

固定 last 的全部 285 TRAIN／1140 候选：普通回归 TipValid 989、语义正确 1045、TipClear 1080；独立扩散分别 392／1121／399，集合分别 465／1116／476。两扩散臂事件正确率均为 1。端点／事件并非当前主要失败分母；两扩散臂语义正确但碰撞为 729／651。TRAIN 最近正参考 ADE 约 9.89／9.75 cm，普通为 1.78 cm。这已经是训练场景内问题，不能只用 DEV 泛化解释。

既有 teacher（先逐输入算坐标 RMSE，再取六输入均值）：

| t | 独立 XYZ RMSE mm | 集合 XYZ RMSE mm | 独立／集合末端欧氏误差 mm | 独立／集合事件 MAE |
|---:|---:|---:|---:|---:|
| 0 | 35.118 | 37.048 | 16.369 / 21.195 | .002485 / .003643 |
| 25 | 69.684 | 72.050 | 17.504 / 20.931 | .002712 / .003010 |
| 50 | 126.472 | 138.774 | 17.349 / 21.160 | .001872 / .001863 |
| 75 | 126.753 | 133.955 | 16.786 / 20.812 | .001881 / .001416 |
| 99 | 130.175 | 131.067 | 16.945 / 20.727 | .001813 / .001284 |

新增的**零模型算术对照**使用原 `default_rng(400000)`，按原顺序重建 30 份 `(1,4,23,4)` float32 噪声，全部 SHA 与原 teacher rows 匹配。t=0 时 `alpha≈0.9993687272`，不去噪的反缩放输入 `x_t/sqrt(alpha)` 相对真值的坐标误差恰为 `sqrt((1-alpha)/alpha)*epsilon_xyz`，不需要重新打开参考。等效单坐标噪声标准差为 25.133 mm，六输入实得如下：

| TRAIN id（前缀 `two_row_reach_`） | 不去噪 RMSE mm | 独立 teacher mm | 集合 teacher mm |
|---|---:|---:|---:|
| 283200_target0 | 24.533 | 34.629 | 36.673 |
| 283200_target1 | 25.007 | 32.341 | 32.200 |
| 283200_target2 | 24.624 | 51.857 | 58.837 |
| 283201_target0 | 24.569 | 28.998 | 32.094 |
| 283201_target1 | 26.770 | 29.070 | 26.270 |
| 283201_target2 | 25.521 | 33.815 | 36.212 |
| 均值 | **25.171** | **35.118** | **37.048** |

11/12 个臂×输入的低噪声重建比反缩放输入更差。这一比较同为三个坐标的 RMSE，包含终点、排除已固定的首点；不是三维欧氏距离，也没有按碰撞筛例。它证明该有限 teacher 面板存在可改善的低噪声重建空间，**不证明直接复制这些输入能成为有效自由生成器**。噪声标准差本身还大于 20 mm tip 检查尺度。

高噪声处也不能宣称“噪声未被读入”：t=25/50/75/99 的等效反缩放单坐标噪声为 0.444/1.044/2.548/2029.21 m，而网络显著压低了误差。高噪声 teacher 的条件最优解可有多模态不确定性，单个正例 RMSE 并非分布质量；不能仅据约 13 cm 数值推断模式坍缩。这里既未保存全部中间自由采样 x_t，也没有已实测的逐 t 梯度冲突，因此不能定位“第几步把有效路线毁掉”。

原训练日志每 25 步一次的 x0 MSE（非全步平均）在四个 3000 步区间的均值：独立 `.011012 → .009649 → .008975 → .008570`，集合 `.011020 → .009604 → .008852 → .008469`。损失在下降，不能称完全不收敛，也不能将下降等同已充分拟合。末步 x0／grounding 标量分别约 `.007992/.074116` 和 `.008253/.074112`（后者已乘 .02）；标量大小不等于梯度主导权。末段记录到的预 clip 梯度范数最大值为 .774／.451，未显示持续裁剪到 1 的证据。没有理由据此再盲目加步数或只换 seed。

## 4. 唯一建议：有上限的 Min-SNR 时间步加权配对

**待证伪假设，尚未实测：** 当前均匀 t、均匀 x0 MSE 的优化分配不利于低噪声路线主体精度；一次标准时间步加权可能改善低噪声恢复，并在固定 40 步自由生成中改善几何有效率。该假设不包含“已经证明时间步梯度冲突”或“已确认 x0 参数化错误”。

选择最小修改是为了保留坐标、端点、事件、网络、100/40 时刻与 DDIM 的现有含义。**本次不同时加归一化、skip 模块、局部几何、更多采样步或重新初始化骨干。** [Min-SNR 原论文](https://arxiv.org/abs/2303.09556)及[作者 `training_losses` 的 START_X 分支](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/main/guided_diffusion/gaussian_diffusion.py)给出 x0 的 `min(SNR,gamma)` 权重；这属于既有优化方法，不是本项目创新，也不是完整复现该论文。

拟固定 gamma=5，不扫参数；对每个输入先计算原 4×23×4 的 MSE，再乘

\[
w_t=\frac{\min(\bar\alpha_t/(1-\bar\alpha_t),5)}{\frac1{100}\sum_{u=0}^{99}\min(\bar\alpha_u/(1-\bar\alpha_u),5)}.
\]

分母是本项目为了保留平均 route-loss 标量尺度而明确增加的常数归一化（约 1.981479），不是作者实现原样；grounding 系数仍 .02。它不能保证梯度范数不变。当前 t=0/25/50/75/99 权重约 2.523/2.523/.463/.0778/1.23e-7，故必须检查高噪声及自由生成退化，不能只报低 t 改善。完整 weight100 数组及 SHA 应预封存。

若 root 决定执行，建议仅做下列一次 TRAIN 诊断配对：

1. **仅独立扩散**，两臂从同一个已完成 `last12000`（SHA `53b8550cc950be68c39fbabc46efbf7a1143429c4f13a5251c37da9632deaa8d`）在两个新输出目录分叉。共有模型、AdamW／scheduler、4 个 stream RNG 和训练 RNG 都从原状态恢复；原目录／summary 不改。A 原 loss，B 仅上述加权。新 lineage 协议记录这是新增试验，不能修改旧 config 的 steps 然后冒充原 resume。
2. 各新增 **500×32=16000 输入抽样／64000 正目标槽**，同285 TRAIN、全部已知正参考策略不变，共用新增 parent/reference/t/epsilon 实际 digest。lr、权重衰减、clip、K4/H24、预测端点、事件均不改；固定 step12500 结束，0 DEV 选模。沿用 checkpoint/journal 的失败保存和拒绝隐形重放。
3. 每臂一次原首6 TRAIN teacher 面板：6 geometry、30 denoise、120 中间状态；一次全285 TRAIN自由池：285 geometry、11400 denoise、1140最终候选。相同原 seed300000、40 步、K4，分别评价，不合池。检查器、unknown、失败分母原样保留。每臂新增总 geometry=791、denoise=11930、optimizer=500；这些与旧12k成本分别计。
4. 预先报告两个目标：低 t 0/25 的配对坐标／body RMSE，以及全285 TRAIN自由 TipValid、TipClear、语义正确、已分类 Unique、有效 unknown、最近参考 ADE。前者改善而后者不改善（或只靠端点／unknown变化）就不足以支持扩展；高 t 明显退化必须原样报告。首先看逐父配对差，不把一次 seed 的小数值差当可靠方法优势。
5. 如果 B 不同时改善路线主体恢复与自由几何生成，就停止这一个加权修复，不调 gamma／温度／seed／更多步来追结果。若通过，才由 root 决定是否值得从头的完整公平强基线；本次 fork 自旧权重、有额外曝光、仅 TRAIN，**不可加入 MAIN 当独立正式性能结果，也不能据此声称集合机制成立**。

依据历史完整主体与固定 TRAIN 作业，两臂合计预计约 3–5 分钟量级，属于估计而非承诺；启动／加载、500步训练、teacher、TRAIN自由评价和 checker 成本分别实记，外层与主体不重复相加。可设每臂所有阶段合计 6 分钟的请求／更新边界停止上限，保留中断和 CUDA overshoot；不自动加预算。此报告没有授权或启动上述实验。

## 5. 研究判断的边界

当前能确认的是：真实冻结 Qwen 和观测几何已接入，数据／随机流配对成立，数学与单位未发现错误，事件／端点相对稳定，但路线主体在 TRAIN 上仍缺乏充分拟合证据。这个状态下，独立与集合的 DEV 差别不能用来论证扩散无效或为新核心机制背书。去噪的有限优化修复和可变布局数据建设可以分开推进；不能用普通优化工程替代用户要求的清楚机制贡献。

变量布局 eba0994 的冻结 runner 和 tests／prepare／pilot4 三 wrapper 已另获 observation 只读复核，无实际 blocker。114 精确 JUnit 身份／0skip、metadata-only prepare、fresh 首4、CPU1／CUDA隐藏及仅清理自身 Xvfb 均对齐；本次未部署或启动它们。
