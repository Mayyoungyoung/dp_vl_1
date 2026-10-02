# 原观测 A* 的空间路径惩罚对照（预登记 v1）

本项是**标准多样化规划的受限预算适配，不是方法新颖性证据**。当前仅实现与本地接口测试，尚无真实 TRAIN/DEV 结果。根线程负责冻结源、记录作业和实际启动；不修改历史规划器、物理采集器或已有结果。

## 证据、假设与最近先例

固定32 TRAIN 的原观测 A*v2 在36 DEV输入上，144槽中96条 TipValid，已知有效类型数24（Unique=24/36），已分类有效重复65（均值1.8056）；没有完全相同的 grid path 重复。原报告 SHA256：`6c108dff3c01dbf54053f80ea9eba563e57e654de6959466e360b00e2218e06e`。这提示“惩罚同一条离散边”可能不足以获得不同路线类型，但**不能证明原因**。全部已分类有效路线均为 middle–middle；保守的可见自由图可能已经排除了其他类型。

唯一检验：保持原端点、可见自由图、接入、搜索上限和K4，扩大对历史完整路径的空间代价支持，是否能提高已知有效类型覆盖，同时报告有效率与时间代价。没有预设它一定成功。

| 原文核验 | 已有内容 | 本次边界 |
|---|---|---|
| Kobitzsch、Radermacher、Schieferdecker，ATMOS 2013，[Evolution and Evaluation of the Penalty Method for Alternative Graphs](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.ATMOS.2013.94)，[原文PDF](https://drops.dagstuhl.de/storage/01oasics/oasics-vol033-atmos2013/OASIcs.ATMOS.2013.94/OASIcs.ATMOS.2013.94.pdf)，§4.2 | 重复搜索后提高已用边及相邻重接边代价；惩罚、多样性与路径长度的权衡已有先例 | 本项不是其完整算法复现；其不同搜索/筛选预算不能直接写为本文K4 |
| Li等，ADC 2022，[Diverse Shortest Paths in Game Maps: A Comparative User Study and Experiments](https://aamircheema.com/research/ADC2022_DiversePaths.pdf)，§2.2–3 | 将 Penalty、Plateaus、Dissimilarity 适配到二维游戏地图可见图；Penalty可迭代到收集足够unique路线或达到长度限制 | 本项只允许四次搜索，不循环到四个unique；3D观测图的Gaussian场是工程适配，不声称为该文原样实现 |

原文公开的代码入口为 [ADC作者代码](https://bitbucket.org/lingxiao29/customized/src/master/)，本轮未运行或复现该代码。Gaussian半径、普通A*、代价更新本身均不构成新贡献。2026-10-03核验作者PDF与出版源；出版源本轮一次网络超时不改变前轮已读原文记录。

## 唯一代价变化

对每条先前已返回的完整 raw 路径 `P_j`（含 exact start/goal connector），在原自由格点顶点 `x` 上计算：

`phi(x) = sum_j exp(-distance(x, P_j)^2 / (2 * 0.05^2))`。

`distance` 是到完整连续线段的精确欧氏最小距离，包括零长段；不是到H24顶点或路径类型标签的距离，也不是离散距离变换近似。每条路径贡献一次Gaussian；重合路径仍分别累加。固定 `sigma=2 voxel=.05m`，原 `voxel=.025m`。

grid edge `(u,v)` 的新代价为 `length(u,v) * [1 + 4 * (phi(u)+phi(v))/2]`。虚拟start/goal边保留原 `length * [1+4*Counter(edge)]`，不改变端点或接入集合。代价非负且不低于原欧氏长度，因此原欧氏启发式仍可用；有限节点/时间上限依旧可能失败。

scoped adapter只在一次请求内替换 `astar_virtual`，退出或异常后恢复。记录每次返回的cells和raw路径；每次搜索前严格检查旧调用方Counter等于全部此前返回路径的边累计，从而兼容原更新行为。对路径是否通过后置proxy/Tip/语义评估完全不可见；**全部完整raw路径都参与惩罚**，不是 invalid-aware coverage。未返回路径不添加虚构草稿。零先前路径直接传回原Counter，连cost proxy都不引入，首路径的cost与tie次序原样保留。

旧三文件 `observation_multiroute_astar_v2.py`、`observation_multiroute_astar.py`、`observation_prototype_grounding.py` 字节不动，沿用原已审LF/CRLF双SHA门禁。原geometry、mask、clearance、unknown policy、26邻居supercover、接入、H24插值和后置检查均复用。新代码不修改原 evaluator。

## 数据、条件与阶段

数据固定 `/home/wzy/dpvlm/route_set_v1/data/observation_two_row_prefix44_v1`，export SHA256 `6ed7786823275f26dba38fff9039bd33127b71567de4e4c0c80a37cfd91e46f9`。32登记TRAIN实际31父/93输入，缺图父283220不替换。共享prototype+workspace必须与原实际A*32完全一致，canonical JSON（只排除 prototype.training_seconds，完整真实耗时仍保存）SHA256 `384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b`。

- TRAIN可行性：固定父283200–283203的全部3条指令，edge/spatial各12请求、48槽，合计24请求96槽。共享同一实际TRAIN拟合，无调参选择。保留真实TRAIN角色，不改名为DEV。每请求先生成/保存/hash K4池，再读该请求的验收标签。仅该请求标签被解码；export文件hash及其他角色机械闭合状态可以核验，不打开未使用DEV原始图像/轨迹/验收NPZ。
- 固定DEV：单独启动同36 DEV输入（283264–283275全部3指令），只新增spatial 36请求144槽；对照既有封存的原A*32池，不重复搜索旧DEV。必须读取同源码、同配置、同拟合结果的封存TRAIN机械preflight，且全部preflight产物hash不变。根线程先审实际TRAIN结果后决定是否执行该预登记DEV，不自动联跑。
- TRAIN预检只验证接口/预算/信息/首路径是否一致，**不以指标挑选sigma或代价系数**。若首路径由于计时边界不同而不一致，同样保留失败并停止，不调整参数重跑。

每个请求只接受RGB、depth/相机、当前夹爪状态、语言及共同TRAIN拟合。目标坐标由原prototype预测。验收真障碍/目标/类型仅在该请求K4池封存后用于相同Tip评价，不进入生成器，不用于筛选或补采样。TRAIN拟合只把原positive route/current传给两个旧fitter，不额外传语义目标、模式或guide。

## 预算、时间与失败

固定每请求四槽、最多四次A*；每次仍为20000节点/2秒，原deadline每64扩展检查方式保留，实际超出情况由原日志反映。定位失败仍占四槽，返回路径不管proxy结果均保存，失败留NaN、重复也计槽。没有隐藏候选、筛后补齐、迭代修复或训练。两阶段新增上限60请求/240槽；实际调用次数与失败槽分别报告。Qwen、GPU、训练、模拟器调用均为0。

每次A*之前从所有先前路径重建field，精确距离计算按16384个自由格点分块，原最多200万格点限制保留；分块只控内存，不改变距离。field预处理不在旧A*的2秒计时器内，**是额外计算，必须进入连续请求walltime**。记录每次预处理秒数、顶点数、field字节数/hash、原搜索秒数及二者和；全请求还包括IO/hash、backprojection、定位、图构建、接入、原proxy、保存封存、标签IO和检查。另列共同TRAIN拟合与完整pipeline耗时。本次是固定候选数对照，不能据此宣称固定时间优势。

新输出必须fresh，不支持自动resume/retry。意外异常停止当前阶段，保留已发出池和failure receipt，未完成请求保留最多四槽不确定上界，后续未尝试请求单列；禁止用同输出或新run ID隐性重推。root使用record_job保存外层PID、exit、线程和wrapper源hash。

## 测试与执行接口

本地实际：初次新12测0.65秒；发现旧prototype包含真实运行时字段后，将拟合身份校验仅排除`prototype.training_seconds`并增加参数/来源拒绝测试。最终 `F:/ProgramData/anaconda3/python.exe -m pytest tests/test_observation_spatial_penalty_astar.py tests/test_two_row_astar_control.py -q`：**23 passed，0.91秒，exit0（新13+旧10）**；纯CPU，无服务器作业。覆盖连续线段/退化段距离、逐路径可加Gaussian、虚拟边原代价、真实原A*首路径和扩展数完全一致、失败proxy仍加入历史、四次上限/Counter一致、作用域恢复、TRAIN/DEV真实角色、封存前不读标签、标签变化不改变输出、unknown失败保留、严格角色/配置、旧三文件字节。真实数据预处理成本和效果尚未验证。

根线程冻结后，CPU1/GPU隐藏、OMP/BLAS/MKL/NUMEXPR各1线程。下列为接口示例，`<immutable-source>` 和 run目录须由根线程固定并记录，不是已执行命令：

```bash
cd <immutable-source>
P=/home/wzy/dpvlm/route_set_v1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/scripts"
"$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_spatial_penalty_v1" --run-id train_preflight --resume-strategy fresh-output -- \
  "$P/.venv/bin/python" -m scripts.observation_spatial_penalty_astar \
  --data "$P/data/observation_two_row_prefix44_v1" --config configs/observed_two_row_spatial_penalty_v1.json \
  --stage train_preflight --output "$P/runs/observed_two_row_spatial_penalty_v1/train_preflight"
# DEV is a separate root-controlled launch after the actual mechanical gate:
"$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_spatial_penalty_v1" --run-id dev --resume-strategy fresh-output -- \
  "$P/.venv/bin/python" -m scripts.observation_spatial_penalty_astar \
  --data "$P/data/observation_two_row_prefix44_v1" --config configs/observed_two_row_spatial_penalty_v1.json \
  --stage dev --preflight "$P/runs/observed_two_row_spatial_penalty_v1/train_preflight/report.json" \
  --output "$P/runs/observed_two_row_spatial_penalty_v1/dev"
```

## 可证伪解释与停止边界

比较相同父/指令下的TipValid、AnyTipValid、UniqueClassifiedTipValid、已分类重复、Unknown有效数、KnownReferenceTypeCoverage、语义、长度、失败槽与端到端时间；未采到的路线不标负，Unknown不强分新类型。更高曲线间距不能冒充更多有效类型。

若空间惩罚仍只给middle–middle，或更多曲线间距仅来自无效/unknown绕行，不能称为覆盖改进。若仅覆盖提高但valid/time退化，明确量化取舍，不称全面胜出。若固定TRAIN12没有覆盖收益，下一项只读诊断应核验这些TRAIN已知正参考能否通过**原visible/grid proxy**，以区分观测图排除与代价不足；本卡未实现该诊断，不改free-map、不调sigma、不增加模块。历史失败的local refiner、copy gate、cross-goal辅助和DTW假设不随本项复活。
