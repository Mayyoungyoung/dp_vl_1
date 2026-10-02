# 约束变化下的路线更新：开发机制试验

2026-10-02。状态：正式 release `bb307709ecfb81b39eaf266012a64b7a9810bcc7` 全仓 CPU1 **48 项测试通过**，变化数据及真实冻结模型控制已完成；三臂2000步seed0开发筛选已完成且未支持门控，详见 [CONSTRAINT_UPDATE_RESULTS.md](CONSTRAINT_UPDATE_RESULTS.md)。此文是可证伪方法卡，不是已成立的方法贡献。

## 具体瓶颈和假设

静态场景上 joint4 已强于自产 2+2。先前 coverage max 的三训练种子优势不稳定，自产草稿训练修复虽明显提高 2+2，其整体仍未超过直接 joint4。因此本轮只检验一个不同的问题：一处通道关闭之后，有限的新增路线预算是否能通过学习复制不必改变的部分，提高更新后有效覆盖。

假设：给定相同旧集合和新条件，逐点预测保留/改写范围可减少整条回归中的无关几何误差，因而在相同新增 b=1/2 条完整路线预算下优于强全路径补全。若只有训练配对策略带来收益，而显式门控无收益，则不保留门控。若自由匹配全路径头更好，则恢复该强基线，不继续叠加模块。

## 近邻工作与新颖性边界

| 工作及已核验来源 | 已有机制 | 本试验差别与限制 |
|---|---|---|
| [MPNet 扩展原文](https://arxiv.org/html/1907.06013)，v3 页首注明 T-RO 接收；[作者代码](https://github.com/ahq1993/MPNet)，已读 [replan_path 实现](https://raw.githubusercontent.com/ahq1993/MPNet/master/MPNet/neuralplanner.py) | 对不连通的相邻节点局部重规划；代码保留可直连段，调用神经规划修复失败段；原文另有经典规划 fallback | 保留有效段和神经局部重规划已存在。本试验只探索集合条件下、固定新增完整路线预算的预测门控；没有递归无限调用或 fallback。尚未复现 MPNet，不能称胜过 MPNet。 |
| [MARS，作者机构存档，IEEE Access 2023](https://re.public.polimi.it/handle/11311/1255951) | 用预计算多条路线和增量图复用来响应动态障碍，进行 anytime 规划 | 多路线复用和变化后再规划不是新思想。当前测试是几何可控的神经机制筛选，未复现 MARS。 |
| [MTR，NeurIPS 2022 原论文入口](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html) | 意图查询、地图读取与轨迹迭代细化；已在总相关工作表核验 | 输出头、cross-attention、普通细化不能单独作为创新；本试验检验可变约束下保留/替换的预算表现。 |

当前门控属于常见机制组合，不能据此宣称顶会新颖性。只有真实的跨种子、独立设置和观测条件实验说明其必要性，才值得进一步核对与收敛为核心贡献。

## 数据协议：multigate_change_v1

- 仅读取已物理分离的 `development.npz`，有任何非 TRAIN/DEV_MODEL 行便拒绝。原锁定数据未读取。
- 对原参考类型数 R≥6 的每一父场景，枚举全部“关闭某一面墙的一处现有开口，仍留其他开口”的干预。父 ID、split、起终点与其他几何完全不变。不开新父 split。
- 只将该开口 mask 从 1 改为 0，开口坐标与槽位不重排，模式仍为原物理开口对 `4*first+second`。关闭后仍合法的原正参考保留，逐条用原完整线段 checker 复验。
- 旧集合来自已训练普通 saturation joint4 的实际预测，默认 checkpoint 为服务器 `runs/v2_round2/saturation_k4_seed0/best.pt`。实际路径、checkpoint SHA、数据 SHA、原训练配置/步数与代码 commit 入 manifest。
- 每个父场景可能派生不同数量的干预；训练先等概率取父再取其干预；评价先父内平均再跨父平均。不会把多次干预当独立父场景增加样本量。
- 数据中的参考、模式、关闭位置和新有效性仅按规定参与训练标签/评价。模型输入只有旧/新真实几何、旧 4 条真实预测、共同完整路线 checker 的有效位。这仍是受控层 A，不能冒充观测机器人任务。

## 共同推理和三个训练臂

每法先对旧 4 条路线执行同一完整路径 checker，按新条件无效优先、原槽位次序处理平局，为新增查询指定源路线；全部 4 条旧路线都经 attention 输入。此规则是共同的确定性准备，不算新机制，也不含逐段编辑真值。

| 臂 | 路线训练目标 | 输出方式 | 辅助标签 |
|---|---|---|---|
| full_free | 对尚未由有效旧路线覆盖的所有已知正参考作 saturation 自由匹配 | 普通 attention 全路径回归 | 与另两臂相同的独立固定配对代理 |
| full_paired | 按源旧路线到剩余正参考的距离做最小成本 b 对配对，再拟合该正例 | 同一普通 attention 全路径回归 | 固定配对代理 |
| local_paired | 与 full_paired 逐元素同一配对正例 | 对网络预测为保留的点精确复制；其他点由完整回归提议改写 | 固定配对代理 |

所有臂的参数形状、初始化、训练父场景/顺序、b 日程、辅助头与辅助权重相同。开始时门全开；初始输出在浮点舍入范围内相同。`full_free` 防止为局部编辑设计的配对策略削弱强集合基线。它的路径匹配会随预测改变，但辅助标签固定且与另两臂一致；辅助头在 full 臂不控制输出。

固定配对使用全部尚未覆盖正例，不随机抽子集；已知正例不足 b 时允许合法重复；已覆盖全部已知模式时退回完整正例池。没有把未匹配输出监督为不存在，也没有把参考条数当总解数。

逐点标签是“固定配对正例与源旧路径相差超过 2cm”的距离代理。它不是几何碰撞 mask、因果影响真值或真正最小编辑区间。标签只进入 TRAIN 的 BCE 辅助损失，不在 forward 参数中，DEV 查询该标签的函数会直接拒绝。local 在训练用硬门控直通梯度，测试用预测符号阈值；不使用测试真值 mask，不事后修复局部方法结果。

## 预算、主要指标和控制

默认三臂各 2000 更新、batch64，b1/b2 交替，因此各 **192,000 个新增完整路线反传槽**；width128/depth2，lr3e-4、辅助权重0.01，每500更新在 DEV_MODEL 选优。数据准备、训练缓存准备、前向、检查、评估与额外延迟实测，GPU小时和峰值显存记录。该预算为初次筛选，是否延长应读收敛曲线后决定。

每个请求的生成预算是 **旧4 + b = 5/6**。完整编辑后的路线也算新增，失败、重复、拒绝路线全部保留并计数。保留旧有效路线是所有法共有行为，不能靠必然较高的保留率宣称机制优势。当前报告累计候选池的 UniqueValid/ReferenceCoverage/Valid/AnyValid 与新增有效类型、碰撞和旧失效数分层；候选池不合并随机重复。未实现独立学习评分器，因此不虚报 SelectedValid。

数据准备同时输出：no-update（总预算4）、同一冻结普通头新条件下 K1/K2 + 共同 checker（总预算5/6）、新条件重算 joint4（累计8）。冻结头仅在 K4 训练，因此截断 K1/K2 是零适配诊断，不是充分训练的 K1/K2 强基线。三臂中 full_free 是本轮认真训练的强条件补全对照。

延迟区分第二次更新请求和两次请求总成本。数据准备实际量测 CPU 上旧4+新b+完整检查；训练评价量测当前设备第二请求的 checker、网络、输出复制与完整检查。尚未把 CPU 旧请求时间与 GPU 新请求时间拼成端到端数字，亦未把缓存批吞吐写成单请求延迟。若有正信号，下一次公平部署应在同设备直接测完整两请求与固定时间预算。

## 证伪和升级条件

主要判断是 local_paired 对 full_paired 及 full_free 的更新后有效类型/覆盖，而非代理 mask 准确率。分别看原本 0/1/2/更多旧路线失效、因变化新失效的分层。若总体收益由原本无须修复的场景贡献，不能宣称替换可靠。

local 若仅胜 full_paired 而不胜 full_free，局部门控不构成对强基线的优势。若配对全路径已同样好，丢弃门控。若距离代理几乎全点为编辑，说明不存在所设想的局部保留空间。先做真实数据/预算/收敛诊断，单次失败不停止整个研究，但也不据此追加十个模块。

只有低成本筛选显示质量—成本优势后，才启动三种子、独立几何变化及真正观测的局部障碍变化。当前 RLBench 自由绕行数据不能用曲线抖动代表离散通道，更不能被本受控结果替代。

## 已实测验证与启动命令

正式 release 的全仓测试 **48 passed in 6.29s**，作业 `cpu_tests_bb3077` exit0。随后 `make_data_bb3077` exit0，核心数据准备14.83秒，含进程导入的作业墙钟21.46秒，均CPU1、CUDA隐藏。只读代理诊断 `train_proxy_bb3077` exit0，36.40秒。三个完整作业记录已复制到 `reports/constraint_update_v1`，无后台遗留作业。

数据 SHA256：`e5658451e937ca4f07196d9779fee62746c21ae2a5b470121e8518a970a8e687`；输入 development SHA 与历史匹配，为 `f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057`。冻结模型 SHA `b2cb51ae0de50785a12f129c2ff12ecc981a9df826b5ee7ae1462de6366f102c`，选择步1500。正式数据含381 TRAIN父/2364干预，57 DEV_MODEL父/356干预；不是2720个独立父场景。全部保留正参考的完整线段验证和模式ID检查通过。

### 正式数据准备结论

TRAIN内点距离代理平均编辑比例 b1为77.12%、b2为76.69%，所以约23%的内点存在复制空间。b1/b2 分别有19.46%/23.39%的路线需要编辑全部内点；仅16.33%/20.14%的路线能复制至少一半内点。该设置**存在但并不宽裕的局部保留空间**，不能预先假定局部模型会胜出。

仅在 TRAIN 注入已知配对正参考，并按该距离代理复制源路径未编辑点：b1的2364条、b2的4728条混合路径均100%有效、100%保留参考模式身份，与不复制的参考注入结果一致。父场景加权 union UniqueValid 为3.65194/4.48186，新增类型0.97900/1.80892。这说明当前代理不会在这批训练参考上必然破坏可行性，但它是**训练标签注入诊断**，不是学得性能、DEV结果或一般理论上限；没有读取DEV真值编辑mask。可复现实测脚本及SHA见 `reports/constraint_update_v1/analyze_training_proxy.py` 和 `train_proxy_bb3077.json`。

DEV_MODEL真实冻结控制（57父均匀加权）：

| 方法 | 累计生成数 | 变化后 union UniqueValid | 新增候选有效率 | union ReferenceCoverage |
|---|---:|---:|---:|---:|
| no-update | 4 | 2.69323 | 不适用 | 0.45955 |
| 冻结K1零适配 | 5 | 2.70013 | 1.33% | 0.46023 |
| 冻结K2零适配 | 6 | 2.71036 | 7.60% | 0.46279 |
| 冻结joint4重算 | 8 | 2.70702 | 32.37% | 0.46237 |

这个弱控制结果不能用来夸大新模型优势。历史模型只在有效开口为前缀槽位的表示上训练，本干预为维持物理模式ID而保留关闭后的空槽；关闭中间开口会产生未见过的非前缀mask。它同时面临约束变化与表示分布变化。`full_free`、`full_paired`、`local_paired` 都将接触同一变化分布并接受相同训练预算，必须以认真训练的 full_free 为强基线判断门控。该问题是增加强基线的理由，不能事后给局部方法单独重排/修复输入。

决定：数据和代理的几何自洽检查支持按原计划进行一次三臂配对筛选。其目的仍是证伪门控必要性；如果无法胜过 full_free，则保留最强普通补全，不能拿本表的零适配失败作主要积极证据。

### 早期针对性验证和可恢复命令

2026-10-02 CPU1、GPU 隐藏，`.venv`：`pytest tests/test_constraint_update.py -q`，**6 passed in 5.57s**。验证包含保留物理模式 ID/全部闭口后正例可行、父 split、拒绝 locked archive、DEV 不能获取训练代理标签、相同初始化/不接收推理 mask、失效与重复仍计候选预算、真实 4 步 vs 2 步加断点恢复的所有模型权重逐元素一致。

验证采用独立源快照（非正式训练 release），目录 `research_v2/checks/constraint_update_8dafdb13552e`，源 tar SHA256 `72c5b325f691922824aa18123c970998df86f61d08459c375e92d33d8020e32c`。验证后只修正了数据准备延迟字段名 cpu1→cpu；正式训练须由根线程 commit 后固定 release。

从正式 immutable release，服务器项目根路径记为 `P=/home/wzy/dpvlm/route_set_v1`：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$P/.venv/bin/python" -m scripts.make_constraint_update \
  --data "$P/data/multigate_v1_partitions/development.npz" \
  --checkpoint "$P/runs/v2_round2/saturation_k4_seed0/best.pt" \
  --output "$P/data/multigate_change_v1.npz" --threads 1

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$P/.venv/bin/python" -m scripts.train_constraint_update \
  --data "$P/data/multigate_change_v1.npz" --output "$P/runs/constraint_update_v1/seed0" \
  --arm all --steps 2000 --batch-size 64 --eval-every 500 --threads 1 --device cuda
```

应由现有作业记录器写 run_id/PID/日志/退出码/代码 hash 与恢复命令。每个臂只留 `best.pt/last.pt`，包含模型、优化器、调度器、Python/NumPy/Torch/CUDA RNG、父场景 sampler、步数与曝光；恢复指定 `--arm <原臂> --output <原臂目录> --resume`。未自行启动 GPU 或创建提交。

## 2000步负结果与一次等预算续训

full_free 的 b1/b2 union UniqueValid为3.63981/4.43344，full_paired为3.60180/4.29793，local_paired为3.56393/4.27147。三臂各192000目标槽、464984参数，所有源路径、父场景、参考池曝光和数据hash均已由分析器校验。DEV仅有0/1/2旧路线失效分层，不对缺失的3/4层作结论。不是负结果终稿，而是排除当前门控贡献的一轮真实机制检查。

local的碰撞主要发生在预测编辑区域内部，而非复制边界；碎片也没有更高失败率。因此没有证据支持添加连通mask模块。三臂best都在最后2000步、配对路线损失仍在下降，根线程决定唯一下一项修复为等预算续训到4000步。若仍不胜full_free，则丢弃门控；不加新模块、不单独延长local、不改变评价规则。

`train_constraint_update.py --continue-from <v1/arm/last.pt>` 初始化一个**新输出目录**，恢复完整模型/优化器/调度器/RNG/sampler/history和累计曝光，原best复制到新目录仍参与选择。除增加总steps、新output和代码来源记录外，数据路径/hash、arm、模型、学习率、batch、seed、eval频率、线程/设备、辅助权重、代理阈值等均要求一致。新目录同一命令加 `--resume` 时恢复自身last，不重新初始化source；若提供不同source便拒绝。

成本字段避免重复计算：`elapsed_s`、`incremental_elapsed_s`、`gpu_hours_reserved`是该新输出树新增成本；`cost_origin`保存原投入，`cumulative_elapsed_s`和`cumulative_gpu_hours_reserved`各含原投入一次；`trajectory_exposures`为累计槽数，`incremental_trajectory_exposures`为本段新增。源last、源best、原summary SHA、原代码commit/步数均入config。若原best仍胜出，summary明确标记 `selected_checkpoint_from_prior_run`。

定向实测：**1 passed in 4.74s**，真实2步原run→新目录续到4与连续4的所有模型权重逐元素一致，scheduler和sampler一致；更换data或lr明确拒绝；原last/best/config/summary四文件hash均未变。CPU1/CUDA隐藏，PID198395、child198398，exit0。测试源快照 `research_v2/checks/constraint_continue_540dde81c861`，tar SHA `cb509c32a1f9dd89f3e6c3338218e3aeea40fa8062545485a7645caf9c0df30b`；日志和状态在 `reports/constraint_update_v2_validation`。该快照是提交前验证来源，正式运行仍须根线程固定新commit。

根线程正式部署后逐臂执行（不要 `--arm all` 配一个source）：

```bash
python -m scripts.train_constraint_update \
  --data "$P/data/multigate_change_v1.npz" \
  --output "$P/runs/constraint_update_v2/seed0/full_free" \
  --arm full_free --steps 4000 --batch-size 64 --eval-every 500 --threads 1 --device cuda \
  --continue-from "$P/runs/constraint_update_v1/seed0/full_free/last.pt"
```

另两臂将路径和arm成对替换为full_paired/local_paired，使用相同4000总步数。此处命令由根线程放入现有GPU1/35%和record_job封装，当前子任务未启动训练。复用分析器时为续训指定新 `--output` JSON和 `--report` 路径，保留v1报告。
