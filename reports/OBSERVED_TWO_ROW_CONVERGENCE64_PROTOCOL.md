# prefix76：普通基线的固定12000步收敛控制

2026-10-03，前瞻注册。原 prefix76 在1500步时，固定last TRAIN TipValid仅20.11%，语义92.99%，不能视为充分训练的64父强基线。独立 prefix44/6000 已达到 TRAIN TipValid97.04%；本实验检验增加父场景后，在约相同的每输入训练曝光下能否改善拟合与开发集表现。它是普通基线的收敛控制，不是新方法。12000上限在本次训练前固定，不根据DEV延长。

数据仍为已封存 prefix76：64请求TRAIN父、63实际可观测父、189输入、1108已知正参考，与原12DEV父/36输入。缺失父283220的3输入及27未尝试槽保持在原分母；不重抽或替换。沿用已经核验的真实冻结Qwen缓存，新增Qwen编码0，forward输入白名单不变。相同seed0、batch32、K4、H24、lr3e-4及恒定LambdaLR=1，普通架构、饱和正例匹配、端点辅助、每250步评价、评分规则和阈值全部不变。

本次只新增 `train_observed_two_row_convergence64.py` 及独立配置、launcher、测试和本协议。原 prefix44/6000 五文件、base训练循环、历史driver、scaling与collector均不修改。作用域adapter只把旧driver传入的1500改为12000，并追加协议/source哈希；退出与异常都恢复原函数。旧selection的1500字段仅保留为数据和历史配方身份；新checkpoint明确保存实际steps12000，不能把旧selection当实际预算。

## 两个分别启动的阶段

先在全新 `runs/observed_two_row_prefix76_convergence_v1` 从相同初始化训练，使用12000配置但stop_after1500；不能复制原权重制造复现。原1500参考位于 `runs/observed_two_row_prefix76_v1/peak_seed0`，固定last SHA为 `38df15db97342ca37fdd7e8bd5bc8cf5a6d4e5af5f66e6e453bf72552e667721`，初始权重SHA为 `7f81eba70dfcef2a3df19ebd5661ace3881abd552a1f2d4614cf490bf153bc13`。export、原config、原summary也逐一固定SHA，不从更新结果替换来源。

对原last与新1500状态检查所有必需键：model、optimizer、scheduler、显式scaler(None)、完整RNG、sampler、抽样index-chain、step、exposures、best、history及关键训练config。采用零容差严格值相等；两侧缺键不能判相等。只忽略总步数、路径、stop/resume字段、计时与checkpoint容器字节差异。失败即留档，不自动更换seed、拷贝权重或放宽门槛。

通过后保存stage1500的last/best、配置、历史、来源与状态，以及完整审计。root读取proof后另行启动finish，从新跑出的1500状态继续到12000。两个stage均由record_job保存实际命令、PID、日志、退出码、来源及恢复命令；不会自动启动finish。跨source、policy、数据、参考SHA的恢复拒绝。

已到1500但封存中断时，显式resume仅补齐一致的snapshot/audit，不再优化或forward。已有12000 checkpoint时不再调用ordinary/base训练入口；必须先验证完整ordinary完成receipt、summary、1536000预算与全部原预测池SHA，缺失则fail-closed，保留原池交人工封存，不能自动重做末端评价。最后TRAIN诊断先写staging与189请求预算，完成后以SHA receipt封存并原子发布。密封池可零forward恢复外层元数据；未密封的中断保留0–189实际请求上界及756完整路径状态预算，拒绝自动重试。此边界不宣称支持部分推理无损续跑。

## 实际预算及比较口径

| 成本项目 | 优化步 | 输入抽样 | 完整候选路径状态 |
|---|---:|---:|---:|
| 原prefix76参考，原样保留且不再次计入新run | 1500 | 48000 | 192000 |
| 新run精确复现前缀 | 1500 | 48000 | 192000 |
| 新run门禁通过后续训 | 10500 | 336000 | 1344000 |
| 新run累计 | **12000** | **384000** | **1536000** |

receipt从固定steps、batch、K计算这些字段；测试检查阶段和累计恒等式。完成时还核对实际stream batches/draws、checkpoint/summary exposures、初始化，以及完整250..12000开发评价history，不能只相信配置声明。

prefix44/6000使用192000/93=2064.5161次抽样/实际TRAIN输入；本次为384000/189=2031.7460，即前者的62/63，约低1.59%。这是近似每输入曝光对齐，**不是精确对齐，也不是等总训练预算**；本次总draws/pathstates均为其2倍，父场景数和抽样序列不同。DEV选择机会为48对24；相对于原prefix76/1500则为48对6及8倍总训练曝光。所有比较同时列best和固定last，不能将额外优化或更多开发选择归为机制收益。

保留原loop的best TRAIN、best DEV、固定last DEV以及checkpoint/全部历史。12000完成后另外CPU生成一次固定last TRAIN池，覆盖全部189实际输入：**189额外请求、756完整路径状态**，不优化、不重选best、不做DEV或Qwen请求。保留无参考样本的生成与无定义参考误差。该诊断只回答训练拟合，不能当真实机器人执行成功或完整几何有效证据。

base elapsed、driver events与record_job外层墙钟是嵌套计时，不相加；新增初始化、比较、封存和最后TRAIN诊断成本如实报告。现有缓存编码成本列为共同前置。历史6000或1500在线时延不能复制为新12000实测；本协议不含新的在线Qwen时延测量。

## 运行与停止

只在新source冻结、服务器Torch测试和bash语法检查通过后，由root以CPU0单核、原GPU1/35%授权执行。目前无本实验结果。

```bash
SRC=/home/wzy/dpvlm/route_set_v1/research_v2/releases/REVISION
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_convergence64_v1.sh" REVISION stage1500
# root读取本run的stage1500_audit.json，全字段严格一致后另行启动：
taskset -c 0 bash "$SRC/scripts/launch_observed_two_row_convergence64_v1.sh" REVISION finish
# 实际中断且已核验PID/状态后，第三参数resume只恢复对应阶段。
```

到12000即停止，不自动延长、不新增机制或换数据。若充分拟合而DEV仍差，只能说明当前分布/表示/学习机制的泛化仍待改进；若TRAIN仍欠拟合，保留负结果，另作有证据的下一决定。此次不打开DEV_SCORE、CALIBRATION或TEST_LOCKED原始数据。
