# 同一 composite108 末步头：frozen 与末两层 LoRA 配对续训草案

状态：**仅协议草案，未实现本轮训练器，未启动训练。** 先决条件是 [TRAIN6 前缀回放技术探针](TWO_ROW_QWEN_PREFIX_REPLAY_PROTOCOL.md) 的真实模型、梯度、更新和冻结权重门禁全部通过；本卡不以代码或单元测试代替该实测。root 读到实际成本和门禁后，再固定协议/source 与运行队列。这里是普通骨干适配基线，不是论文核心机制。

## 可检验问题与唯一配对

问题是：在相同、已经训练的普通几何/路线头上，再给相同数据与优化曝光，允许 Qwen 末两层进行小幅任务适配，是否比继续训练冻结 Qwen 的普通头更有效？这能检验“冻结表示限制当前任务学习”的一个有限版本；不能证明全部 VLM 微调有用或无用，也不能将 LoRA 自身写成新方法。

共同初始头固定为 composite108 的 **last12000**，不用 best6000：远端 `runs/observed_two_row_composite108_v1/peak_seed0/last.pt`，SHA256 `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`。两臂先复制完整实际头张量，再核逐参数 SHA 相同；所有几何编码、query、anchor、融合、直接条件分支、路线解码和事件参数继续训练。共有头参数1,231,965，不增加/删除路线模块。

| 项目 | frozen 续训 | LoRA 续训 |
|---|---|---|
| Qwen | 全冻结 | 仅语言层26/27的 q_proj、v_proj 训练 LoRA |
| 新增适配参数 | 0 | rank8、alpha16；8个A/B张量，共114688参数 |
| 头初始化 | 同一 last12000 全状态权重 | 逐字节相同 |
| 优化器初态 | 独立新 AdamW 状态 | 独立新 AdamW 状态 |
| 头学习率/weight decay | 常量3e-4 / 1e-4 | 相同 |
| LoRA学习率/weight decay | 不适用 | **常量1e-5 / 0** |
| 追加优化步 | 3000 | 3000 |
| 批处理 | 串行B1，累积32后更新 | 相同，原长度，不padding |
| 候选/路径表示 | K4、H24、原saturation与事件损失 | 相同 |
| 端点辅助 | 原weight .02、sigma .025、peak/surface ±.05m | 相同 |
| 选模 | 每250追加步一次，固定原DEV36 | 相同 |

共同基础 Qwen 为实际 pinned `Qwen/Qwen3-VL-2B-Instruct`，revision/processor `89644892e4d85e24eaac8bacfd4f463576704203`、transformers4.57.1、Torch2.4.1、原SDPA/bfloat16、原min/max pixels与FP32 masked mean+last pooling4096维。LoRA A 固定独立初始化种子，B 为零；之后重置两臂的共有训练 RNG，以免 adapter 初始化改变头/抽样随机流。vision、embedding、语言0..25、末两层基础权重和 final norm 均冻结。模型 eval 模式不关闭 LoRA autograd。

**学习率只选1e-5，不扫参。** 理由是保留技术探针已经锁定的数值条件，并对一个拟合过的头所依赖的预训练表示作保守适配。[Qwen官方 LoRA 脚本](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/scripts/sft_30a3b_lora.sh)也列1e-5；该示例是30B MoE/SFT、另有cosine/warmup，不能证明此2B路线任务的最优学习率，本实验不移植其调度。历史 [RGB-D online LoRA配置](observed_online_geometry_v1/lora_seed0/config.json)采用1e-4，虽有真实训练与参数更新，却未建立稳定质量收益；任务、头、曝光均不同，不能从历史失败归因学习率。本轮不用其结果挑选多个学习率，也不因 DEV 结果换率重跑。

## 数据、抽样和梯度语义

只使用已封存 composite108：旧64 TRAIN + 新32 TRAIN 的288请求输入，实际285可读TRAIN输入/1663正参考；原12 DEV父的36输入/205参考。旧3缺失输入保留记录、不补样本，新 extension DEV32、DEV_SCORE、CALIBRATION、LOCKED 不读。双corpus机械门禁、export/source/hash、完整质量audit与输入白名单必须在加载训练内容之前复核；已有父闭合不表示新权限。

为两臂预先固定同一按实际285输入均匀有放回抽样的 `3000 × 32` ID序列，并封 SHA；seed0、sampler seed100000，独立于模型参数与评价。每个逻辑步一次抽32个ID，按同顺序串行执行。目标匹配所用独立 NumPy RNG也从同一状态开始并保存；不得按当前损失或成功率另采。两臂每步喂相同完整正参考集合；匹配结果可随模型预测不同，但不能给某臂额外类型标签、路径修复或筛选。

每个B1原损失除以32后 backward，累计满32才统一clip并更新一次；不按token长度、参考条数或某目标损失另作权重。沿用global grad-norm clip1，记录裁剪前的头/adapter范数及实际缩放；LoRA臂global集合多了adapter，裁剪耦合应在报告中披露，不能声称只有表示值变化而没有优化器差异。AdamW betas/eps固定为已有实现默认(.9,.999)/1e-8。无warmup、重启、学习率衰减或额外辅助损失。

此处“续训”只指从共有头权重继续优化。**两臂均不继承原头 Adam moments**，避免仅LoRA为新状态、对照却续用旧动量。原last中的optimizer/RNG等作为来源封存；本轮自己的新optimizer和新3000步序列构成独立实验。两臂不是原constant12000的无缝第12001步，不将 reset 的效果归入LoRA。

## 前缀缓存的合法范围与开销

按技术探针的官方 layer26 pre-hook 边界缓存冻结前缀：图像/指令0..25层后的 hidden、实际cos/sin、因果mask、position/cache-position、原kwargs与pooling mask。固定B1、`use_cache=False`、past KV=None。DeepStack早期加法已经在前缀中，不重新注入；直接调用原26/27层与final norm。禁止将原最终4096特征作为LoRA更新后的条件，禁止缓存已更新尾层的训练输出。

本轮主配对建议**两臂均从同一冻结前缀串行回放末两层**：frozen尾部no_grad，LoRA尾部保留autograd，每次训练微样本/评价都重算；训练中的所有点特征、attention、anchor与头也重新计算。这样两臂保持相同B1张量形状与输入流程，实际反向代价不同照实报告。冻结臂最终特征在数学上可缓存，但本轮先不额外切换执行路径，避免正式比较中再引入一套未经共同验证的数值实现。

正式缓存最多321个完整Qwen输入捕获（285TRAIN+36旧DEV），最多321次零LoRA尾回放逐值核对；两臂共享一次准备成本，不重复收费。每项核对官方特征、回放特征和现有321份有效缓存的身份/数组，失败保持原证据并停止，不扩容差或重新采图。新前缀缓存键必须绑定原输入/processor/model/source SHA、cut layer、dtype/backend、实际张量shape及每项payload hash。不要把技术探针的6条有限证明宣称覆盖了正式321输入。正式构建属于独立授权阶段，不由本草案启动。

可直接缓存并复用的还有不学习的当前RGB/XYZ/UV/depth/mask；不得缓存可训练的几何特征。前缀可按原dtype常驻CPU RAM并按ID搬运；先记录实际payload字节数和主机内存，再决定是否装载全部。量级计算：仅hidden为 `2 × Σ_i L_i × 2048` 字节；若平均1000token，则约1.225GiB（321项），另加cos/sin/mask等，不能只报hidden作为总占用。保留落盘hash，训练内避免每微样本重复解压、重新tokenize或计算全部模型hash。

## 曝光、评价机会与成本账本

每臂追加 `3000 × 32 = 96000` 观察微样本、`96000 × 4 = 384000` 路径状态、3000次optimizer更新，平均每实际TRAIN输入336.842次抽样。共用原预训练1536000路径状态单列；每臂逻辑累计1920000，不把共有预训练在两臂汇总时重复收费。两臂追加总微样本192000、路径状态768000。

每250步评价一次是合理的预声明折中：只有12个同预算选模机会，保留原任务评价口径且评价量远小于训练。step0用固定来源已保存的DEV last池作基准，不参与新增best选择；正式初始数值一致性另由缓存/头门禁证明。新增best仅在250..3000的12个点按原 `UniqueClassifiedTipValidAtK + .05 × TipValidAtK`、相同tie规则选择，并保留固定last3000。不能从12个点再按另一指标挑checkpoint。

| 每臂实际计划 | 请求/微样本 | K4路径状态 | 说明 |
|---|---:|---:|---|
| 训练 | 96000 | 384000 | 32次backward/逻辑step，3000次更新 |
| 12次DEV选择 | 432 | 1728 | 每次36，全部失败保留 |
| fixed-last TRAIN | 285 | 1140 | 旧189与新96分别报告 |
| 额外best/last DEV | 0 | 0 | 已保存每个选择点的池，best与step3000直接索引，不重推 |
| 合计头调用 | **96717** | **386868** | 不含共有前缀准备和技术探针 |

12个DEV预测池封存后再读标签检查；不修复、重试、淘汰或补充候选。训练loss当然读取TRAIN正参考，但不能进入生成条件。fixed-last TRAIN与所有DEV保存逐候选指标、known/unknown/dup、语义×净空四格和逐父配对。没有新评分器，因此不补造SelectedValid。

**现有真实成本不能直接当回放速度。** 历史同GPU、RGB-D online普通基线是1000步×累积4=4000微样本：frozen训练body361.952746s，LoRA405.251897s，均另外报告setup；见各自原summary。这折算为含其评价等开销的90.49/101.31ms每训练抽样，并不是纯microstep计时。若机械外推到96000微样本，约2.413/2.702 GPUh，两臂约5.115GPUh，**只是旧全Qwen在线流程的量级提醒**；优化步/评价数增长比例不同，任务也不同，既不是预测上限，更不是当前回放实测。

真正估算须来自通过后的探针时间与正式前缀准备的实际字节/搬运量：

`T_pair = T_shared_capture_and_verify + 96000×(t_frozen_micro + t_lora_micro) + 3000×(t_opt_F+t_opt_L) + 717×(t_eval_F+t_eval_L) + T_checkpoint+T_startup+T_checks`。

探针只有两个B1更新，冷启动与hash/状态复制占比很大，必须拆开已有计时范围，不能将整个probe除以2或直接按28/2宣称加速。若其没有稳定的微步分项，本卡只给条件预算，不编造实际速度；root在正式启动前确定可持续时长。

| 某一臂平均训练微样本时间（情景，不是实测） | 仅96000微样本 | GPUh |
|---|---:|---:|
| 5ms | 480s / 8min | 0.1333 |
| 10ms | 960s / 16min | 0.2667 |
| 20ms | 1920s / 32min | 0.5333 |
| 40ms | 3840s / 64min | 1.0667 |

可减少开销而不改本轮含义：共享不可变前缀/原始几何CPU存储、保持原dtype/长度、B1即算即backward释放图、只在逻辑更新边界同步记录聚合统计、固定抽样序列下预取下一已指定payload；best直接引用已封存DEV池，不重复forward；保留小adapter/头/optimizer而不反复存2B基础模型。padding、跨样本attention、压缩/量化前缀、改变图像token预算、跳过尾层或减少累积数均不属于本草案优化，不能悄然采用。

训练gpu_hours记录模型加载、前缀搬运、所有训练/DEV/fixed-last、hash、checkpoint与退出全程；同时提供分项与GPU峰值，嵌套不相加。共享缓存构建费用和既有原321最终缓存费用分别列明，后者不重复计。此训练评价的prefix replay不是单请求端到端时延；如后续要质量—成本曲线，必须另行固定真实RGB/processor/全Qwen/几何/头/检查的在线请求协议，不能复用原74ms给新的LoRA权重。

## 恢复、预算与最低工程门禁

独立fresh输出、每臂专属lock，record_job包含run_id/PID/日志/退出码/实际命令及源SHA。启动后不改source或脚本。实现恢复要保存：完整头和adapter、每个AdamW参数组/状态、常量scheduler、Python/NumPy/Torch CPU/CUDA RNG、目标匹配RNG、sampler/固定draw序列位置、逻辑step、实际累计曝光、best分数与step、历史12点、输入/cache/source/hash、实际成本和请求journal。原Qwen资产只存索引。

常规checkpoint只在完成32次累积与一次更新之后原子写入，明确没有悬挂梯度；保存频率可预先定为每25个逻辑步，并保留最新少量、best与阶段终点。受控暂停先结束当前逻辑步再checkpoint。若要从累积中段精确恢复，必须另存已累积梯度、pending32个ID/当前micro位置与全部RNG；当前草案不默认具备此能力。崩溃发生在未封存区间时，journal保留已发微样本和上界，**不能静默重放后仍声称相同实际曝光**；由root明确恢复决定，已完成逻辑曝光与重复/未完成实际计算分开收费，配对预算若不再相同必须标明。

最低工程验证：小模型串行B1累积32的连续4步 vs 2步完整恢复+2步，逐参数、adapter、AdamW、scheduler、全部RNG、抽样链、loss历史与曝光严格一致；拒绝模式、lr/rank/alpha、共同初始头、数据/cache/source、累积数、步数或选模间隔变化的恢复。两臂在训练前的头hash/抽样序列/零B输出一致；没有新LoRA时不得把模型前向缓存复用为可训练特征。实际训练后核八个adapter张量的有限梯度与变化、所有冻结基础参数hash不变，以及头确实继续训练。第一步A零梯度是B零初始化的预期，不伪报“所有张量第一步均更新”。

失败包括cache等价/依赖/梯度/恢复门禁不通过、OOM、非有限输出、未完成训练；都保留原run和成本。不自动降像素/缩短序列、换种子、扩大预算或改变比较口径。

## 停止条件与可解释边界

本配对最多这一组frozen/LoRA追加3000步，串行使用GPU1、35%显存、CPU1；不同时训练，不用GPU空闲推导额外授权。先读技术probe结果再冻结正式实现，当前没有新的服务器任务。

实际研究报告同时给best与fixed-last、TRAIN公共189/新96与固定DEV36、父级配对和全部成本。若LoRA改善仅来自更多可训练参数、或只在TRAIN提高、或best微涨但last/覆盖/成本恶化，必须按证据描述；不能称核心机制成立。若这一个合理设置无收益，不继续扫lr/层数/rank/种子；保留最强普通基线，继续原研究主线。即使有收益，仍需独立多任务/多种子确认，不能把重复使用旧DEV升级为TEST。原末两层低秩适配失败也不能推导全模型适配或充分数据下的VLM能力无效。

本卡只新增此Markdown；现有训练器、collector、缓存器、模型、全局状态和结果表均未修改。引用的实际已完成成本与本轮条件估计已分开，未将文献示例或技术探针写成正式效果。
