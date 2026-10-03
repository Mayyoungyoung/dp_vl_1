# Composite108 观测独立/集合扩散：普通强基线设计附录

2026-10-03。**现进入实现准备，尚未server运行、训练或评估。** root已批准并行实现普通强基线；实际GPU实验仍须当前frozen/LoRA配对结果审查后独立启动。它补齐观测层强基线，不是核心方法，也不能由胜出反推“确定性MSE在取错误路线均值”。旧controlled源码/结果、全部观测历史实现保持原样。

## 固定问题与对照

- 数据只用已冻结composite108：请求96 TRAIN父、实际95父/285输入、1663条正参考；原12 DEV父/36输入/205参考。旧283220三条缺图保持缺失；extension新DEV禁止读取。所有unknown类型正参考保留。
- 冻结官方同revision的Qwen，原4096维mean+last缓存；RGB-D、相机、current与原ordinary相同。全部cache/source/role/hash门禁复用composite的实际身份。不能使用真实终点、箱体、guide/mode、未来状态作为条件。
- 从头seed0，两臂independent/set；同观测几何conditioner、输出表示和全部共享初始张量。只切换跨候选通信，独立臂的inactive attention参数不计active参数。没有额外候选query、slot embedding或类型标签。
- K4/H24，宽128、深2、point64、stride2、peak anchor、surface endpoint bound .05m，与ordinary的共有模块配置一致。共有模块从原ordinary seed0初始化重建并核hash，**不加载其训练后的权重**。噪声和父采样不受模型初始化消耗随机数影响。
- 每臂12000×32×4 = **1,536,000梯度路径槽**、384,000输入抽样、每250步一次旧DEV，共48个选模机会。AdamW constant lr3e-4，沿用原ordinary weight decay1e-4 / global gradient clip1.0并从config/source冻结，不新扫学习率。
- 唯一已存同曝光主控制为composite108 ordinary12000。common-head frozen/LoRA是额外3000步且不同优化器起点，单列，不和fresh12k混称同训练成本。native100k仅提供几何/覆盖参照：其定位训练人口及搜索成本不同，不称同训练信息。

## 输出表示：端点与事件必须生成

不能直接调用 `routeset/common.py::encode_paths/decode_paths`：它用真实start-goal桥固定两端。不能直接调用controlled `PairedTrainingStream`：它过滤 `modes<0` 并按模式采样。`RouteDenoiser` 当前仅接受 `[B,K,H-2,3]`，也不能假称它已支持观测终点/事件。

新独立wrapper沿用 `ObservedGeometryEncoder` 和普通头的feature/state条件编码。每次请求只编码一次观测几何，得到context、预测anchor；训练时其新初始化参数全部可训练。干净扩散状态定义为除已知起点外的23点：

`y0[...,t,:3] = reference_xyz[...,t+1,:] - observed_current_xyz`

`y0[...,t,3] = 0.2 * reference_open[...,t+1]`

单位为米，事件按既有0.2缩放，不另用测试统计归一化。**最后一点和事件均在噪声状态中**；已知起点及当前open最后由caller从观测拼回。训练noisy target是标准去噪输入，不是可缓存的生成条件；推理状态完全来自高斯噪声，不注入真实路线或真实终点。

采用普通**x0 prediction**，使每一步干净预测使用与ordinary相同的输出域：22个中间点由current→预测anchor的参考线加自由残差，最后一点为 `anchor + .05*tanh(raw_endpoint)`，open为`sigmoid(raw_event)`。模型将这些完整预测变成上述current-relative状态返回。预测anchor来自当前RGB-D与Qwen，不固定为真终点，也不把grounding的训练标签缓存进condition。这个端点结构是网络参数化，不增加碰撞修复、终点吸附或候选筛除。主损失为采样正例对应的全部23×4状态x0 MSE；附加原有endpoint-attention监督，沿用grounding_weight .02 / sigma .025及原全部正例支持构造。它的监督信息不多于ordinary。

共享去噪器可复用已有 `_RouteBlock` / FourierTimeEmbedding概念，但新增92维route输入/输出，不改历史类。independent中每个token仅依赖自身noise+共有context；set仅启用候选间attention。noise per-path独立，set训练目标为随机排列的正参考子集，无固定左右上专家。

x0是标准参数化，官方 [improved-diffusion代码](https://raw.githubusercontent.com/openai/improved-diffusion/main/improved_diffusion/gaussian_diffusion.py) 已有 `START_X`、对应MSE target及DDIM。此处选择它是为保留相同有界端点和事件接口；不是由controlled的v结果证明它更优，也不是创新。若实际denoising不收敛，不能据这一次默认参数化宣称扩散无效。

## 正例、时间与噪声流

`draw(batch=32,K=4)`从同固定285输入人口抽样，与原ordinary同ID顺序和抽样分布；保存实际ID链，而非仅报seed。对于该输入的全部R条有效采集正参考：R≥4时uniform without replacement取4条；R<4时先包含每条一次，再uniform with replacement补到4，最后uniform permutation。R=0不得假造正例；当前封存数据若与预期不符则拒绝启动。unknown不排除、不合并、不判负。

两臂共用独立的parent/target/time/noise RNG与预定义种子，逐step记录所选ID、reference index、排列、t和epsilon的滚动SHA；t按输入uniform取0..99，同输入K条共用t、各自独立epsilon。模型内部RNG另存。两臂真实stream必须逐项相同，resume含所有stream state及滚动digest。

**目标集合不是iid**：R≥4的不放回与R<4的覆盖后补齐，引入正例间依赖；两臂都见同一集合，set可以利用这个依赖，independent只学习其边缘。需报告R分层及实际重复target数。ordinary的full-positive saturation每次可访问全部R参考并匹配，扩散每次采样4条；1.536M梯度路径槽相同不等于参考监督访问完全相同。分别记录all-positive pool访问量、采样target总数与逐参考频次，不把此比较当单一损失因果实验。

## 100-step cosine训练噪声与40-step DDIM

沿用已经审计过的100步cosine beta数值构造，保存全部beta/alpha hash；新增schedule与历史epsilon/v schedule隔离。预先固定40个从99到0的rounded linspace索引并确认无重复。每步以模型x0预测计算

`eps_hat = (y_t - sqrt(alpha_bar_t) * x0_hat) / sqrt(1-alpha_bar_t)`

`y_prev = sqrt(alpha_bar_prev)*x0_hat + sqrt(1-alpha_bar_prev)*eps_hat`

DDIM eta0；最后alpha_bar_prev=1，直接返回第40次调用的x0预测，不增加第41次cleaning。没有PG、碰撞loss/引导、额外通用坐标clip、搜索、重试、隐藏proposal或top-M筛选。每个请求始终4槽，NaN/格式失败/重复照常计入。40次中间路径状态也需披露，不能说每请求只有一次网络生成成本。

## 选择、收敛与实际预算

48次DEV选择各使用同一固定独立evaluation noise stream/seed，independent与set匹配，与TRAIN随机流隔离；使用原ordinary同一预声明selection metric/tie规则，不按去噪loss挑模型替代路线质量。每次保存全部36×4池、逐场景检查、checkpoint SHA及noise SHA。原Tip2cm/目标3cm/事件/两排类型函数保持原样，生成池封存后再读验收标签。

最终best与last分别做3个独立采样repeat，逐repeat评价后平均，禁止合并成K12。已有该checkpoint的48次选择池可作repeat0并原字节引用；仅追加repeat1/2，best=last时复用同一checkpoint的池。选择仍是单个固定noise realization，另两repeat不能反过来重选step。每个checkpoint完整报告Valid、Any、UniqueClassified、ReferenceCoverage、unknown、重复、语义×净空四格、长度/事件及父配对差。

按B1逐请求计数，单臂48次DEV为1728请求、6912完整候选、**69120 denoiser调用**；已有repeat0之外，两个不同checkpoint的另两repeat上限144请求、576候选、5760 denoiser调用。训练每步一次批量denoiser，共12000次/1.536M noised paths。独立臂不把B×K拆成4次额外编码；两臂每请求观测几何只编码一次。fixed-last285 TRAIN另行stage（至多285请求/1140候选/11400 denoiser调用），不自动加入训练启动。微步/失败调用由实际journal计账，不能只用上述计划数。

reasonably trained至少要求：记录48点训练与DEV趋势、最后阶段波动、固定少量TRAIN按t分段的x0坐标/端点/事件误差、真实采样的终点/碰撞失败；诊断次数与成本单列。loss有限或跑满12k不是收敛证明。若仍明显未收敛、错误参数化或条件失效，先排工程问题，保留失败；root再决定是否有理由做双方共同的有上限修复，不自动延长或换种子。没有合理收敛证据时不能把它写成已战胜的强基线。

计算方面，1.536M曝光一致不代表FLOPs/GPU小时或推理时间一致。保存train/geometry/denoise/check/startup、峰值显存与全部evaluation成本；先报告cached-condition sampler延迟。真实在线时延必须另做逐请求Qwen+RGB-D+40denoise+封存+check，不能沿用ordinary的约74ms或把缓存吞吐冒充E2E。固定K与固定时间比较分列；未测真实时间不得补估成绩。

## 最小接口与实现门禁

预期新文件边界：独立观测denoiser/schedule/stream模块、独立trainer/evaluator入口、配置、测试、正式协议；复用原composite verifier/数据读入/原two-row检查，局部适配恢复后还原。旧普通/controlled类和在跑frozen/LoRA source不改。

本轮核心实现接口（trainer由独立代理负责）：

- `ObservedRouteDiffusion(set_attention, seed=0)` 在独立CPU RNG上下文重建原ordinary完整初始化，核原模型SHA `7f81…bc13` 和原CPU RNG SHA `c35b…a154`；共有geometry/feature/state逐张量再核。新模块独立固定seed1000000，恢复调用者RNG。没有加载训练权重；独立臂attention allocated但requires_grad=False、单列inactive参数，其余active均可训练。
- `encode_observation(observation_tensors) -> {context, anchor_xyz, attention, current}`，仅接受`features/current/world_xyz/rgb/uv/depth/valid_mask`七键；不接收监督/验收结构。
- `forward_x0(noisy[B,4,23,4], t_integer[B], encoded) -> clean_state`；same shape，端点/事件完整参与。`targets_to_state(paths,events,current)`只供训练监督，生成不调用。
- `ObservedX0Schedule.q_sample(x0,t,epsilon)`必须显式传noise；`sample(model,encoded,initial_noise,before_denoise=None) -> paths[B,4,24,3], open[B,4,24], budget_receipt`固定40调用。可选hook紧邻每次forward之前，供driver持久issue记账；异常不重试。`audit()`含beta、alpha、40indices及各SHA。
- `PairedPositiveStream(data,train_ids,seed=0).draw(batch_size)`返回`indices/reference_indices/reference_permutations/reference_counts/timesteps/epsilon`。`audit/state_dict/load_state_dict`保存逐参考频次、累计输入/目标/全正参考池访问与实际共同chain；四独立RNG seed为100000/200000/300000/400000。parent选择与原ordinary `np.default_rng(100000).choice`逐项相同，另存原index-chain格式；正例/time/noise不依赖模型RNG。训练checkpoint还由driver保存model、AdamW、scheduler、所有RNG、logical step、实际issue journal、source/data/cache身份，支持完整step行政暂停；硬崩溃未封存调用拒静默重放。
- 唯一配置`configs/observed_two_row_diffusion_v1.json`固定实际export/quality/cache SHA、eval noise seeds300000/300001/300002与25step checkpoint。独立`denoising-diagnostic`仅last12000首6TRAIN、t=0/25/50/75/99：6geometry+30denoise批/120中间路径状态、0optimizer/0DEV；不是正常生成，不自动执行。

必要测试关注点：

1. 真实/伪造GT终点、模式和验收标签不进入`encode_observation`；仅修改supervision不会改变相同噪声的inference条件。training teacher tensor不能被写入缓存。
2. 起点精确、surface endpoint±.05、event范围；改变noisy endpoint/event会进入网络，而不是被真值硬钳制。oracle x0的t0/t99公式与最终40调用严格正确。
3. independent跨候选不依赖；set置换等变；identity/共有初始化张量hash。noisy正例中的unknown/R<4/R≥4与空正例拒绝覆盖。
4. 两臂实际draw/target/t/epsilon逐字节一致；4-step连续与2-step暂停恢复的真Torch参数/Adam/RNG/stream/journal精确一致；拒绝跨arm/数据/采样/schedule恢复。
5. 12或48池封存、失败4槽、separate-repeat不合池、旧two-row验收与全部逐场景指标一致；sampler仅40次而非40+1，观测encoder一次、无额外K。

没有额外模块作为“若失败再加”的默认分支。若独立/集合仅展示标准分布建模收益，结论就是加强普通基线；真正核心机制仍须由其剩余失效给出新的、可被同信息强对照否定的假设。
