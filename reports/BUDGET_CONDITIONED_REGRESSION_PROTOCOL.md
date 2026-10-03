# 真正 K 条件普通集合回归 v1

状态：独立实现/测试准备；尚未服务器训练。是强普通基线，不是新核心方法。此前 [TRAIN768 指派审计](budget_assignment_conflicts_v1/RESULTS.md)的100冲突全部是固定K4合法重复落在前两槽、被K2注入指派强制换标签，不能作为联合指派机制收益依据。历史模型/训练/检查器源码均不改。

## 固定数据、输入与模型

`configs/budget_conditioned_regression_v1.json` 固定 development.npz SHA `f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057`，原768 TRAIN/128 DEV_MODEL完整父ID；先核角色/身份，再读取几何与正参考。禁止把含锁定角色的全数据archive传入，也不使用新的观测采集资料。

输入仍是受控层34维真实起终点/墙与开口几何，加单一 `log2(K)/3` 条件。现有条件MLP学习其embedding；K=1/2/4/8每次真实进入模型。现有SetRegressor最大8个查询参数，每次只取当前K个构造tokens，经全部block/输出层，只产生K条H24路线。不是先算8后截断，不用类型、真实路径、合法标签或R作为条件；已知起终点硬约束仍仅限这一受控层。

width192/depth3/heads4，新增condition_dim35和8查询需要fresh初始化。不能宣称与旧固定K4初始权重相同；未来配对须复用本次新初始SHA、相同parent/K流。没有补全记忆/顺序模型/扩散/几何修复/重排/采样过滤。

## 优化、选择与真实成本

- seed0，parent sampler独立seed100000，batch64，3000步，K循环1/2/4/8各750步。AdamW lr3e-4/weight_decay1e-4、constant LambdaLR、clip1、FP32，无scaler（检查点显式None）。完整正参考池和原 `positive_assignment_loss(...,'saturation',...)`：R>=K注入匹配，R<K覆盖全部已知正例且允许额外有效重复，无absence标签。
- 192000训练请求，按K请求均48000；完整输出槽分别48000/96000/192000/384000，合计 **720000**；计算包含16个pad参考的11520000成对成本。不是旧K4的768000槽，也不隐藏多查询算量。
- 每500步在原128 DEV分别真实生成四K，共6次选择。单一分数为 `mean_K(UniqueValid/K + 0.05*Valid)`，严格大于才更新best、平局保留最早。选择共3072请求、11520完整槽；固定last无条件保留。它是新预登记选择配方，不改历史checker/结果，不各K分别挑最有利checkpoint。
- 每次选择先落盘完整原候选，再做标签评价，保存原池、每父valid/type/长度/碰撞/coverage及整体指标、模型SHA与receipt。最终best/last按选中step和权重SHA核验并引用这些原池，**0额外最终DEV forward**。若best==last，完全同一池、同计时结果hash复用。
- 对两个不同最终checkpoint，逐K用固定首TRAIN父 `multigate_v1_TRAIN_00000` 做2warm+10timed；48请求/180槽每checkpoint，最多96请求/360槽。warm、timed及被拒绝调用均记预算；保留所有生成坐标。计时含单请求CPU条件传移、模型、decode、CPU回传和相同受控exact validity检查；不含离线参考覆盖标签计算。没有Qwen/VLM，不以batch吞吐冒称观测端到端延迟。
- 总生成槽上限 **731880**=720000+11520+360；best==last时731700。完整轨迹数量与前向batch调用不同：每选择K用4个batch，每选择点16个head forward；生成槽按实际128K计算，不称一次网络调用等于一路。

每优化步先持久化issued的step/K/父indices，成功后记录loss/梯度范数；parent/K hash链按原int64字节累计，reference_pool_access实记。正参考访问不是新路径生成。所有选择/计时先记录issued，生成成功封存后closed，失败不抹去分母。未来候选不能只匹配720000数字而抽样顺序、K链、目标访问不同。

## 恢复、封存与边界

每250步及显式 `--stop-after` 保存last，检查点包含model/optimizer/scheduler/scaler/RNG(NumPy/Python/Torch/CUDA)/sampler/globalstep/最近200loss/完整选择history/最好step、实际曝光与reference计数、draw/K链和ledger SHA。显式暂停不增加DEV调用或选择机会。生产配置/source字节/CODE_COMMIT/初始modelSHA/数据hash严格恢复；active.lock不自动删除他人或存疑旧PID。

**故障恢复边界明确：**若检查点后已有issued但尚未被该检查点封存的优化/评价调用，恢复拒绝，保留原ledger和失败；不会静默重放超预算，也不承诺任意时间断电都可无损续算。完成3000步后的独立计时可复用已完整sealed的池；存在未封存timing目录则fail-closed，不能删除后再次生成而不计费。已completed运行再次 `--resume` 只核hash、不构造模型/forward、不改原summary。

各进程最终保留独立 `sessions/*.json` 与GPU-reserved-hours；summary的训练+选择耗时跨已保存暂停累计，finalization另列。内层sessions各自求和、外层record_job各自求和并分别报告；**外层包含内层，总成本只取一个层级，禁止两者相加**。不要把仅最后进程的峰值冒称多session历史最大；每session状态分别保存显存峰值。

GPU门禁不仅读取环境声明。root已在实际Torch2.4.1+cu121做0模型元数据核验：`get_device_properties(0)`不含uuid，因此旧属性门禁未投入训练即更正。现在 `CUDA_VISIBLE_DEVICES` 必须是完整授权UUID；经 `libcuda.so.1` 的公开 `cuInit(0)`、`cuDeviceGetCount`、`cuDeviceGet(ordinal=0)`、`cuDeviceGetUuid` 读取16字节UUID，必须恰好一个可见设备且真实UUID一致。每个CUresult严格核0；不调用context创建、内存分配或kernel API。训练入口随后核Torch visiblecount=1、读取name/total memory，并设置device0的35% allocator cap。模型/数据/优化/K预算未改。

已核 [CUDA12.1官方Device API](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-driver-api/group__CUDA__DEVICE.html) 的C签名和UUID16字节、[初始化API](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-driver-api/group__CUDA__INITIALIZE.html) 的flags=0，以及 [CUDA环境变量](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html) 的UUID可见设备选择。ctypes显式声明指针/整型参数与返回类型；纯mock覆盖写入真实16字节、错UUID、错误返回码、多个设备与ordinal环境拒绝，不把mock当真实服务器验证。

冻结后root可先独立执行元数据入口，**不import Torch、不读数据、不加载模型，也不设置allocator cap**；结果保存成功/失败、源/config SHA、实际四API返回、elapsed/0模型计数。此检查成功后仍需训练入口实际设置35% cap：

```bash
CUDA_VISIBLE_DEVICES=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab \
RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab \
taskset -c <allocated_core> .venv/bin/python -m scripts.train_budget_conditioned_regression \
  --config configs/budget_conditioned_regression_v1.json \
  --probe-cuda-metadata --output <fresh_metadata_receipt.json>
```

## 测试和执行授权

纯测试覆盖真实K日程/预算恒等式、选择归一化、data角色门禁、stream顺序、ledger尾部拒绝、封存更改/零重放。Torch测试覆盖实际block只见K tokens、显式条件、八查询梯度、共享优化step连续vs暂停恢复全参数/Adam/scheduler/RNG/stream逐位一致，以及完整小型driver的选模原池复用和completed零模型重放。当地Anaconda缺Torch时标skip，不称实际通过；root冻结后在已授权环境补真实CPU检查。

待root冻结并明确启动，不能仅因源码写完自动运行。建议先2步暂停、核账本/最后状态，再root单独resume3000（2步不是独立结果实验）。完整训练参考命令由root放入不可变wrapper并用record_job记录，CPU1/GPU1/35%不变：

```bash
CUDA_VISIBLE_DEVICES=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab \
CODE_COMMIT=<frozen_sha> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c <allocated_core> .venv/bin/python -m scripts.train_budget_conditioned_regression \
  --config configs/budget_conditioned_regression_v1.json \
  --output <fresh_run_directory> --device cuda --stop-after 2
# 此进程真实exit0且状态核验后，另启动同源同配置 --resume（去掉stop-after）。
```

该基线回答同一网络不同真实候选预算的质量/覆盖/成本曲线。即使覆盖提升或K8达到上限，也不能据此宣称预算条件本身新颖、机器人观测有效或论文核心成立。
