# 观测扩散独立入口与实际计费协议

2026-10-03。实现准备；尚未进行服务器测试、训练或采样。它补充普通强基线，不是新核心机制。科学设定以 [TWO_ROW_OBSERVED_DIFFUSION_DESIGN.md](TWO_ROW_OBSERVED_DIFFUSION_DESIGN.md) 和 `configs/observed_two_row_diffusion_v1.json` 为准。

## 数据、条件和初始化

入口 `scripts/train_observed_two_row_diffusion.py` 只接受固定 composite108 export：285 TRAIN 输入/1663 正参考，旧36 DEV 输入/205正参考；请求288 TRAIN 输入中的三条缺观察保持缺失。未知类型正例全部保留。原96请求父、实际95训练父以及旧12 DEV 父不变；extension 新 DEV 不读取。

启动先核实际 export `04ba2729…74fdc`、TRAIN质量 `ef60c759…864d0`、Qwen缓存收据 `9f2ccf21…15cff`，复用原 composite verifier、全部缓存字节/revision/processor/input contract 和质量门。读取原 ordinary `composite_training_receipt.json` 的源码、数据与缓存身份逐项核验，**不加载其训练权重**。新源码、完整配置、实际数据/几何 fingerprint 和运行环境进入 checkpoint。前向只接严格七个观测张量，训练目标在独立loss函数中转换。

core重建原seed0普通初始化，逐张量核验共有geometry/feature/state权重。两臂共有初值相同；只有跨候选attention启用状态不同。独立臂 inactive attention参数不进入optimizer，单列allocated/active参数。其余新增参数可训练。各臂完整父抽样链在12000完成时还须等于已存ordinary实际链；正例子集/t/epsilon另有共同流哈希与逐参考计数，不能把监督访问量说成与ordinary saturation完全相同。

## 五种阶段完全独立

| `--stage` | 数据/权重 | 最大新增调用与候选预算 | 是否选模 |
|---|---|---|---|
| `train` | fresh seed0，12000×32 | 12000 geometry batches +12000 denoiser batches +12000 optimizer；1,536,000梯度路径槽。48×36 DEV另含1728 geometry、69120 denoiser、6912最终候选/276480去噪路径状态 | 原UniqueClassified+.05TipValid，250步一次，strict `>`保留更早平分 |
| `repeat1` | 已完成原best和last | 两个不同step上限72请求/288最终候选/2880denoiser；每请求geometry一次 | 否 |
| `repeat2` | 同上，独立噪声 | 同上 | 否 |
| `fixed-last-train` | 固定last12000，全部285 TRAIN | 285 geometry/11400denoiser/1140最终候选/45600去噪路径状态 | 否 |
| `denoising-diagnostic` | last12000，规范顺序前6 TRAIN | 6 geometry+30 denoiser batches/120中间路径状态，0optimizer/0DEV | 否 |

repeat0直接引用原48次选模中该checkpoint的池，不再生成。repeat1/2噪声分别固定300001/300002，repeat0固定300000；两臂/同阶段同输入噪声逐字节一致，不受训练RNG/检查点步数影响。best与last确为同一step和权重时复用该stage已有池，不能把K4三个重复合成K12，也不能根据补充repeat重选checkpoint。每次geometry一次、DDIM40次，没有第41次清理、候选筛除、修复或隐藏重采样。

最后一行是**真实正例加噪后的去噪技术诊断**，不是正常推理质量：每输入按既有参考索引前4条（不足则循环补齐），t固定0/25/50/75/99，噪声独立seed400000。保存所选参考索引、噪声hash、x0 MSE、坐标RMSE、端点和事件误差。不访问DEV，不增加训练，不暗加到285正常采样预算。

## 预测封存、失败与原评价

每个B1请求先记录geometry和每一个denoiser的durable issued账本，再执行调用。全部请求结束后先保存 `initial_noise.npz`、`predictions.npz` 与 `generation.json`；封存SHA之后才调用原 `observation_metrics`、原两排 `add_two_row_metrics` 和重索引语言控制。检查函数、2cm净空、3cm身份/目标、事件与类型标准不变，没有额外forward。

数据loader已为训练加载参考与监督metadata；这里的后置边界指**验收检查只能使用已封存预测**，不是声称此前从未读取任何label。模型的encode/sample参数仍只有观测张量与明确噪声。

完整NaN候选、重复、未知类型、失败都保留K4槽。调用抛异常时保留 `.staging/failure.json`、已发调用、所有请求形状的预测数组：失败/未尝试部分为NaN，绝不自动重试或替换；不出completed质量摘要。每个池保存完整逐请求40次时间索引、噪声hash、实际geometry/denoise时间，以及原指标/逐场景/语言重索引池。`pool_receipt.json`绑定全部文件、调用前后账本和模型张量hash；后处理不得改变预测或模型。

所记单请求延迟是**已缓存Qwen条件的sampler成本**，输入/geometry、40次去噪与后处理分列。几何预处理/数据验证启动成本另记；真实RGB→Qwen端到端时延未测，不能继承普通头时延。总体训练墙钟/GPU占用包含48次选择；候选数相同并不代表FLOPs、训练时间或端到端时间相同。

独立诊断的 `summary.elapsed_seconds` 是载入后的stage body；`status.process_elapsed_seconds` 和外层 `record_job` 才包含本次启动、数据/权重验证和加载。归档须分列嵌套成本，不能用body冒充完整pipeline或相加重复计费。

## 恢复与可核查调用

`last.pt`含model/AdamW/常数scheduler/所有Python、NumPy、Torch CPU/CUDA RNG/独立stream四种RNG、实际滚动hash、逐参考频次、global step、history/best、调用账本边界、费用与source/config。每25步、每次DEV前后保存；行政 `--stop-after N` 在完整更新和该步DEV之后保存paused，不写completed summary，不改变12000总预算。

硬崩溃后的未封存issued调用默认拒绝重放。唯一可恢复额外跨度是与当前checkpoint完全同模型、同step、同输入/噪声、且完整封存的DEV池；核所有40调用后可零forward接纳原结果、恢复原strict选模，并把未入checkpoint的池耗时加回累计。已完成阶段不能再次fresh/覆盖。48次选模池全部永久保留；权重保留best/last，checkpoint SHA最终绑定原池模型张量。

初始/恢复配置、arm、data/cache/source、schedule、stream任何变化均拒绝。外层 `record_job` 的每次pause/resume启动/退出成本也须保留；内部累计与外层墙钟是嵌套值不能相加。

## 冻结后的运行边界

root冻结全部core/stream/driver/tests/config后，先在私有环境实际运行 `tests/test_observed_route_diffusion.py`、`tests/test_observed_diffusion_stream.py`、`tests/test_observed_diffusion_driver.py` 及相关旧两排回归；当前本机无Torch，不能用skip替代真实恢复/梯度/预算通过。

实际每次只启动一个阶段、一个arm。由root生成不可变wrapper并用 `scripts.record_job --output <job_records> --run-id <unique> --resume-strategy flag|none -- <python> -m scripts.train_observed_two_row_diffusion ...` 包装；设置 `CODE_COMMIT=<完整已核验SHA>`、`PYTHONPATH=<该release>`、CPU单核affinity、GPU1/35%与已授权UUID。数据位置固定 `data/observation_two_row_composite108_v1`，质量来自 `runs/observed_two_row_composite108_preparation_v1/train_quality.json`，ordinary来源 `runs/observed_two_row_composite108_v1/peak_seed0`。输出为独立family，不改旧结果。

共同CLI：`--arm independent|set --data <固定data> --quality-audit <固定quality> --ordinary-run <原ordinary> --config <同release/configs/observed_two_row_diffusion_v1.json> --output <fresh stage>`。train支持明确`--resume`；所有非train阶段另需`--train-run <该臂completed输出>`。不提供自动下一阶段、不扫步数/噪声/sigma，不由本实现自行启动GPU。

本地首轮：11项pure通过，5项真实Torch待服务器；核心另有自己的测试计数。此记录仅证明本地已执行部分，尚无实际训练或方法优势。
