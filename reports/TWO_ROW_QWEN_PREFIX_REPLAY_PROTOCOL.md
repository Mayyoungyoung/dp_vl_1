# TRAIN6 冻结前缀／末两层 LoRA 技术核验 v1

状态：仅实现与预登记，尚未运行真实模型 probe。不是方法创新、路线质量结果或新的候选机制。现有训练器、Qwen 缓存器和模型源保持原字节。后续普通 frozen/LoRA 配对尚未实施。

## 固定输入、权重与实现依据

只读 composite108 已封存导出的前六行：旧 TRAIN 父 283200、283201，各三条真实同图不同目标指令。只在两父的 target0 上各做一个逻辑微更新；不重抽、不读任何 DEV、DEV_SCORE、CALIBRATION、LOCKED 原始样本。其余四条只用输入验证编码，不打开路线 NPZ。语义坐标、障碍、路线类型和未来路径从不进入 Qwen 或路径头条件；训练监督仅已记录路径／夹爪事件及既有端点辅助损失。

固定初始头为 composite108 原普通集合回归 last12000，checkpoint SHA256 `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`，Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`。路径头沿用 peak anchor、K4/H24、surface ±5cm、saturation、端点辅助权重 .02/sigma .025；所有头参数参与训练。两分支从完全相同的实际头张量开始，AdamW 新状态；头 lr3e-4，LoRA lr1e-5，均常量，梯度共同裁剪1。此两步只检查链路，不进行学习率或模型选择。

只读核对服务器 transformers4.57.1 实际 `modeling_qwen3_vl.py`，SHA `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da`：28层语言模型，vision DeepStack 加法发生在前3个语言层。缓存 layer26 **输入**已包含全部视觉和语言0..25计算。通过官方完整 `backbone.model` 的 layer26 pre-hook 捕获 hidden、实际 cos/sin、实际因果 mask（允许原调用为 None）、text position、cache position及原 kwargs；保留原 pooling mask。缓存写盘后重新加载，再直接调用原26、27层和原 final norm；禁止再调用整个 TextModel。固定 `use_cache=False`、past KV=None、串行B1、原SDPA/backend/dtype。模型 `.eval()` 不关闭梯度，前缀捕获用 `no_grad`，不用可能生成不能保存给反向传播张量的 `inference_mode`。

视觉、embedding、语言0..25、末2层基础权重和 final norm 全冻结。仅末2层 q_proj/v_proj 的 LoRA A/B 共8张量、114688参数可训练（rank8/alpha16，A独立固定种子，B零初始化）；每次更新后重算末两层、norm和 FP32 masked mean/last，绝不复用历史最终4096维缓存作为更新后的条件。现有最终 hidden 缓存不是可训练尾部的输入缓存。

## 调用上限与零额外选择

| 阶段 | 完整 Qwen | serial replay | 路径头 | K4路径状态 | optimizer执行 |
|---|---:|---:|---:|---:|---:|
| 6输入官方捕获及零LoRA回放 | 6 | 6 | 0 | 0 | 0 |
| 两逻辑微步，full/cut分别更新 | 2 | 2 | 4 | 16 | full2 + replay2 |
| 每步后同一输入 feature复核 | 2 | 2 | 0 | 0 | 0 |
| 总上限 | **10** | **10** | **4** | **16** | **4** |

两个独立 head/AdamW 状态共享一份不可变2B基础权重以限制显存。每逻辑步先保存 LoRA pre-state，full分支更新并封存其梯度、loss、路径、参数和Adam状态；恢复同一个 pre-state及RNG后执行 replay分支，比较全部状态。不能将两分支合计4次optimizer执行写成2次。后续逐项权重hash确认所有冻结参数保持原值。额外图像处理、模型加载、权重hash和序列化实际耗时也记录；不当作在线请求延迟或训练加速结果。

每个调用发出前写账本，失败同样占预算。失败保留原输出／状态／堆栈并退出1；禁止自动重试、换输入、扩大容差或重新运行同输出池。中间 checkpoint 仅为诊断证据，不授权重复已发调用。轻量单元测试可在合成小模型上测试2步vs1+恢复1，不增加真实6输入预算。

## 必须通过的技术门禁

1. 六输入官方输出与写盘再读的零LoRA serial replay mean/last逐值**精确**相同；实际输入只含用户RGB和原指令，无答案token。
   另零额外forward校验本次六个官方feature逐值等于 common-head 训练时封存的六份 `mean_hidden/last_hidden`。NPZ哈希必须经原 composite cache receipt／old225-by-id逐项绑定；原观测行key绑定image/instruction/parent/split，image hash、processor/revision、token数量及提取器源均核验。历史NPZ**没有保存完整prompt或token IDs**，不能声称逐token旧值对比；本次完整保存，旧记录只能证明输入/构造来源及token count。
2. 两步 full/replay 的 feature、真实路径＋grounding损失、所有梯度、裁剪后的实际参数、两个独立Adam状态、RNG、候选路径／事件逐值精确相同。
3. 每步更新后同一输入 full/replay feature精确一致，至少一个更新后feature相对初始实际改变。不能只证明零适配器等价。
4. 八个LoRA张量两步后均实际改变；第二步均有非零有限梯度。第一步A梯度为0是B零初始化的预期，不判失败。路径头所有参数 `requires_grad=True`，逐参数记录梯度与实际变化。
5. vision/embedding/prefix/末2层基础权重/norm 全量前后hash一致；不存在其它可训练基础参数。实际调用预算不越界。

本门禁只覆盖**串行原长度**，不同输入长度亦各自单独回放。未证明padding/batched replay、加速比或正式学习效果。若未来3000×32配对仍采用串行microbatch累积，可复用此接口；如果改batched padding，必须另行预登记mask/position/pooling及梯度数值等价门禁。

## 执行与证据

root冻结source并实际跑测试后才可启动唯一probe，GPU1/35%，taskset单核，独立record_job和fresh输出。示例参数（release由root填写）：

```bash
python -m scripts.audit_two_row_qwen_prefix_replay \
 --config configs/two_row_qwen_prefix_replay_probe_v1.json \
 --model /home/wzy/dpvlm/route_set_v1/data/qwen3-vl-2b-instruct-89644892 \
 --data /home/wzy/dpvlm/route_set_v1/data/observation_two_row_composite108_v1 \
 --head /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_composite108_v1/peak_seed0/last.pt \
 --output /home/wzy/dpvlm/route_set_v1/runs/qwen_prefix_replay_probe_v1
```

保存 source／官方依赖／模型资产hash、输入prompt/token IDs、选定原始数据hash、6份冻结前缀、官方初始feature、每步全量分支证据／状态、更新后feature、所有梯度/参数变化、冻结参数前后hash、调用账本、状态及artifact索引。不输出DEV指标；技术通过之后由root决定普通同common-head额外3000×32 frozen-vs-LoRA对照，不自动启动。

状态另保存实际 CUDA peak allocated/reserved bytes 与全任务 elapsed/3600 保守 GPU-reserved-hours。GPU模型尚未请求加载即失败时显存字段为null、GPU小时为0；不把峰值显存或总墙钟当作利用率和单请求时延。

实现阶段本地收据：追加历史六缓存门禁后，18项纯协议／预算／路径隔离／封存缓存测试通过（0.58秒）；3项需要Torch的真实小Qwen/状态测试因本地无Torch而跳过，未把跳过项宣称通过。冻结后的服务器测试和真实6输入probe是独立门禁。
