# 正式 common-head serial3000 frozen/LoRA 续训协议 v1

本协议落实已批准的[单一草案](TWO_ROW_COMMON_HEAD_LORA_CONTINUATION_DRAFT.md)，不增加方法模块或参数扫描。前置技术探针 source `eddfacaddab2d12c67f5a56fd775de173c05f9b2` 已由root实际确认exit0：37项服务器测试通过、0跳过；TRAIN6缓存前后feature、两步full/replay梯度/loss/Adam逐值exact，8个LoRA及全部共有头参数实际变化、基础权重不变。本轮正式3000步尚未运行；技术探针不是质量结论。

## 唯一设置及输入门禁

- 共同初始权重是composite108 `last12000`，SHA `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`，不用best。
- 原export SHA `04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc`。两source live机械门禁、全部正参考质量audit、原cache/source证明先过，再加载固定285 TRAIN/36旧DEV。旧3缺失输入保留。新DEV32和其它reserved不读。
- 两臂现有1,231,965参数几何/anchor/路线/事件头全部继续训练。全新独立AdamW、头lr3e-4/weight_decay1e-4。LoRA仅语言层26/27 q/v，rank8/alpha16、114688参数、lr1e-5/decay0。Qwen revision/processor `89644892e4d85e24eaac8bacfd4f463576704203`；其余权重冻结。无scheduler变化、warmup、rank/层数/率扫描。
- 3000个逻辑step，每step按相同序列串行B1累积32；各loss除32，满32后clip总norm1、AdamW更新一次。LoRA全局梯度集合较大，裁剪耦合如实披露。
- 原K4/H24/saturation、grounding .02/sigma .025、peak/surface ±.05、当前RGB-D几何输入均保持。只接受原输入白名单，验收box/type/未来路径不进入条件。

配置唯一来源 `configs/observed_two_row_lora_continuation_v1.json`。driver严格逐项校验，不提供CLI放宽预算、结构或阈值。GPU1/指定UUID、35%显存、单CPU affinity，固定Torch2.4.1/transformers4.57.1/SDPA/bfloat16；最后两层常驻GPU，但两臂分别启动，不并行。

## 与独立前缀语料的接口

`PrefixCorpus(root).fingerprint` 是正式manifest文件SHA；`.ids`是原321观察顺序，`.rows_by_id`提供身份/角色/hash，`.get(id)`首次验SHA/安全CPU反序列化后在RAM memoize。训练器先逐321项核验，记录总payload bytes；之后只搬原dtype、原长度，不padding或量化。

payload仅layer26输入hidden、实际位置/cos/sin/因果mask/kwargs、pooling mask、cut_layer。每个微样本在两臂都重算26/27和norm及masked mean+last；frozen no_grad，LoRA保留梯度。geometry point features和路线头始终重算。旧最终4096缓存只用于原数据载入/初始身份验证，实际forward的feature被本次尾层结果覆盖；不会在adapter更新后继续作为条件。

manifest需绑定真实成功探针receipt、原export及cache receipt、实际Qwen资产和replay源码。每项historical/replay exact与原321身份核验通过。共享前缀准备的321 full捕获/321零尾回放单独计费，不算在以下每臂训练中，也不由训练器自动启动。

正式321缓存已由root实际完成，source `4371e5b98a8bbab91d2107ed79de7d9e107b9bea`，53项测试0skip，全321 historical/replay exact，0头/optimizer。policy与reader现锁定实际manifest SHA `4a3152db319e67c158f692a4b88e8575fd5e00b7ebe483ebfe4fafec1552bf15`，路径 `/home/wzy/dpvlm/route_set_v1/data/observation_two_row_composite108_v1/qwen_prefix_corpus`；不额外重做正式缓存。新source guard包含真实RouteBlock依赖`routeset/models.py`。

## 抽样与调用预算

共享 `--draw-plan` 是固定285 TRAIN ID和 `3000×32`均匀有放回index序列的封存JSON；seed100000。首次创建后两臂必须读到相同内容/SHA，不重生成或根据损失调整。保存initial/final sampler状态及逻辑cursor；目标匹配独立RNG和Python/NumPy/Torch CPU/CUDA RNG也完整保存。

| 每臂stage | tail调用 | head调用 | optimizer调用 | K4候选状态 |
|---|---:|---:|---:|---:|
| train逻辑更新 | 96000 | 96000 | 3000 | 384000 |
| 12次DEV选择 | 432 | 432 | 0 | 1728 |
| 独立fixed-last-train | 285 | 285 | 0 | 1140 |

训练成功summary强制journal计数恰为96000 train_tail、96000 train_head、3000 optimizer、432 dev_tail、432 dev_head。新完整Qwen调用0。每臂共有预训练1536000单列，逻辑累计1920000；两个arm不重复收费共同预训练和共同prefix准备。追加每输入平均336.842抽样。

每250步固定全部36 DEV，12次机会，原 `UniqueClassifiedTipValidAtK + .05 × TipValidAtK`、严格大于才更新best的tie规则。先全部36个当前checkpoint的尾层feature，再调用**原two-row evaluator、batch_size=1**产生36个K4池。其语言控制复用池而不另forward。所有路径生成后才做语义/物理检查。只在局部context把旧export verifier替换为composite verifier，不使用拦截fresh初始化的旧training_adapter、不改旧实现。

每个选择点保存当前尾特征、预测、逐场景指标、汇总、pool receipt；best/last引用这12个池，无额外DEV生成。每池receipt绑定完整当前模型/adapter hash、config、顺序ID、前后journal高水位与全部产物SHA。必须有tail_features、predictions、per_scene、metrics，不能空artifact表冒充完成。

训练结束只保存summary，不自动运行fixed-last285。root独立以同source/同arm/完成的last3000调用diagnostic stage；仍串行当前尾层、原评价器，候选/失败全部保留，旧189和新96在后续保存池分析中分别报告。它不更新optimizer、不读新DEV。prefix replay评价成本不称单请求E2E；没有完整在线测量就保持E2E字段空。

## 硬中断、恢复与封存

每输出exclusive active.lock。`requests.jsonl`是append-only、fsync、带序号/前项hash的请求账本；实际tail、head、optimizer调用之前分别issued，失败也保留。**issued并不自动等于成功返回**：只有完成逻辑step和完整pool seal才能认证对应完成范围。中断的issued保守计费，未认证完成范围不能写成准确完成数。

每25逻辑step原子last checkpoint，另保存头两步便于真实梯度核验；只在清空累积梯度后的更新边界保存。保存头/adapter、独立AdamW全部moments/group、scheduler、所有RNG、draw指纹/cursor、逻辑曝光、history/best、梯度审计、累计时间及journal高水位。暂停不能半个micro直接宣称可恢复。

行政参数`--stop-after <绝对logical step>`支持可控暂停，例如2或25；它不进入immutable训练policy/config、不改变总3000步或draw plan。该完整更新边界先保存checkpoint；若到250倍数，照原协议完成该次DEV，再写status=`paused`返回，不写completed summary。恢复使用同一source/实参、删除`--stop-after`并加`--resume`，接着下一逻辑step执行。停止点不能小于已恢复step；3000仍正常完成。实际默认仍直接3000，行政暂停不是可调研究超参数。硬中断后的未封存issued账本仍拒绝自动重放。

恢复默认要求journal恰等checkpoint高水位。未封存micro、optimizer之后但checkpoint之前、checkpoint rename失败等情况，均保留证据并拒绝静默重放；不通过回滚journal凑相同实际预算。root可另行处置，但当前driver不自动重试。

唯一例外是“训练边界已存，但一次DEV池完全封存、末checkpoint尚未写入”：核完整pool身份/hash、全部36tail+36head的精确连续journal span与当前checkpoint后无其它调用后，采纳原池/history，零forward。原封存pool的真实elapsed同时加入累计，并以`recovered_sealed_evaluation_seconds`写入checkpoint/summary，避免将恢复复用说成免费。若边界后尚未发评价请求，则恢复后先补该预定评价，不能跳过机会。部分评价的staging保留并fail closed。

新best先原子保存权重，后写指向它的last；中断时可由完整池恢复这次选择。全部历史pool receipts的SHA进入checkpoint/history；最终best张量hash还必须等所选pool身份。completed输出仅可只读核seal并复用，零模型加载/forward；不能把旧checkpoint改名为新训练。

fixed-last阶段完全封存池可核验后复用；未封存时禁止重复发285请求。原last/head文件均只读。基础2B权重不反复写盘，保留小头/adapter、best/last及必要状态即可。

## 实测要求、费用及启动形式

工程测试覆盖pure配置/抽样/hash/锁/异常恢复/完整pool与partial拒绝，以及真实Torch B1累积32连续4 vs2+恢复2的完整参数、Adam、scheduler、RNG、draw/journal一致；fault覆盖issued forward、optimizer之后、原子rename、完整/不完整eval封存。恢复拒绝改arm/lr/data/cache/source/预算/评价间隔。源冻结后由root运行真实Torch测试，不把本地无Torch的skip写成通过。

实际头参数逐项初末hash，全部60张量必须更新；8个LoRA最终全变、第二步与末步各有非零有限梯度；基础权重全量初末hash必须一致。第一步A梯度为零是B零初始化预期。日志每25步只打印损失、norm、LR、逻辑draw与累计时间，不循环打印所有张量。

总GPU时间从当前process启动包括验证/模型加载/搬运、所有训练和DEV、基础hash、checkpoint写入，record_job外层也保留；恢复的已保存累计成本和本段成本分开可审计。失败job的外层实际费用不能从checkpoint回滚后消失。固定TRAIN诊断另外记账。旧全Qwen串行外推5.115 GPUh仅草案量级参照，不是正式replay速度；实际速度从运行日志实读，不能按28/2层比编造。

root冻结并验证后，分别独立启动两臂，示例仅表接口，不构成此次执行：

```bash
python -m scripts.train_observed_two_row_lora_continuation \
  --stage train --arm frozen \
  --config configs/observed_two_row_lora_continuation_v1.json \
  --model <pinned_model> --data <composite108_export> --head <original_last12000.pt> \
  --prefix-corpus <sealed_prefix321> --draw-plan <shared_draw_plan.json> \
  --quality-audit <complete1663_quality.json> --output <fresh_frozen_run>
```

LoRA只把arm与独立output改为lora；共有其它实参、来源与draw SHA不变。训练边界的显式恢复加`--resume`；不会绕过未checkpoint实际调用门禁。独立诊断使用`--stage fixed-last-train --train-run <completed_arm_run> --output <fresh_diagnostic>`以及同一其它来源实参。

本实现只承担一组普通骨干适配对照；无收益不继续扫曲线/rank/seed。有收益也仍需数据、参数量和计算差异边界、多种子/独立任务确认，不能据此宣布核心路线集合方法成立。
