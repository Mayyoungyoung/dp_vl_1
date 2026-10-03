# 已授权321观察的冻结 Qwen layer26 前缀缓存 v1

仅缓存实施预登记，尚未实际执行。目的为普通部分LoRA强基线降低冻结前缀的重复训练计算；不是方法创新、推理加速或路线质量结果。成功TRAIN6门禁并不证明LoRA有路线泛化收益。

## 已通过的前置证据

唯一实际probe来源 `eddfacaddab2d12c67f5a56fd775de173c05f9b2`，`runs/observed_qwen_prefix_replay_probe_v1/probe`。原status SHA `528d730af8a1a0c7910605883d56e856963a40aa2efae662c0553609fa2718fc`，preflight `23d120d6770e32f69c0fd5932d04ec1f1b8bb4b93d0b9a88374e93e05cc100d1`，artifact_index `776553d2b5a1d81cb2313c3c12ef88de5dab667961fdff7436a28d8fdae2b78a`，final_audit `32ebdbdc62eb42e92877d82f0ee05f7004fcdac445d52f54c8e65b8139540673`。新入口逐项验证这些原始字节、completed/gate、源依赖及全部模型／processor资产hash；不读取probe的PT或重跑probe。

该probe实际37项服务器测试通过，真实TRAIN6的10 full/10 replay/4head/16候选状态/4optimizer执行完成；历史feature、零LoRA回放、两步loss/梯度/Adam/RNG/参数及更新后feature均精确一致。8LoRA和全部head参数真实改变、基础权重保持。它只验证serial原长度，未验证batched padding。

## 固定样本与信息边界

只用封存 composite108 export SHA `04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc`：旧64请求TRAIN中63个可观察父／189指令，固定旧12DEV／36指令，以及首32个新TRAIN／96指令，共**285TRAIN +36旧DEV =321**。顺序逐ID固定，283220缺初始观察事实保持；不重抽父、不补输入、不包括新extension DEV32、score/calibration/locked。

只有原RGB和原语言进入Qwen。读取观察5key JSONL、已封存最终feature NPZ及机械layout门禁；不打开supervision、轨迹、几何验收标签。原旧DEV36仅为后来固定评价重新编码条件，不计算任何预测/指标、不选择模型或数据。机械gate可比较全部注册/closure的hash与角色，禁止读保留角色raw。

## 唯一缓存过程和预算

前置核验全部321历史NPZ的原schema、文件SHA、observations行key、image hash、token count、processor revision及模型资产。随后按原顺序每输入恰好：

1. **一次**官方完整frozen Qwen forward，在原layer26 pre-hook捕获输入与全部实际kwargs；没有LoRA或其它可训练参数。
2. CPU payload写盘并用 `torch.load(weights_only=True,map_location='cpu')` 重新加载；保留原dtype、原长度、cos/sin、mask、position/cache_position、None/字面值。隐藏输入 `[1,L,2048]`，原cos/sin `[1,L,128]`；不padding、压缩token或修改位置。
3. 官方FP32 mean/last必须逐值等于原封存最终feature；**一次**原26、27层与norm serial replay，结果亦须精确一致。无适配器的原尾部等价于probe已验证的零LoRA状态。

最大 **321 full +321 replay，0 head，0候选状态，0 optimizer**。每调用先写issued，失败占预算；任一门禁失败终止并封存，不自动重试、换输入、恢复池或追加编码。输出目录fresh-only。CPU1/GPU1显存35%，root冻结并独立启动；脚本不会启动训练。

正式训练必须每次adapter更新重算末两层/norm/pooling，不能用本轮 `audit_features` 或旧最终feature缓存代替。审计feature仅用于等价证据，reader不返回它。前缀可长期复用的前提是vision/embedding/语言0..25全部继续冻结，model/processor/前缀源码和输入均相同。

## 训练读取API与封存

```python
corpus = PrefixCorpus(root, expected_fingerprint=checkpoint_config_hash)
payload = corpus.get(observation_id)
```

公开 `manifest`（dict）、`fingerprint`（manifest.json实际文件SHA）、`ids`（原321顺序tuple）、`rows_by_id`。`expected_fingerprint`非None时必须逐字相等。构造器验证completed status、实际321/321且0训练预算、manifest SHA、最终artifact_index、完整ID/角色、源身份与文件存在/尺寸。首次 `get(id)` 校验文件SHA、安全反序列化、正常detached CPU张量及完整tensor metadata/hash；之后复用已验证的RAM snapshot。调用者不得原地修改payload；原 `replay_feature` 仅映射GPU。训练启动须遍历全部321个get完成预检并记录总bytes，后续不重复96k次磁盘读取。

payload只含 `hidden/kwargs/pooling_mask/cut_layer`，无label或最终feature。manifest每行含id/parent/split/file/sha256/bytes/sequence_length/input_tokens/source/tensor_metadata及两项exact结果。顶层绑定export、observations、旧cache receipt、模型／processor revision与资产hash、replay源码、全新入口来源、成功probe收据、285/36计数及最后机械门禁。状态缺失、未完全封存、fingerprint变化或payload SHA不符均拒绝。

保存完整prompt/token IDs方便以后复核；历史NPZ没有这些序列，只能比较原输入行构造来源、token count及最终feature，不能补称旧token序列逐值一致。保存全过程账本、actual feature审计、源和模型hash、全冻结权重前后hash、实际墙钟、peak CUDA allocated/reserved、保守全任务墙钟/3600 GPU-reserved-hours。GPU模型尚未请求时失败则显存null、GPU小时0；不把缓存吞吐当在线请求延迟。

## 独立执行

冻结source后由root执行单核record_job；示例参数：

```bash
python -m scripts.cache_observed_qwen_prefix \
 --config configs/two_row_qwen_prefix_corpus_v1.json \
 --data /home/wzy/dpvlm/route_set_v1/data/observation_two_row_composite108_v1 \
 --model /home/wzy/dpvlm/route_set_v1/data/qwen3-vl-2b-instruct-89644892 \
 --probe /home/wzy/dpvlm/route_set_v1/runs/observed_qwen_prefix_replay_probe_v1/probe \
 --output /home/wzy/dpvlm/route_set_v1/data/observation_two_row_composite108_prefix_v1
```

没有自动后继阶段。321缓存全部完成且reader核验通过，才能由root另行决定普通frozen/LoRA续训；旧probe、旧最终缓存器和历史LoRA入口不改。

本地实现核验：24项纯测试通过（2.20秒），7项Torch payload/reader测试和1项Windows符号链接测试跳过；冻结后的Linux/Torch测试必须实际补齐。跳过不计为通过。新测试包括321身份/角色、禁止新DEV和label、失败预算不回放、receipt/header/fingerprint封存、只开固定321历史NPZ以及安全CPU反序列化与RAM memo。
