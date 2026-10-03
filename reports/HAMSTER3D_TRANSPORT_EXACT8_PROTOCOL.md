# HAMSTER3D 官方 bf16 传输技术对照：exact8 v1

仅实现准备，尚未运行服务器/官方模型。旧 b127 源、安装库、权重、失败目录都不改。这不是新方法或路线质量结果，也未授权完整请求。

新增五文件：`routeset/hamster_layer_transport.py`、`scripts/probe_hamster3d_transport.py`、对应config/test、本协议。

## 原失败证据与问题

b127的第一TRAIN请求在299.847637秒达到超时，88次模型forward、87个生成token，余5请求未尝试。首forward输入445token，随后都是1，因此KV cache实际工作。保存的partial是point_3d JSON，不能归因于长篇推理文本。新配置逐字节绑定原目录15个文件，共649754B；运行前复制全部原件至新输出`original_failure/`，含完整log、status、ledger、输入、中间RGB/depth、prompt和87token原数组。复制不会重复计入旧GPU成本。旧输出不覆盖、不清理。

实际Accelerate1.10.1公开cpu_offload路径每次pre把CPU权重移入CUDA，post置meta，并非再复制14GB回CPU。396个decoder参数、每层385892864B、36层合13892143104B（12.9381GiB）/完整forward。其set_module_tensor_to_device默认clear_cache=True/non_blocking=False，旧hook未覆盖；参数上载/meta操作最多约792次缓存清理/完整forward（不含buffer）。这是源码与参数账本支持的机制，不是已测PCIe带宽/时间占比。源码4SHA绑定实际安装版本，不能由当前在线文档替代。

## 唯一新传输实现

官方权重bf16、算子、SDPA、processor、float16 depth roundtrip、640 resize、greedy/use_cache均保持。只换项目内decoder根hook：层0..3常驻；其余32层用一次建立的pinned CPU字典，完整逐层pre上载、post置meta。两处均通过公开setter明确`clear_cache=False, non_blocking=False`，不异步stream，不全局monkeypatch、不改installed文件。所有官方CPU→pin/resident tensor逐值与SHA核验，运行后再次核验。无训练/LoRA/量化。

新whole-layer活动权重比旧submodule逐个上载占用稍高，因此计入内存公式。非decoder常驻4387590624B，4层加1543571456B，一层活动385892864B，再加至少2147483648B workspace，总8464538592B；原固定35% cap为8864694272B，剩400155680B用于额外buffers。实际所有buffers额外保守计入，超出直接拒绝；不能减少预留或增加常驻层。

32层pinned参数12348571648B（11.5005GiB），每层转换前按当前RSS+该层副本检查32GiB上限，转换后及每模型forward后核RSS与实际进程峰值。释放旧CPU引用；不留两个完整模型或完整state_dict复制。统计actual pinned bytes、loading/conversion、CUDA allocated/reserved峰值、RSS峰值。CUDA allocator强制原cap；RSS依边界采样及ru_maxrss检查，非OS硬限制。任何超界是技术失败，不能降精度重试。

## 固定两个partial，最多16次forward

仅`two_row_reach_283200_target0`，从原composite第一输入行读取，后续任何TRAIN/DEV行不解析；不读supervision、真几何、目标或任务成功标签。原RGB-D/current文件按exportSHA核验，但模型仅实际官方RGB/depth/语言输入。同一预处理张量用于两臂，前后全部tensor SHA核验不变；其prompt和官方中间RGB/float16 depth还必须与原失败精确一致。

顺序固定：原Accelerate hook `max_new_tokens=8`，然后释放本次KV/递归移除旧hook恢复CPU，安装新hook，同输入`max_new_tokens=8`。两次均从`rope_deltas=None`的首请求状态开始，不传旧past_key_values。每臂恰8tokens/8forward，合计最多2个K1技术partial、16tokens、16forward。若早EOS、超时或任何失败，不补调用；保留已发计数和partial。不存在warmup、重试、候选池合并或质量评价。

模型输出hook记录每次真实`result.logits`的完整tensor形状/值/dtype，所有vocab全部保存为CPU PT；若真实prefill输出含445位置，整份保留且两臂形状必须一致。不强行改变logits_to_keep。CPU复制/clone耗时逐调用计入并单列；输出hook不发新forward。logit比较`torch.equal`且全部finite，NumPy差异展示的fp32转换对bf16无损，实际dtype和Torch逐值比较仍是主门。两臂8tokens也必须与原87前缀的前8一致。不能由这8步声称完整请求或所有输入数值等价。

计时分别给prefill和7个decode。decode计时包含匹配的全部logit CPU capture，不是无仪器生产吞吐；原后新顺序有allocator/OS/cache温热差异，不将观察提速单独归因pin或clear_cache。一次phase-boundary empty_cache允许且成本记入conversion，不在逐token hooks执行。loading≤600秒、转换≤180秒、每partial≤180秒、整个body≤1200秒的剩余预算用于各阶段；无自动延长。输入校验/归档、CPU比较和保存计入whole body，arm时间是嵌套生成仪器范围，不重复相加。

## 预登记通过门与后续权限

必须同时满足：所有8tokens/全部实际logits exact+finite、固定GPU/RSS/≥2GiB预留检查通过、新7个decode中位耗时≤原75%。有工程完成但速度未达标时保留`completed, gate_passed=false`，不是假成功。任何算术/资源/流程错误保存failed。门槛是小样本技术屏障，不是统计性能结论。

通过后也不会自动调用完整生成。root可另决定全新目录下唯一K1/1024tokens/900秒请求；届时完整前87token来源保持，不能混作与原300秒等预算优胜。本脚本没有该stage/CLI。

## 冻结与验证

实际运行须root固定新commit/源码/config/测试后，用已完成私有`.venv-hamster3d`、CPU0、GPU1 UUID、单线程和原cap独立record_job；网络离线。旧b127脚本源SHA固定63a561d8c12fafe2fbe07eafbf3d72bf69db6d507203f14fa316ae10f60fc76d。所有18资产重新核验的读取开销计入技术body，不重复算历史下载/上传；不下载任何新资产。

本地纯测试不加载Torch/官方模型。CPU Torch测试仅用真实小Linear验证新hook值/meta状态和失败释放，设备拷贝映射到CPU、pin属性模拟；不冒称真实CUDA传输已测。真正生产等值必须由本次两个8-token实际调用证明。完整历史失败、每forward issued/completed、全logits、partial、峰值、源码与artifact SHA全部留存；PT可留服务器hash索引，不进普通Git。

本地实际验证：`python -m pytest -q tests/test_hamster3d_transport.py --junitxml=.bootstrap/hamster_transport_exact8_local_tests.xml` 得17 passed、3 Torch依赖skip，0.90秒；两源码`py_compile`通过。Linux正式前置必须让这3项真实CPU Torch测试执行，不能把本地skip当通过。旧`probe_hamster3d_train6.py`无diff；没有服务器执行或额外模型调用。
