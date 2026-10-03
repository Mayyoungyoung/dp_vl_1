# 正式冻结 Qwen 前缀缓存：完成记录

固定 321 条输入全部通过：官方完整前向特征与原封存 mean/last 特征逐值一致，保存到 CPU 后重新加载的第 26 层输入，经原最后两层和 final norm 重算也逐值一致。285 条 TRAIN 与原 36 条 DEV_MODEL 共 107 个实际父场景；没有读取新 DEV、reserved raw 或路线监督，没有训练。

这是一项普通 partial-LoRA 基线的效率准备及数值等价证据，不是路线质量结果或方法贡献。模型与 processor 均为 `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`，实际运行 Torch 2.4.1+cu121、Transformers 4.57.1、BF16/SDPA。

| 实际项目 | 结果 |
|---|---:|
| 完整 Qwen 前向 / 原尾部重放 | 321 / 321 |
| head 调用 / 候选路径 / optimizer 更新 | 0 / 0 / 0 |
| 历史特征 / 重放精确相等 | 321 / 321 |
| token 长度 | 291 条为 89，30 条为 90 |
| 主体耗时 | 80.600073171 s |
| 外层作业耗时 | 81.866961 s |
| 保守 GPU 占用小时（整个主体耗时） | 0.0223889092 |
| CUDA 峰值 allocated / reserved | 4278929920 / 4297064448 B |
| 原始 CPU 前缀文件 | 321 个，133613250 B |
| 数值审计 NPZ | 321 个，11905146 B |
| 实际 Linux/Torch 测试 | 53 passed，0 skipped，4.63 s |

全部 625 个冻结参数张量的前后哈希一致。缓存保留实际 B1 token 长度、BF16 中间状态、真实位置编码、attention mask、cache position、pooling mask 和原 decoder kwargs；训练读取器只提供此前缀，不提供最终 hidden feature 或任何标签。适配器每次更新后，必须重新执行最后两层和 norm/pooling，旧最终特征不得充当更新后的模型条件。本轮不验证可变长度 padding/batched replay。

逐调用同步计时累计 full 19.438646 s、replay 1.185454 s；其余主体成本包括加载、校验、CPU 复制、序列化与哈希。这些是缓存准备成本，不能替代端到端单请求时延。历史 NPZ 只保存 token 数量而非 token ID 序列；本轮保存新 token 元数据，不声称与不存在的历史 token 序列逐值比较。

实际源 commit 为 `4371e5b98a8bbab91d2107ed79de7d9e107b9bea`。缓存作业 PID 678871 / child 678872，`2026-10-03T01:24:34.126730+00:00` 至 `2026-10-03T01:25:55.993691+00:00`，退出 0。测试 PID 678642 / child 678643，退出 0。配置为 fresh-only、失败即停、每个 issued 调用计入预算；没有重试或恢复。

固定 manifest SHA256：`4a3152db319e67c158f692a4b88e8575fd5e00b7ebe483ebfe4fafec1552bf15`。原 export SHA、全部模型资产/source/probe receipt SHA、每输入来源与等价标记保留于 [原 manifest](corpus_metadata/manifest.json)。同一普通 head 来源为 `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`；本轮没有加载或调用它。

实际冻结 wrapper 位于 [source/qwen_prefix_corpus_4371e5b.sh](source/qwen_prefix_corpus_4371e5b.sh)，其 `tests` 与 `cache` 阶段由 root 分别启动，命令、cwd、版本和退出码保留于 [run status](run/cache.status.json) 与 [validation status](validation/tests.status.json)。既有输出已完成且不支持重跑到同目录。

归档保存 28 份服务器原始文本字节（元数据、日志、测试、源与 wrapper）；全部逐项哈希核验。[REMOTE_TENSOR_INDEX.json](REMOTE_TENSOR_INDEX.json) 索引 321 PT + 321 NPZ 的真实远端路径、字节数、SHA，642 项均与原 artifact index 一致，未下载任何张量文件。归档本身没有新增模型调用。PT/NPZ 总量 145518396 B。核验摘要见 [VERIFICATION_SUMMARY.json](VERIFICATION_SUMMARY.json)，本地完整清单见 [ARTIFACT_INDEX.json](ARTIFACT_INDEX.json)。

下一步仅是已另行预登记的 frozen-vs-LoRA 普通续训比较；本缓存成功不预判 LoRA 有效性，不代表任务成功率提升。
