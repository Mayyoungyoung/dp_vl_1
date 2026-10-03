# Composite108：实际测试、TRAIN质量与Qwen缓存

本阶段冻结源 `71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c`；在先前1fcc实际导出的相同manifest `04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc` 上继续。服务器70项测试全部通过、0skip（含3项真实Torch测试），随后独立TRAIN质量与cache均exit0。训练仍是另一个独立阶段，本归档没有读取其结果，不能从这些完成记录宣称方法有效。

## TRAIN正参考与容量

登记96 TRAIN父/288输入，实际95父/285输入；原缺失父及3条不可观察输入不替换。1663条已知正参考全部保留，其中707条类型unknown；285条件均至少有一条正参考。1663/1663满足固定stride2观察点的逐坐标±.05m末端容量，1663/1663事件分段H24通过原两排tip检查，事件转换数量不一致0条。最大最近末端L∞距离0.01793127m。

这只证明当前表示能容纳监督末端以及这些已知正参考的H24几何检查通过，不证明普通头能学会、能泛化或能安全执行。正参考长弧全部保留：raw均值1.584816m、P95 3.237300m、最长7.303556m，105条超过3m。没有按长度/unknown筛选，也没有把未知类别当无效负例。

质量进程PID651312/child651313，2026-10-03T00:15:27.693686+00:00至2026-10-03T00:15:36.486211+00:00；外层8.792525秒，内部7.145094秒，GPU小时0。只解码TRAIN原数组；原DEV字节可被完整性哈希读取，未打开其数组作此质量分析。

## 实际Qwen与精确缓存复用

真实官方Qwen3-VL-2B revision/processor `89644892e4d85e24eaac8bacfd4f463576704203`，torch2.4.1+cu121/transformers4.57.1/bfloat16/max_pixels262144，冻结参数。只对first32 extension TRAIN的96条RGB＋语言输入实际编码，严格5键观察manifest，没有未来路径/目标/模式进入编码。

原225份NPZ（189 TRAIN＋36 reused DEV）逐容器SHA/数组来源核验后按字节复制；新96份也逐原编码容器SHA复制，合计321。原cache config/index/status的固定SHA、提取器实际源码SHA `9b8121ec8c95e4e95d89e191c0fae3b9251db120c7b621bed18dad7fc820d4bf`、新manifest/配置/索引/status、合并receipt及321份NPZ路径/大小/实际SHA均封存。归档时又计算全部321份合并容器和各自old/new origin的SHA，全部相等；不重新编码、不下载这些NPZ。

cache PID652036/child652037，2026-10-03T00:16:54.087262+00:00至2026-10-03T00:17:15.862807+00:00；完整cache job 21.775545秒（保留GPU小时0.006048763）。其中新编码子进程16.901053秒，编码body含加载11.501351秒（0.003194820GPU小时），加载5.200573秒；峰值4279491584字节。上述时间嵌套不能相加，原225份的历史编码成本不重复计入；这也不是单请求端到端规划时延。

## 归档与边界

`train_quality.json`保留全部1663参考统计；`QUALITY_SUMMARY.json`为派生摘要。`cache/`保留实际新96编码日志、索引和receipt，`merged_qwen_cache/`仅合并metadata，`CACHE_NPZ_HASH_INDEX.json`保留321个实际容器hash而不复制大文件。原export/readiness9份字节记录保持，后追加quality6份与cache15份原件均单独有remote索引。validation11份原件在独立 `reports/observed_two_row_composite108_validation_v1/`。

旧12DEV仍是reused开发集。新extension32DEV、旧SCORE/CALIBRATION/LOCKED原内容保持封存。未读取正在训练的模型、指标或输出；没有新增生成、搜索或训练。后续模型结果必须由实际完成收据独立归档。
