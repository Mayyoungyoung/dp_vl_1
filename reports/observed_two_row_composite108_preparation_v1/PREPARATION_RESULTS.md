# Composite108：实际导出准备记录

冻结源 `1fccf4898bfa17233f92e20476adeb8db12c6b60` 的readiness和export均真实exit0。所选108个父（旧76＋新32）全部机械闭合，导出321条实际观察、1868条已验证正参考。本记录仅证明闭合与导出；**不证明新组合数据的TRAIN容量/质量审计、Qwen缓存或训练已经通过**。

|范围|请求父|实际有观察父|实际输入|正参考|
|---|---:|---:|---:|---:|
|旧64 TRAIN＋新32 TRAIN|96|95|285|1663|
|原12 reused DEV|12|12|36|205|
|合计|108|107|321|1868|

原缺失父 `two_row_reach_283220` 仍保留，未用成功父替换。请求324输入中3条不可观察；2916个登记路线提案中2889有实际attempt记录、27未尝试。不能把108机械闭合说成108个成功场景。全部已验证正例按原规则保留，unknown类型和长弧不因本次导出删除；本记录不新增模型能力结论。

实际新增部分只有first32 extension TRAIN：96条输入、555条参考。没有导出/编码extension新32DEV。原12DEV是反复使用的开发集，绝不改称独立测试或OOD。角色参考数量来自原DEV205及其完整字节保留证明，与新总数1868相减得到TRAIN1663；本归档未另开或复制SUP内容。

## 历史字节与来源

exporter在实际导出中完整核验原文件是新文件的精确字节前缀，本地又将证明内SHA与原prefix76封存manifest逐项对照：

|文件|原行数|原字节数|追加行数|
|---|---:|---:|---:|
|observations.jsonl|225|74062|96|
|supervision.jsonl|225|481653|96|
|attempts.jsonl|2025|8559303|864|

各原字节SHA、原manifest SHA、输出SHA和source SHA完整保存在 `HISTORICAL_PREFIX_PROOF.json` 与原 `export_manifest.json`。此处证明的是观察/监督/attempt文件前缀，不是225个Qwen NPZ缓存已复用；后者必须由后续缓存阶段独立完成。

实际跨源门禁检查旧116＋新288共404份注册计划的exact/1mm几何哈希，以及当前148份已闭合actual1mm机械记录；所选108父无missing/blocked/duplicate，raw校验前后均通过。已封存角色仅提供机械哈希，不打开其raw。没有把布局哈希通过当作语义/路线质量认证。

新manifest SHA `04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc`；旧manifest `309966e192a3c582cde503151e607b5ede679eaec596bd8ab2bc1fd06ffdbab1`；原reader `3ed1c96d69376075d8a15c11c6eae1d4fcdc50e116010cbd32b4e6540df13add`；本次exporter `a641245529113e47e80355991de8c1fbe737aacde1b47d24a0e1bc64e029d9f8`。manifest含3097项来源哈希和4个导出文件哈希，完整保留。

## 实际运行与归档边界

readiness PID645056/child645057，2026-10-03T00:04:24.280210+00:00至2026-10-03T00:04:24.638944+00:00，0.358734秒。export PID646961/child646962，2026-10-03T00:08:18.921787+00:00至2026-10-03T00:08:26.925079+00:00，8.003292秒。wrapper隐藏GPU、线程环境均1，GPU小时0；未作模型forward或搜索。

服务器数据目录 `/home/wzy/dpvlm/route_set_v1/data/observation_two_row_composite108_v1`。9份原metadata/log/status/source/wrapper逐字节归档，索引为 `REMOTE_ARTIFACT_INDEX.json`，archive SHA为 `e84ca1fe635f87611d6e6cffeaabc836a11bbc1701dd5de67f1a5022087023cb`。未下载图片、depth、轨迹NPZ或完整observations/supervision/attempts语料文件，未改变既有online归档或global文档。

原selection中的“prospective/no export”文字是注册时状态，保留原始字节；本次完成状态以实际job收据和manifest创建时间为准。后续质量/cache/train须各自真实收据，不能从这次成功导出推断。
