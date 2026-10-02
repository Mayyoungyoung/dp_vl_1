# Formal116 正式采集：机械闭合归档

**116个登记父场景的闭合标记全部存在，两条协调器进程均正常退出。闭合不代表轨迹采集成功，更不代表116个成功父场景或3132条有效路径。** 本次仅归档运行与机械状态，不进行新实验，也不统计保留角色中的成功、失败或轨迹内容。

## 运行身份与实际结束

- 数据目录：`/home/wzy/dpvlm/route_set_v1/data/observed_two_row_formal116_v1`。
- 运行session：`runs/observed_two_row_formal116_v1/sessions/20261002T183023Z_485548`。
- 固定采集源：`262648796fd47b25a2051cc227b6838515b651c6`。
- session状态`completed`，启动2026-10-02 18:30:23 UTC，结束21:10:27 UTC，外层墙钟9604秒（2小时40分04秒）。
- shard0：record PID485591，child485594，18:30:33.824134→21:10:08.153375 UTC，exit0。
- shard1：record PID485593，child485595，18:30:33.838156→21:10:27.327172 UTC，exit0。
- 两shard分别固定CPU2、CPU3单核，CUDA隐藏；记录的外层墙钟包含等待/协调/检查，不应当作成功轨迹的CPU成本或把两进程时间加成实际CPU消耗。

初始targeted tests和registration_prepare的状态文件也为completed/exit0；本归档没有打开其日志或重新执行测试，因此不从状态文件推断测试条数。

## 闭合登记与读取边界

独立只读核验于**2026-10-02 21:20:56.047888 UTC**完成，CPU affinity仅`[2]`。它读公开的预登记计划以取得父ID、角色和编号，然后只检查闭合文件名、存在性、字节数及SHA256；没有解析、复制或打印任何closure JSON内容。

| 预登记角色 | 请求父数 | 闭合文件存在数 | 此次结果内容读取 |
|---|---:|---:|---|
| TRAIN | 64 | 64 | 无 |
| DEV_MODEL | 12 | 12 | 无 |
| DEV_SCORE | 12 | 12 | 无 |
| CALIBRATION | 12 | 12 | 无 |
| TEST_LOCKED | 16 | 16 | 无 |
| 合计 | **116** | **116** | **没有新增保留集统计** |

这张表的角色来自预登记，不来自闭合内容。没有读取worker日志、coordinator日志、attempts、原始图像、深度、路径、验收标签或指标。闭合文件的opaque哈希允许以后验证字节未变，但不能证明物理有效性、状态恢复质量或可训练资格；后续使用数据仍须执行原机械数据门禁。

预登记每父3条目标指令、每指令9个请求槽，116父共3132个**预定**请求槽；此次未读取槽账本，不给出实际尝试数/有效数，不将未尝试与失败合并成成功。既定划分、无替换与父场景隔离保持不变。

## 仅引用已有开发数据报告

既有 [prefix76报告](../OBSERVED_TWO_ROW_PREFIX76_BASELINE_RESULTS.md) 已记录：64个TRAIN登记父中63个有输入，共189指令与1108条正参考；缺图父283220的3输入/27未尝试槽保留。原12个DEV_MODEL父共36指令、205条正参考。这些是**已有开发报告的引用**，本次没有重读原始数据重算。

不由这些开发数推断DEV_SCORE、CALIBRATION或TEST_LOCKED的结果数量、成功率、分布或质量。此归档也不改变任何方法结果、模型选择或论文达成状态。

## 来源与完整性

registration SHA256：`e85cc75d0a6d813a9dc292420edbb3ab123ffe595a9aafed2d22424971bcd0ff`，与固定release的registered配置逐字节相同。session原有`source_sha256.txt`所列6项来源均与实际固定release字节一致：collector、canonical repair、registration脚本、registered配置、shard wrapper和launcher。

本归档保存：

- `mechanical_audit.json`：116个文件的名称/存在性/opaque SHA、预登记角色、原始状态来源与6项来源复验。
- `session/`：明确白名单中的原始coordinator status、PID、起止时间、exit code、source元数据及冻结launcher/wrapper；所有文件从服务器按原始字节保存并核验SHA。
- `source/audit_two_row_formal_mechanical.py`：根线程原`.bootstrap`审计入口的原字节副本。
- `source/audit_completion_metadata.py`：本次更完整的只读opaque hash归档入口；经SSH stdin运行，CPU2，exit0，命令与stderr记录于`audit_transport_receipt.json`。
- `mechanical_audit_transport.json`：原始SSH stdout传输回执，仅含上述允许元数据的base64，不包含closure内容。
- `LOCAL_ARTIFACT_INDEX.json`：本地归档全部文件（自身除外）的字节数/SHA与原服务器白名单文件逐项相等检查。

该采集session已经结束；不要再次启动完成的采集，或以恢复名义重发槽位。本次没有新训练、Qwen、GPU、模拟器调用，也没有新增或替换父场景。
