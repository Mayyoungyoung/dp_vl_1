# 两排 prefix76：64TRAIN前缀的真实准备结果

固定源码 `9e0094aff152d48cd34eadd24e554a05f0cf4a0f` 的CPU准备于2026-10-02 **20:16:42.949992 UTC**完成，export与TRAIN质量审计均exit0。预登记64TRAIN＋原12DEV闭合后导出，未替换失败父。沿用该release已完成的140项服务器测试（17.16秒，exit0），见[共享验证索引](observed_two_row_scaling_validation_v1/artifact_index.json)；本次未重复测试或启动GPU。

| 分区 | 请求/有观测父 | 请求/实际输入 | 请求/完成路线槽 | 已接受正参考 | 完成但未接受槽 |
|---|---:|---:|---:|---:|---:|
| TRAIN | 64 / 63 | 192 / 189 | 1728 / 1701 | 1108 | 593 |
| 原DEV_MODEL | 12 / 12 | 36 / 36 | 324 / 324 | 205 | 119 |
| 总计 | 76 / 75 | 228 / 225 | 2052 / 2025 | 1313 | 712 |

全部64个TRAIN机械closure逐字节归档，并逐一核对其SHA已被完成的export manifest绑定，而非将后来闭合的父冒充准备时已闭合。64个worker均正常退出、64个实际几何指纹均与登记匹配，但只有63父保存初始观测。完整[机械汇总](observed_two_row_prefix76_preparation_v1/mechanical_train64_summary.json)保留初始64个闭合身份和槽分母。机械closed/exit0不等于路线成功。

唯一缺失父仍为`two_row_reach_283220`：没有初始图像，27请求槽全部未尝试，三个请求输入未伪造；`model_eligible=false`，没有用其他成功父替换。原worker耗时18.287秒、几何匹配和关闭原因范围见[原closure](observed_two_row_prefix76_preparation_v1/missing_parent_283220/closure.json)。机械记录本身不足以指出具体初态失败项，本次没有额外读取或重跑其私有outcome。

TRAIN质量审计包含全部189输入、1108正参考，已记录输入中零参考条件数为0。终点stride2观测点逐坐标±5cm容量检查 **1108/1108**，最大最近点L∞为1.784cm；模型event-aware H24原tip检查 **1108/1108**，事件转换计数变化0/1108。460条unknown正参考（41.52%）全部保留，不计为无效，也不当成460种路线类型。

| 全部TRAIN正参考 | raw | 模型H24 |
|---|---:|---:|
| 路径长度均值 / 中位数 | 1.577 / 1.354m | 1.505 / 1.306m |
| 长度p95 / 最大值 | 3.104 / 7.304m | 2.954 / 6.807m |
| 超过2m / 超过3m | 217 / 67条 | 185 / 55条 |
| 最高世界z | 1.978m | 1.977m |

长弧、unknown、失败和未尝试槽均保留；没有截断/筛选，也没有改变2cm tip/3cm目标标准。容量与参考tip检查通过不证明模型可学、泛化或全机械臂执行有效。原始逐参考统计与来源hash见[train_quality.json](observed_two_row_prefix76_preparation_v1/train_quality.json)，便于复核的分区与长度汇总见[quality_summary.json](observed_two_row_prefix76_preparation_v1/quality_summary.json)。

导出作业PID543164/child543165，进程墙钟4.748秒；TRAIN审计PID543172/child543173，进程5.600秒（审计内部4.037秒），两者均CPU affinity=[0]、线程1、GPU隐藏。源命令、UTC、退出码、恢复说明在对应status.json，原日志未改写。导出机械gate恰有76父闭合，无duplicate/blocked，只记录283220初始观测不可用；不是全116父最终完成声明。

[fixed_dev_identity.json](observed_two_row_prefix76_preparation_v1/fixed_dev_identity.json)是准备阶段已经完成的原36DEV身份一致性receipt。本次只归档其字节及metadata，不新打开DEV图像/路线数组，更未读取SCORE/CALIBRATION/TEST_LOCKED raw。GPU后续重编码、初始化、训练和在线验证属于独立阶段，本准备报告不记录其指标。`preparation_receipt.json`内`gpu_stage_started=false`准确表示准备收尾时尚未开始下一阶段，保留原字节，不据后来启动改写。

80个服务器文件在只读同步中复制前后双读SHA，与本地逐项一致；包含64个机械closure，未复制任何模型权重或轨迹数组。本地全部83个原始/派生证据索引在[artifact_index.json](observed_two_row_prefix76_preparation_v1/artifact_index.json)。同步使用CPU affinity=[1]，GPU使用0，传输压缩包313,204字节，SHA256 `0f9569cb72778cc7a5fe49997d44d5085cf1612b67ce2a94f8de019f8ff49aa4`。

- export manifest：`309966e192a3c582cde503151e607b5ede679eaec596bd8ab2bc1fd06ffdbab1`
- TRAIN quality：`2d3859688cd58d9c646574e687c1c139dcbd9105102c8a8abd5ff4c25ac41f8d`
- 固定DEV身份receipt：`d1c125d1423c5ba337c39a8d58298c29935ae99775de8fd331ea24afb3af2d02`

本次仅归档已完成准备证据，没有启动/停止服务器作业、生成候选或提交Git。
