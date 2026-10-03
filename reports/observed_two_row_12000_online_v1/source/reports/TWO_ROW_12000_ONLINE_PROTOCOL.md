# 12000步普通两臂的真实Qwen在线成本验证

仅补齐实际端到端成本与保存池一致性。固定 `constant` 与 `no_direct` 各自已完成12000步的**原best**，各36个原DEV请求、每次K4：共72次真实Qwen编码、288候选槽。两个best步数允许不同，训练预算/原48次选模相同；不重新选择checkpoint，不跑last，不新增训练，不访问extension288的新32封存DEV。

constant源固定 `1417cdc802674a00e70d4ced3181c6a6ede2a99e`，原best SHA `6b99a2171d5f241d879797371b5a9a406228e48c009db7554e7bc186d69b926d`；no_direct源固定 `1a3eef1fb12d55e98d4d188a091ea40ea62c0a02`。分别从完成的 `convergence_result_receipt.json` / `no_direct_result_receipt.json` 核验原best/last/summary哈希、实际12000/1536000训练曝光与48次history、config/source身份。no_direct须其共有初始化/实际抽样完成检查通过。不得把未完成模型、旧1500或cosine模型代入。

独立入口 `scripts/evaluate_two_row_12000_online.py` 不修改原硬锁1500的 online文件。用 scoped adapter 注入新的严格preflight和对应原模型类，然后复用原 `FrozenGenerator`、`process_request`、`LabelEvaluator` 和串行main；finally恢复入口/模型类。原model/loop/checker/cache源逐文件和各自实际training release比对。no_direct仍无直接分支、几何仍语言条件化；eval中全部参数冻结，不做endpoint钳制或额外特征前向。

每请求独立读取/hash RGB与current NPZ，真实RGB-D反投影，原processor处理RGB+语言，Qwen真实forward，无hidden cache输入；随后一次K4头输出。先写NPZ/hash封存，才解码该条目标/验收几何标签并同两排2cm tip/3cm目标/type/event规则评价。生成器只持输入白名单、观察哈希与冻结模型，标签对象不回传生成。没有修复、过滤、重试或额外候选。

实际连续单请求wall包括读取/hash、processor、Qwen、RGBD、头、GPU传输、预测封存及该条label IO/check。模型加载/metadata启动分开记录；无warmup，首请求保留；两臂独立进程/独立加载，不把温缓存优势当方法结果。OS文件页可能被完整性预检预热，如实说明。请求失败算4 NaN槽，随后未尝试请求也保留注册36/144分母；不自动重开失败池。

完成后只读原 `dev_model` NPZ/per_scene/metrics与训练cache，零额外forward比较每候选坐标/event/决定及mean/last隐藏特征和token精确字节。原batched与在线serial数值容差和精确字节结果均保存；差异不重跑、不修正。不存在score模型，SelectedValid和完整机器人执行保持null。

root须先阅读no_direct实际完成结果，再在新冻结source执行两臂。每臂独立新output、record_job，GPU1/35%、单CPU；两个原训练成本不重复加入新在线成本。若某臂失败，所有已封存结果保留，由root决定是否停止，未启动另一臂不能记成已完成72请求。

```bash
# 在root冻结的新SOURCE中；MODEL为实际已校验的官方Qwen本地目录。
taskset -c 0 /home/wzy/dpvlm/route_set_v1/.venv-qwen/bin/python -m scripts.evaluate_two_row_12000_online --arm constant --model MODEL --output NEW_CONSTANT_OUTPUT
taskset -c 0 /home/wzy/dpvlm/route_set_v1/.venv-qwen/bin/python -m scripts.evaluate_two_row_12000_online --arm no_direct --model MODEL --output NEW_NO_DIRECT_OUTPUT
```

需按既有launcher设置 `CUDA_VISIBLE_DEVICES=1`、单线程、不可变release/PYTHONPATH，root record_job记录实际SHA/PID/退出状态。本入口不提供自动训练、自动checkpoint重选或生成恢复。
