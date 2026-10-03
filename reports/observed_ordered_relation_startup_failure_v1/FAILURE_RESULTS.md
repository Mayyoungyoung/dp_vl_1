# 有序关系续训：首次启动指纹范围错误

这是**模型初始化之前的工程失败**，不是A损失的科学负结果。原`ece16ba41daab9cfcb9422f0014e640a1c934c7b`输出完整保留；没有优化、路线生成、候选质量指标或MAIN结果。

原45项服务器测试全部通过、0skip；真实CPU父检查确认60个Adam状态条目、LambdaLR step12000、完整五RNG域，以及model/optimizer/scheduler/RNG/sampler/index audit六个组件hash。45项工程测试当时未覆盖父config中`dataset_fingerprint`的实际范围语义，因此没有发现此错误。

| 阶段 | UTC开始→结束 | record/child PID | exit | 外层秒 |
|---|---|---|---:|---:|
| CPU真实45测试 | 07:26:04.082580→07:26:11.031942 | 849805/849810 | 0 | 6.949362 |
| CPU父检查 | 07:26:25.451625→07:26:30.780782 | 850133/850138 | 0 | 5.329157 |
| 首A启动 | 07:26:53.031938→07:27:13.787621 | 850395/850400 | 1 | 20.755683 |

A在新driver的数据身份门拒绝：`Historical all-positive dataset changed`。实际无`requests.jsonl`、无`last.pt`、无模型初始化/forward或optimizer。外层20.755683秒包含runner20.453931、子进程19.105028和内部18.677414秒；这些嵌套数不能相加。

## 实际根因与反证

原`train_observed_geometry.py`把`geometry['fingerprint']`存入config的`dataset_fingerprint`。它包含基础loader指纹、观测RGB-D源SHA和stride。新续训driver错误地拿裸`data['fingerprint']`与它比较。

独立CPU2、CUDA隐藏的只读检查真实exit0：

- 裸loader：`970f9b533e0ea0beb6c6f1fe9d702586d7fd26d7dd675a552b3786925fc2f677`。
- 当前geometry与原父config均为：`df4356838911271ba42f8662a11127e59ea06ec1235583604f7c06487ba59931`。
- 2406个原输入、正参考、缓存和观测几何文件的SHA联合表，与原父`source_hashes.json`逐项相同；原14计算源码也全部保持实际字节。
- 321输入，其中285TRAIN/1663正参考、36旧DEV；没有读取新DEV或保留集。
- 记录现时解码数组的ID、features、current、paths、events、mask、geometry index SHA。原训练未保存这些完整解码数组的独立SHA，所以不虚构历史数组逐值对比；可证据是同一确定性loader源、全部输入字节和最终几何指纹相等。

检查未初始化模型、未读checkpoint、未调用检查器或生成路线；读取的是已授权同一321输入的原文件。内11.046259秒、SSH外12.013672秒，单独记CPU诊断成本。后续失败归档本身只读元数据/日志/源，不再读原始数据。

## 修复与接续

`bcbc58848463cae42c84bdd9bada1956d73a4b43`最小修复将门禁改为**geometry指纹匹配，并要求原source联合表相等**；数据、损失、抽样、模型和验收都不变。新增scope混淆及继承成本两项纯测试，新总数47，实际新源测试尚以根任务后续作业为准。

新输出`runs/observed_ordered_relation_continuation_v2`保持旧失败不动。A的4500秒累计预算继承旧外层20.755683秒一次，B/C继承0；内部18.677414秒不再相加。运行时核对旧失败的source/状态/时间及无journal/last，再计费，不能靠换目录重置已耗成本。CPU测试、父检查、只读根因诊断分别记录，不冒充GPU训练。

本目录含原日志、状态、身份/结果receipt、45项JUnit、真实父检查、原源码`.py.txt`和CPU根因检查证据。没有PT下载；索引保留每个归档字节SHA。结论仅限这次身份门工程修复，不证明有序关系机制有效。
