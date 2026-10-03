# Common-head 续训真实工程验证

固定源码 `f41be1ff1a35b0eab3d36edf4ca23197359f4731` 的服务器私有 `.venv-qwen` 验证实际 **79 passed，0 failed，0 skipped**。JUnit计时6.738秒（pytest文本四舍五入6.74秒）；外层record_job为8.206423秒。

| 组件 | 实际通过 |
|---|---:|
| common-head续训与恢复 | 26 |
| 冻结prefix corpus | 32 |
| 官方Qwen prefix replay | 21 |

续训包含真实Torch frozen/LoRA两个B1×32连续4步与2步保存恢复后2步完整状态等价测试，以及optimizer后/原子checkpoint rename故障不能静默重放的测试。本地曾跳过的Torch与Linux符号链接测试此次均实际执行，没有将skip计为通过。完整用例见 `validation/pytest.xml`，原命令、日志和状态未改。

record_job PID686844/child686846，2026-10-03 01:42:39.333634–01:42:47.540057 UTC，exit0。tests分支 `CUDA_VISIBLE_DEVICES=''`、各线程上限1；状态中的GPU UUID是共享启动环境字段，并不表示这些CPU测试使用GPU。测试GPU小时0。工程测试没有产生真实任务路线质量指标，也不证明3000步训练会有收益。

实际部署归档SHA `7b1eb86aaf659f0d311056e1113971a41b251428a3d29baab873d96c356efc77`，5,488,640字节；冻结wrapper `qwen_continuation_f41be1f.sh` SHA `4993e6e60538068892bb877550e86169a953249a9d427ec2db2c5bfa24dfd0e8`，2,106字节。两者与本地对应冻结文件相同。归档含5个实施文件、实际依赖/三套测试/record_job，共25份源码，逐文件与实际部署tar字节核对；另4份validation原件和1份wrapper，共30份远端原件。`FREEZE_METADATA.json`、`REMOTE_ARTIFACT_INDEX.json`、`SYNC_RECEIPT.json` 给出服务器POSIX路径与所有SHA。

实际formal缓存指纹 `4a3152db319e67c158f692a4b88e8575fd5e00b7ebe483ebfe4fafec1552bf15` 已固定；恢复源码保留按已消费调用fail-closed的门禁、完整DEV池零forward复用及成本计账。`--stop-after`只在完整更新/应有DEV完成后写paused，不改总3000步或写completed摘要。

负责人已另行启动frozen的pause2训练验证；本归档未打开其正在生成的文件，不预报训练状态或质量。无权重/PT下载、无额外模型forward、规划或仿真，无新DEV/reserved原始数据读取。这里只保存完成的工程验证，后续实际训练证据另归档。
