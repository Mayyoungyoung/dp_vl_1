# HAMSTER 单请求完整生成：已完成准备证据

固定源 `04ed74fbc2912af58d65c487ce9947ea5cc71dc1` 的442文件实际部署SHA全部通过。服务器私有 `.venv-hamster3d` 在 CPU0、`CUDA_VISIBLE_DEVICES=-1` 下完成48项测试，48通过、0跳过：新29项包含2个真实CPU Torch测试，旧probe19项回归。此处CPU小模型测试不构成官方HAMSTER完整生成或任务质量结果。

实际外层时间 `2026-10-03T06:01:03.980119+00:00` 至 `2026-10-03T06:01:08.351190+00:00`，PID 810525 / child 810530，exit0；外层 4.371071 秒，runner主体 4.237370468 秒，两者嵌套不相加。GPU小时0。JUnit SHA `4d8444712d0c02d432a0de7d91951950b6d1c4d4a61fa2e9831445e363a08e2b`，逐测试身份与冻结manifest精确一致。

保留此前root本地错误测试文件名调用的零测试XML，其错误原因来自root执行记录（XML自身只记录0收集项），没有把它改成通过。正确文件名的第二次本地执行为46通过、2个Torch测试因本机缺Torch跳过；随后这两个项目已由本次服务器真实运行通过。原文件、冻结配方、部署收据和本地/服务器收据原字节分别留存。归档没有重跑任何测试。

code-only archive SHA `fad564affbe3e299aac1d9efadcacbdc579920bed3ede0685206b799f5a075d7`；9个关键源/依赖原件与该包逐字节一致，Python源码以 `.py.txt` 归档避免pytest重复发现。完整442文件SHA清单及actual incoming runner/wrappers保留。基础权重、数据标签、运行中输出没有复制。

root在读取通过收据后另行启动了唯一TRAIN0完整请求，独立输出 `runs/hamster3d_pinned_full1_v1/probe`。本准备报告不读取或推断该请求结果；其真实生成、解析、失败和费用应另行归档。没有额外模型调用、候选或自动重试。

原件恢复路径在 `REMOTE_ARTIFACT_INDEX.json`；本地全部文件校验在 `LOCAL_ARTIFACT_INDEX.json`。本次只读归档源见 `archive_source.py.txt`。
