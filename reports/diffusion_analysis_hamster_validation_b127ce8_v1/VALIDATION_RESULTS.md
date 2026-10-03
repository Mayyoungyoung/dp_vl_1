# b127 扩散分析与 HAMSTER 探针工程验证

冻结源码 `b127ce8514a63917cc3f1b7278db8f65289ba985` 的服务器私有 `.venv-qwen` / Python3.11 实际 **32 passed、0 skipped、0 failed**：扩散保存池分析13项、HAMSTER探针纯契约19项。全部为纯测试，没有载入模型或真实数据数组；不是模型技术探针或质量结果。

CPU0单核、单线程、`CUDA_VISIBLE_DEVICES=''`，GPU小时0。record_job PID757336/child757341，`2026-10-03T04:08:17.438325+00:00` → `2026-10-03T04:08:19.193841+00:00`，exit0。验证recipe主体1.659495424秒，外层1.755516秒，两者嵌套不相加。JUnit精确32个用例身份与13/19分组已重新核查，没有执行测试重跑。

实际420份源文件逐SHA与代码包一致，部署收据也明确420/全部incoming匹配。代码包5,683,200字节，SHA `555bce082b7ea0799acbbeb59f5609c7c840b8e1665ff3ef6d44679693ea9ec9`；原包保留 `.bootstrap` 与服务器incoming，索引其位置和hash，不重复进入普通Git。这里归档实际测试stdout/status/registry/JUnit/identity/receipt、八份关键source原字节、runner与三个独立wrapper、完整420文件manifest和本地冻结/部署收据。`.py`副本追加`.txt`以防pytest收集。

本次只读归档未运行分析、HAMSTER模型、资产或环境阶段，未读取运行中的训练结果。analysis与probe wrapper分别要求实际32项验证完成；analysis仍须两臂全部10阶段封存，probe仍须资产与私有环境两完成收据及GPU排队。三个入口不自动串行触发。
