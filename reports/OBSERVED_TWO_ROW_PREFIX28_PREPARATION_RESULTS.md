# 两排 prefix28：真实导出与 TRAIN 表示审计

固定源码 `ce548f43ab22e804a8b70dea2f8bf297e20c8b84` 的准备阶段已实际完成。2026-10-02 19:10 UTC，预登记的16TRAIN＋12DEV父全部闭合后导出84个真实图文输入、756个尝试记录和490条已接受正参考；缺输入数为0，没有失败父替换。TRAIN为48个输入、285条正参考；DEV为36个输入、205条正参考。本次归档仅复制准备元数据与TRAIN审计，不读取新的DEV参考数组或锁定内容。

| 实际检查 | 结果 |
|---|---:|
| 冻结服务器测试（更正命令后） | 83 passed，16.05秒，无skip |
| 导出进程 | exit0，2.719秒 |
| TRAIN审计进程 | exit0，3.766秒；审计内部1.461秒 |
| TRAIN终点逐坐标±5cm观测支撑 | 285/285 |
| 最大最近观测点L∞距离 | 1.778cm |
| 模型event-aware H24的原两排TipValid | 285/285 |
| raw→模型H24事件转换数变化 | 0/285 |

容量门禁通过，支持沿预登记的spatial-peak＋surface±5cm普通头继续实验。它只说明每条TRAIN正参考末点在该输出表示内有可见点支撑，以及这批模型H24重采样未破坏既有tip标准；不证明模型能从语言选择正确物体、学会路线集合、泛化或通过全机械臂执行。长弧全部保留：TRAIN raw最长5.895m、最高世界z为1.832m；模型H24最长5.677m、最高1.828m。没有截短、删难例或改变2cm/3cm评价口径。

首次测试命令误写不存在的 `tests/test_observed_geometry_training.py`，实际0项测试、**exit4**；原日志和状态完整保留。改为已存在的 `test_observed_geometry.py` 后使用独立run_id `targeted_tests_corrected`，83项实际通过，包含新增6项Torch两排评价与完整中断恢复测试。不能把首次失败计为通过。

证据目录：[准备产物索引](observed_two_row_prefix28_preparation_v1/artifact_index.json)。12个服务器小文件在复制前后核对SHA不变，并与本地逐字节校验；追加后的registry保留原失败项。早期14父机械gate原件保留，另从实际导出manifest提取28父gate，二者不混称同一次检查。

- export manifest SHA256：`5c14a6494d6f3ba887347f64767c72e8f1aa7c280cd4d43933063f42a53da484`
- TRAIN quality SHA256：`c6190a9c7987f28e207dfe24309d7d486365a06a832cae6a97cfeea52df2226a`
- 机械gate：28父闭合，blocked/duplicate/缺初图均为空；`raw_locked_opened=false`。

本记录仅覆盖准备阶段，不把已启动的普通训练或缓存当作完成结果。未启动/停止服务器实验作业，未修改全局状态文件。
