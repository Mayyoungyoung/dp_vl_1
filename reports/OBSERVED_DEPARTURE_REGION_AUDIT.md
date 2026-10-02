# 观测离开区域：TRAIN 表示证伪结果

2026-10-02，固定源码 `d73a4f2da9459a663d2ba9769b889c1a2b3c2e78`。**否定当前连通边界区域表示，停止其分配器实现。** 预声明8父×3指令×3半径的72次构建全部成功且全部只有一个区域J=1，不存在可供联合分配的区域选择空间。这个结论限于当前表示与已审计TRAIN观测，不能推广为“场景只有一条有效路线”或所有拓扑表示无效。

## 协议和信息边界

固定旧32 TRAIN 的前8父272000–272007，全24条指令及其全部40条正参考；14条类型未知仍保留。先验取原TRAIN probe报告，SHA256 `2b228acc808a1544d3cb7fa8111203a47d427123ade4cac254ad9e22ad0ffc62`：原32父精确指令RGB-D颜色原型和181条TRAIN路径拟合的workspace均不重拟合。它不是Qwen或开放词汇定位；构图终点来自观测预测组件，不是真目标。

使用观测A* v2原自由/未知图、2.5cm体素、原起点接入、可见表面20mm阈值及原接触例外。半径0.15/0.25/0.35m按当前末端欧氏距离预声明。每个半径内全部自由节点若超过20000，构图前即记预算失败；不偷偷只计可达子集。26邻接的边仍检查全部supercover体素。边界定义为可达球内自由节点中有合法边通向球外自由节点者；在边界诱导子图取连通分量，不人为切角度/左右方向。

所有24份区域NPZ全部写入并保存 `generation_complete.json` 后，才打开正参考路径和类型做评价。真箱体、验收标签、未来路线、模式标签均不参与构图；完整路线搜索数为0。参考用精确首次离球点前一段至多sqrt(3)体素长的supercover与冻结边界相交映射，不强制最近区域归属。映射规则保守，不能把未映射参考判为真实无效。另报的“前缀栅格代理通过”只是原离散观测图检查，也不替代真实几何或原虚拟接入规则。

## 实测

| 半径 | J=1指令 | 生成失败 | 离球正参考 | 映射覆盖 | 映射且前缀代理通过 | 不同已知类型对：分离/共同映射 |
|---|---:|---:|---:|---:|---:|---:|
| 0.15m | 24/24 | 0 | 40/40 | 29/40 (72.5%) | 23/40 | 0/4 |
| 0.25m | 24/24 | 0 | 40/40 | 22/40 (55.0%) | 8/40 | 0/2 |
| 0.35m | 24/24 | 0 | 40/40 | 16/40 (40.0%) | 6/40 | 0/0，不可评价 |

每个映射只对应一个区域；可共同映射的不同类型全部塌缩。没有同类型成对参考可用于碎片率分母，因此不能声称碎片率得到可靠估计。0.35m零已知类型对也不能写成“类型正确率100%”。增大半径没有产生分支，参考覆盖反而降低；本审计没有用结果改半径、阈值或映射方式。

三个半径最大纳入节点496/2142/5653，均远低于20000，不是预算截断造成J1。总邻接边尝试5,099,328，supercover体素检查19,220,544；几何检查的成本已计入。单指令区域生成中位0.2570秒（含三个半径和共同前处理），最大2.2235秒；不能把这些测量当完整路径/VLM请求延迟。

CPU1/CUDA隐藏：selftest记录PID292173、child292174，exit0；正式audit PID292176、child292177，12:03:42.023366至12:03:51.592707 UTC，exit0，进程墙时9.569秒，内部实算9.346秒，GPU小时0。原源码与结果未改；没有在运行的本worker作业。

## 决定、复现和产物

**放弃此区域表示，不实现联合预算分配器，不补左右专家或碎片模块。** 下一步按根线程安排做TRAIN可见点支撑诊断，帮助普通局部读取基线选择预声明尺度；该诊断不把普通cross-attention/refinement写成新核心贡献。

服务器实测入口由冻结launcher执行，完整命令及退出状态见[作业记录](observed_departure_regions_train8_v1/jobs/audit.status.json)。无resume；需要复现时必须换新输出目录。数据为 `data/obstacle_learning_curve_new32_v1`，运行源码为上面的immutable release；环境为 `.venv/bin/python`，`CUDA_VISIBLE_DEVICES=-1`，OMP/MKL/OPENBLAS/NUMEXPR线程均1。

```bash
python -m scripts.audit_observed_departure_regions --self-test
python -m scripts.audit_observed_departure_regions \
  --data /home/wzy/dpvlm/route_set_v1/data/obstacle_learning_curve_new32_v1 \
  --output /home/wzy/dpvlm/route_set_v1/runs/observed_departure_regions_train8_v1_reproduction \
  --prior-report /home/wzy/dpvlm/route_set_v1/runs/observation_multiroute_astar_obstacle_new32_v2/train_probe_reexport/report.json \
  --prior-report-sha256 2b228acc808a1544d3cb7fa8111203a47d427123ade4cac254ad9e22ad0ffc62
```

- [机器结果](observed_departure_regions_train8_v1/summary.json)、[逐参考评价](observed_departure_regions_train8_v1/reference_evaluation.json)、[成本细节](observed_departure_regions_train8_v1/diagnostic_summary.json)。
- [24份NPZ索引与SHA256](observed_departure_regions_train8_v1/artifact_index.json)全部下载后核验通过；二进制保留于本地/服务器 `runs/observed_departure_regions_train8_v1`，不进普通Git。无训练checkpoint，因为本轮没有模型训练。
- 实际部署审计源码SHA `23a6a11cd19375a7eeb602cc7507f0490ab756708cb955207f6fc0e1e5a88287`；换行规范化后与Git blob `3151cc1327722446636763708c475343cd31415362f3e8983ce625089ba4fb3b`严格一致。
- [固定首父target0的边界图](observed_departure_regions_train8_v1/first_parent_boundary.png)，仅展示观测区域与正参考首次离球点，类型未知照实保留；[本地哈希复核与绘图入口](observed_departure_regions_train8_v1/reproduce_archive.py)。
