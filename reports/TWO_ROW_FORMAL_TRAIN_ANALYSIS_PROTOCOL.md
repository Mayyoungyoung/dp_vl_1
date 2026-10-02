# 正式两排语料首16 TRAIN全槽质量审计

分析协议：`two_row_formal_first16_train_all432_slots_v1`。采集源固定为 `262648796fd47b25a2051cc227b6838515b651c6`；本分析独立只读，不修改采集器、路线接受标准、分类规则或正式角色。

固定分析父为登记 index 0–15，即283200–283215，全部TRAIN。每父3目标、每目标9请求槽，分母16父/48目标/432槽。必须全部16个机械closure已存在且身份、source、metadata SHA、实际几何门禁核验通过后，才打开其中任一父的原始结果。不根据成功数量替换父或槽；实际几何不一致或重复的父保留在质量分母，并明确不可用于模型。

读取边界：全角色只允许既有 `live_layout_gate` 的机械metadata；原始图像、路线、初始化世界、成功/失败记录只读上述16 TRAIN目录。禁止读取DEV_MODEL、DEV_SCORE、CALIBRATION、TEST_LOCKED原始结果。相对路径越界、符号链接越界、hash变化拒绝。

`scripts/analyze_two_row_formal_train.py` 输出：

- `analysis.json`：16父闭合/初始化/全部成本，48目标有效参考数、已知关系数、重复数、unknown、R>K4支持、精确请求分母，源/hash与机械门禁。
- `all_432_slots.json`：全部固定槽，包括accepted、failed、interrupted、unattempted；不完整ledger尾部最多一槽的不确定性单列，不把它编造为确定未尝试。记录实际row crossing、guide误差、失败执行阶段与raw/H24复核。
- 48张目标图：每张九槽，每槽XY与XZ投影；失败路径和无路径槽均显示，统一按同目标所有路径确定图范围，不挑好例。
- 每个有初图的父复制一张原像素 `front.png`，保留字节SHA。输出artifact索引覆盖全部图及分析文件。

接受参考的raw、原采集H24、端点、实际通过关系按原函数重算，存储H24必须与原resample逐位一致。发生不一致则分析失败并保留证据，不能自动重标为有效。失败路线的tip代理可能通过，但仍保持原机器人失败，不将box-only检查提升为整机成功。unknown有效路线仍是正参考，既不算已知新类型也不删除。连续路径解数未知。

这项审计描述窄ID采集质量，不是方法优势。正参考进入当前模型重采样、可见表面支撑和长弧容量问题由独立 `audit_two_row_train_reference_quality.py` 分析；本脚本不截断或筛除参考。

冻结后实跑入口（CPU1，无模拟器/GPU）：

```bash
PYTHONPATH=<fixed-release> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 1 <python-with-numpy-matplotlib> -m scripts.analyze_two_row_formal_train \
  --corpus /home/wzy/dpvlm/route_set_v1/data/observed_two_row_formal116_v1 \
  --output <fresh-analysis-directory>
```

实际作业必须使用record_job记录源版本、PID、日志、退出码、运行时间；暂未执行。测试时可使用 `--no-plots`，正式审计不可省略48张图。
