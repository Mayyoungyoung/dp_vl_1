# 两排正式语料：首16 TRAIN全请求槽实测

固定首16 TRAIN父（283200–283215）已完整闭合，432个请求槽全部实际尝试，得到285条接受参考（65.97%）。48个目标条件中14个有超过K=4的已知不同路线类型。这个结果支持开始窄ID集合生成的同数据对照；它本身不是学习方法优势，也不是广泛机器人泛化证据。

采集源 `262648796fd47b25a2051cc227b6838515b651c6`；分析源 `ce548f43ab22e804a8b70dea2f8bf297e20c8b84`。分析作业 `observed_two_row_formal_train16_analysis_v1/analysis`，PID496182/child496183，2026-10-02 18:53:33.054–18:55:20.498 UTC，exit0，107.44s。CPU core1、线程1、GPU隐藏；未开启新模拟器。实际服务器83项相关测试全部通过，包含本分析12项（Linux symlink测试未跳过）。

[原始汇总](observed_two_row_formal_train16_v1/analysis/analysis.json)、[432槽逐项记录](observed_two_row_formal_train16_v1/analysis/all_432_slots.json)、[全48图索引](observed_two_row_formal_train16_v1/FIGURES.md)、[运行记录](observed_two_row_formal_train16_v1/analysis.status.json)。

## 固定分母与质量

| 项目 | 实测 |
|---|---:|
| 登记/完成父 | 16/16 |
| 初始静态与观测门禁通过 | 16/16 |
| 同一初图对应3条不同目标指令 | 16/16 |
| 请求/已尝试/未尝试路线槽 | 432/432/0 |
| 严格逐候选恢复通过 | 432/432 |
| 接受参考/失败 | 285/147 |
| 已知关系参考 | 181：175 lateral、6含over |
| 有效unknown参考 | 104，占接受参考36.49% |
| 条件内已知类型重复参考 | 1 |
| 有R>K4已知支持的条件 | 14/48（29.17%） |
| 当前机械重复/模型封闭父 | 0/0 |

已知R表示已采到并按原实际穿越规则分类的不同正例关系，**不是全部连续解法数**。unknown仍是有效正参考，不删除、不监督为不存在。条件 `283205_target2` 有4条unknown正参考、R=0，不能称为无解。机械重复门禁是本次分析时刻的快照；后续模型使用还需重新核验整个已闭合语料。

| 已知R | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 48条件计数 | 1 | 0 | 7 | 13 | 13 | 10 | 3 | 1 |

## 失败分解与表示边界

147次失败中33次为真实模拟步机器人碰撞，114次被原端点/raw/H24净距或类型一致性判据拒绝。没有规划阶段返回失败、没有严格恢复失败。本次分析没有改变接受标准，也没有把失败轨迹通过tip代理的情况改称机器人成功。

114次接受判据拒绝中的非互斥计数：端点超过3cm为0，raw完整线段tip净距失败61，H24 tip净距失败61，raw/H24实际类型改变50。组合计数如下，总和恰为114。

| 同一次拒绝的触发组合 | 次数 |
|---|---:|
| raw tip + H24 tip | 55 |
| 仅类型改变 | 48 |
| 仅H24 tip | 5 |
| 仅raw tip | 4 |
| raw tip + H24 tip + 类型改变 | 1 |
| raw tip + 类型改变 | 1 |

这说明参考采集的主要损失来自几何/表示稳定性，而非端点到达。这里的数值只是保持原协议的失败诊断，不能据此放宽H24或分类门槛。已接受285条raw与原采集H24逐位复核一致，并全部再次通过原几何和类型判据。

接受路径长度最小0.574m、均值1.479m、中位1.351m、95分位2.630m、最大5.895m；40条超过2m、11条超过3m。全部长弧保留，不截断、不作为长度最优示范。104条unknown中80条实际穿越记录含over，43条含原规则的ambiguous-height区域（这些谓词可重叠），图中还存在绕回；不将guide编号冒充实际类型。具体crossing xyz、方向、弧长进度和分段误差保存在432槽记录。

原采集H24检查与模型的事件分段重采样检查属于两个记录环节；后者和末端观测支撑由独立TRAIN-only表示审计负责。本报告不替代该模型容量门禁。所有tip检查只约束末端折线，不能证明连续整机净距。

## 实际TRAIN语言颜色覆盖

只从上述16父已保存的 `observations.jsonl` 读取真实指令，逐文件SHA与本次全槽分析记录一致。48条件覆盖官方20色中的17色，未见gray、red、teal；未读取DEV语言来调整训练或采集。完整48条指令与hash在[颜色审计](observed_two_row_formal_train16_v1/train_instruction_color_audit.json)。

| 颜色 | 条件数 | 颜色 | 条件数 |
|---|---:|---|---:|
| azure | 4 | black | 4 |
| blue | 1 | cyan | 2 |
| gray | 0 | green | 4 |
| lime | 5 | magenta | 4 |
| maroon | 2 | navy | 2 |
| olive | 4 | orange | 2 |
| purple | 3 | red | 0 |
| rose | 3 | silver | 1 |
| teal | 0 | violet | 2 |
| white | 4 | yellow | 1 |

这为随后比较Qwen和精确指令匹配的传统颜色原型提供TRAIN覆盖边界，不是任何方法已学会颜色或语义泛化的证据。不因颜色稀有或缺失补抽/替换父。

## 成本、图和复现

16父累计worker墙钟2614.41s（两CPU分片之和，并非端到端elapsed）；显式路线 `get_path` 3654次，规划322.19s、模拟推进656.55s。每槽最多9段提案，内部IK/OMPL嵌套调用不算额外完整候选，也不冒称没有内部搜索。初始化、观察、恢复、写盘等其余成本留在worker总时间中。分析没有使用GPU，GPU小时为0。

全部48目标九槽图和16张原像素初图都已同步；8页contact sheet按登记顺序覆盖全部48图，并已逐页做可视检查。每槽XY/XZ成对，绿色接受、红色失败或partial；失败图即使显示实际关系文字，也只是诊断信息。灰框是投影，单幅投影重叠不等于3D碰撞。长弧按同一目标九槽的完整范围展示，没有裁掉离群路径。[图索引与逐条件支持](observed_two_row_formal_train16_v1/FIGURES.md)。

服务器原始语料：`/home/wzy/dpvlm/route_set_v1/data/observed_two_row_formal116_v1/parents/TRAIN/`。本分析只读取index0–15原始数据；全role仅既有机械hash门禁，未读取DEV_MODEL及以上原始结果。重跑命令保存在record_job状态文件，使用相同source和新的output，不能覆盖原输出：

```bash
cd /home/wzy/dpvlm/route_set_v1/research_v2/releases/ce548f43ab22e804a8b70dea2f8bf297e20c8b84
PYTHONPATH="$PWD:$PWD/scripts" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 1 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.analyze_two_row_formal_train \
  --corpus /home/wzy/dpvlm/route_set_v1/data/observed_two_row_formal116_v1 --output <fresh-output>
```

传输原始分析archive为15,083,520 bytes，SHA256 `44d694ccc3dcc06a533b809026c762a5b0325b59d99ba379362360838473867e`，双端一致。66个原始分析artifact SHA全部核验通过；颜色补审和本地contact sheet/组合统计另有来源与最终index，未覆盖原分析artifact索引。原分析SHA `610be4fcf4a2d75448d894b223c0c5e1a826d6409004d1858b42b28fd7e8261d`。

下一步维持登记语料和全部参考，执行相同输入/候选/训练预算下的普通集合头基线，再根据生成的几何质量与已知覆盖瓶颈选择机制。全48条件和按已知R分层结果同时报告，不能只用14个R>K4条件宣称整体收益。正式采集继续，锁定数据保持封存。
