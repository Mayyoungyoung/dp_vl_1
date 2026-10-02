# 两排 prefix28：传统观测 A*v2 的真实结果

固定源码 `743d9b269c27054b05c48cc5bab15017ab3fae55` 的10项服务器测试实际通过（0.64秒），随后原参数A*v2完成全部12DEV父、36个实际图文条件、144个候选槽，exit0。记录PID516428/child516435，2026-10-02 19:23:27–19:23:53 UTC，实际单CPU affinity `[1]`、OMP/OpenBLAS/MKL均1线程；GPU小时0。此前0d730换行SHA门禁失败及修复证据完整保留。

输入仅RGB-D、相机、当前末端/open及精确语言。颜色原型和workspace只从同一16TRAIN/285条已知正参考拟合，三个旧planner源码未改；没有Qwen、DEV真实目标/障碍/模式输入、额外候选筛选或真值修复。每请求四槽先封存SHA，再用原两排2cm/3cm标准检查。

## 全部候选及成本

| 指标 | 实测 |
|---|---:|
| TipValid@4 | 76/144，52.7778% |
| AnyTipValid@4 | 19/36，52.7778% |
| UniqueClassifiedTipValid@4 | 0.527778/条件；19个有效条件均为1 |
| 已分类有效类型重复槽 | 47，均值1.305556/全部36条件 |
| 有效unknown类型槽 | 10，全部保留 |
| KnownReferenceTypeCoverage@4 | 3.19444%，36条件均有已知正类型 |
| 实际找到完整路径 / 失败槽 | 76 / 68 |
| 精确网格路径重复 | 0（不代表类型不重复） |
| 完整逐请求中位 / p95 | 0.779898 / 1.571641秒 |
| 首次请求 / 全36请求合计 | 1.192036 / 23.640952秒 |
| TRAIN拟合 | 1.661799秒 |
| 整个子进程墙钟 | 26.155872秒 |

逐请求包括IO/hash、反投影、颜色定位、网格与attachment、最多4次搜索、原proxy检查、输出封存及两排label/check。原2秒是每64展开节点检查的搜索预算，不含attachment等额外成本，也不是硬墙钟截止。它是同K4、不同时间成本的传统控制，不能称为与普通头等时或同骨干比较。SelectedTipValid和全机器人执行保持null。所有76条成功产生的路径均通过tip检查；其条件均值末端中心误差2.531cm、路径长度0.548m是有限预测条件下的统计，不能当全部36条件的无条件误差。

## 68个失败槽的完整来源

| 类别 | 条件 / 槽 | 证据 |
|---|---:|---|
| 未支持的精确语言 | 8 / 32 | red5、teal2、gray1；TRAIN只形成17种颜色原型，没有用DEV补齐 |
| 无可接受目标连接 | 9 / 36 | 原目标attachment均返回0，没有后续替代搜索 |
| 找到路径 | 19 / 76 | 四次搜索均完成并通过原tip检查；没有找到后再筛掉的候选 |

9个连接失败中，`283264_target1` 的原型定位错到另一物体，目标中心误差82.86cm且局部无free attachment。其余8个目标定位满足原身份＋3cm标准（误差2.542–2.593cm），但43个原局部free连接候选全部被原20mm非目标可见点线段检查拒绝；其中1个同时未通过ray检查。它说明失败发生在哪个固定代理判据，不证明该判据错误，也没有以DEV结果放宽接触区或阈值。

完整36条件及9个连接失败细节见 [保存结果诊断](observation_two_row_astar_v2/saved_failure_and_duplicate_analysis.json)。没有缺失输入；84请求输入、36DEV输入及144DEV请求槽的分母均完整。

## 所有19个有效条件的类型重复空间

19个有效条件都有且仅有一个**已分类**类型 `middle→middle`；另有10条有效unknown，不能把它们当失败或臆定为重复。边惩罚产生了不同网格折线，却未覆盖不同已分类通道组合。

下表只计算 `min(已分类有效重复槽数，当前未覆盖的已知正参考类型数)`。这是使用参考标签的计数上界，**没有检索/替换任何预测，没有产生额外候选，也不代表观测模型或固定搜索预算能实现相应收益**。

| 条件（省略two_row_reach） | 已知正类型数 | 重复槽 | 有效unknown槽 | 已知缺失类型 | 可替换计数上界 |
|---|---:|---:|---:|---:|---:|
| 283264_target0 | 4 | 3 | 0 | 4 | 3 |
| 283265_target0 | 5 | 3 | 0 | 4 | 3 |
| 283265_target1 | 4 | 2 | 1 | 3 | 2 |
| 283266_target0 | 3 | 2 | 1 | 3 | 2 |
| 283267_target0 | 3 | 3 | 0 | 3 | 3 |
| 283267_target1 | 1 | 2 | 1 | 1 | 1 |
| 283267_target2 | 3 | 2 | 1 | 3 | 2 |
| 283268_target0 | 3 | 3 | 0 | 3 | 3 |
| 283268_target1 | 6 | 2 | 1 | 5 | 2 |
| 283269_target1 | 3 | 3 | 0 | 3 | 3 |
| 283270_target1 | 1 | 2 | 1 | 1 | 1 |
| 283271_target0 | 3 | 3 | 0 | 3 | 3 |
| 283272_target1 | 2 | 2 | 1 | 2 | 2 |
| 283273_target0 | 2 | 3 | 0 | 2 | 2 |
| 283273_target1 | 4 | 2 | 1 | 4 | 2 |
| 283274_target0 | 2 | 2 | 1 | 2 | 2 |
| 283274_target2 | 6 | 2 | 1 | 6 | 2 |
| 283275_target0 | 5 | 3 | 0 | 4 | 3 |
| 283275_target1 | 3 | 3 | 0 | 2 | 2 |

全部19条件都有正的上界，合计43个额外已分类类型计数；若仅作标签反事实、保持失败和unknown槽不变，Unique均值上界为 `(19+43)/36=1.72222`，不是实际方法结果。已知正类型数1–6同样不是“全部可行解数”。尤其14/19条件的有效 `middle→middle` **不在已采参考类型中**，仍须按实际几何有效计入；这也解释低ReferenceCoverage不能单独认定其输出错误。未匹配预测不能据此作为不存在/无效负标签。

这一结果支持继续检验实际类型覆盖瓶颈：它同时保留了传统控制的较高tip有效率、明确的封闭语言缺口和类型重复成本。它尚未证明新的集合机制有效，也不授权把DEV模式标签输入规划器、调搜索惩罚或按参考筛选候选。

## 可复核产物

服务器原始目录：`/home/wzy/dpvlm/route_set_v1/runs/observation_two_row_astar_v2/dev_model`。完整report、拟合模型、逐请求result/seal/status和日志已同步；184个实际产物逐项SHA/字节数验证。73个NPZ原件位于本地被Git忽略的 `runs/observation_two_row_astar_v2/dev_model`，位置映射见 [本地/服务器索引](observation_two_row_astar_v2/local_artifact_index.json)。[同步收据](observation_two_row_astar_v2/sync_receipt.json) 保留输入manifest SHA及服务器根文件SHA，原source LF/CRLF实际字节证明保留在report。

局部只读诊断命令如下，运行时不调用任何planner/model，也不打开参考轨迹数组：

```bash
python -m scripts.analyze_two_row_astar_results \
  --run runs/observation_two_row_astar_v2/dev_model \
  --metadata .bootstrap/two_row_astar_actual743d/input_metadata \
  --output reports/observation_two_row_astar_v2/saved_failure_and_duplicate_analysis.json
```

上述实际输出已存在；重现须使用新的output路径，不能覆盖原诊断。正式CPU评估的精确命令、source commit和退出状态均在 `dev_model.status.json`，本轮未追加任何搜索、训练或测试集选择。
