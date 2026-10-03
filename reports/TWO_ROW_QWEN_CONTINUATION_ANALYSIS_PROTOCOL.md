# 封存 common-head frozen / LoRA 配对分析协议

此入口只分析已完成结果，不参与选模或重新生成。新增文件为 `scripts/analyze_observed_qwen_continuation.py`；原训练器、模型和评价协议不变。当前只有纯解析检查，尚未运行真实两臂分析。

运行前必须同时具有 frozen、lora 的 completed3000 与各自 completed fixed-last285。先检查四个状态，任何 paused/failed/缺失直接拒绝，不读取一臂中间质量得出结论。共同初始模型固定 composite108 last12000，96,000条追加抽样/384,000个训练候选状态每臂，12次原旧DEV选择，每次36输入×K4，另285×K4固定TRAIN诊断。

## 来源与预算核验

- 固定285 TRAIN ID、36旧DEV ID和顺序，283220的3个缺输入保持请求分母288，不补父、不解封新DEV。
- 两臂的config除arm必须完全相同，含source、数据/几何、prefix指纹、质量audit、draw SHA、优化器设置与预算。实际初始头全部tensor hash及公共基础权重hash相同；LoRA q/v wrapper名称规范化后比对冻结基础权重，不把新adapter参数误当共同参数。
- 共享draw plan重新按已注册PCG seed100000构造并逐项一致；两份持久账本的全部195,864条issued逐顺序对应32个tail/head、optimizer及预定DEV tail/head，不仅检查总数。完成证据认证返回；失败调用不能从账本删除后凑预算。
- 检查原best/last checkpoint文件SHA（只读字节，不反序列化/执行）、全部12份pool receipt和其tail features/prediction/per_scene/metrics等SHA。原严格大于选优、相等保留最早步规则复算；必须与summary的原best相同，禁止重选。
- fixed-last必须引用该臂实际last3000；当前头/adapter hash等于最后DEV池，285个tail/head请求顺序与完整池封存一致。
- 共同初始last12000的DEV与fixed-last TRAIN通过历史receipt核验；它是额外更新前上下文，不是同追加更新预算的对照。

## 输出与统计口径

best与last各保留36 DEV/144槽。固定last TRAIN分别报告旧189、新96及全部285，逐父、逐条件相互配对；保留所有候选失败、有效unknown和已分类重复。两臂互比是主要描述，初始模型对每臂的变化单列。

沿既有saved-pool规范报告TipValid、AnyTipValid、已分类Unique、已知正参考类型coverage及其非空分母、语义正确率、tip净空、事件、参考ADE与参考分母。逐候选有限性与保存NaN核对，按原六个bool条件复核TipValid，再算unknown/duplicate。目标正确×tip净空四格完整计数；失败谓词非互斥不相加。保留逐父胜/平/负与不可评价数，不用query槽索引配跨模型路线，不把unknown当不存在。

这只是从原检查结果复算算术，并非重新做几何验收：不打开RGB、depth、supervision、参考轨迹、验收几何或任何reserved原始数据。known-reference coverage直接聚合原逐场景值，不能称完整解集覆盖。所有12次DEV历史、两个阶段全12父×3目标×两臂×K4保存图均输出；图只有预测的XY/XZ与原有效/失败标识，不额外读取几何或参考叠图。非有限槽文字标记并保留预算，不隐藏。

新forward=0，新Qwen encoding=0，新规划search=0，新修复=0。原模型/检查器均不调用。仅允许读取checkpoint原字节计算SHA，不加载模型。

## 成本

读取两臂对应实际输出的所有record_job外层status，包含pause2、resume、独立fixed-last和任何失败attempt；每条保留PID、UTC、exit、source、SHA，实际外层墙钟求和。训练体summary累计和已恢复sealed DEV耗时另列，均嵌套于外层，不叠加。固定TRAIN诊断与正式训练分开。

共享prefix准备的321官方完整捕获+321原尾回放只收费一次，核actual manifest SHA、status/index与原外层记录（command output/source匹配），其body和outer不相加。公共last12000预训练与prefix准备均不重复算作两个arm各自新费用。基础历史最终缓存与技术探针仍是项目独立账目，不在此伪作本轮免费计算或重复叠加。没有真实全Qwen在线推理测量，因此在线时延保持null；训练replay速度不能称端到端优势。

## 独立执行

源码冻结后且负责人确认四stage全部完成，才运行一次新输出：

```bash
python -m scripts.analyze_observed_qwen_continuation \
 --runs /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_lora_continuation_v1 \
 --initial-run /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_composite108_v1 \
 --job-records /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_lora_continuation_v1/job_records \
 --draw-plan /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_lora_continuation_v1/shared_draw_plan.json \
 --training-source /home/wzy/dpvlm/route_set_v1/research_v2/releases/f41be1ff1a35b0eab3d36edf4ca23197359f4731 \
 --prefix-corpus /home/wzy/dpvlm/route_set_v1/data/observation_two_row_composite108_v1/qwen_prefix_corpus \
 --prefix-job-status <actual completed prefix record_job status path> \
 --output <fresh analysis directory>
```

所有输入原SHA在结束时再次验证；输出另存artifact index和分析源码SHA。单种子、反复使用的旧DEV；不能据本分析宣称核心集合方法、机器人执行/OOD或最终锁定测试证据。

本地 `--self-test` 已实际13项通过：严格best tie/score/完整12步、固定TRAIN分组、路径越界、逐ID配对重排不变、逐父胜平负、unknown不变负例、四格分母、四阶段完整状态前置、保存NaN决策拒绝；未读取实际partial训练结果，未运行服务器。
