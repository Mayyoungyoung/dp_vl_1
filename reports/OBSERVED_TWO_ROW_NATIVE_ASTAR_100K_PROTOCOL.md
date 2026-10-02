# 两排传统观察规划：唯一100k节点预算检查（预登记）

本轮只回答：两臂共同扩大搜索节点额度后，原20k上限截断的空间惩罚候选是否能返回，以及质量、已知类型覆盖和实际成本怎样变化。它是传统强基线预算检查，不是新方法；100000是20000的五倍，不作同节点预算或固定端到端时间优势声明。只做这一档，不扫描节点上限、空间场或时限。

## 固定输入与预算

- 唯一数据为 `data/observation_two_row_prefix44_v1`，export SHA `6ed7786823275f26dba38fff9039bd33127b71567de4e4c0c80a37cfd91e46f9`。TRAIN登记32父，实际31父/93输入；缺父不替换。
- 原闭集精确指令颜色原型和workspace按相同全部TRAIN正参考拟合；canonical SHA必须为 `384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b`，只排除实测training_seconds。任务/目标坐标/障碍完整几何/路线模式不进入规划。DEV真值仅在候选封存后验收。
- 先固定TRAIN indices0..3、12输入，edge/spatial两臂共24请求96槽；机械门通过并由root审阅后，独立执行DEV indices64..75、36输入，两臂共72请求288槽。共96请求384槽上限，不合并池、不补失败、不筛掉重复或unknown。没有新Qwen、训练或仿真调用。
- 两臂共同固定K4/H24、100000节点、2秒搜索时限、每64个expanded nodes查时。保持原.025m体素、26邻居及supercover顺序、虚拟端点连接、heap tie顺序、精确edge计数、惩罚4、空间Gaussian sigma .05m。仅节点上限变化。
- 空间场仍由所有此前返回的完整raw轨迹（包括后验失败路线）构造，不能用验收结果筛选。首路径零场，两臂raw/H24必须逐元素完全相等；该机械门不按质量决定继续，也不证明一般实现等价。

## 隔离生产实现与配置语义

新增 `observation_native_astar_100k.ProductionSearch100K` 使用自己的固定100k/2s构造器，不调用旧20k构造器后修改字段，不走test_only入口。只复用原C++ ABI和原适配器的数组装配/search调用；原C++、旧20k adapter/runner/config/tests/launcher及原规划/空间场/验收器不修改。新 `TestSearch100K` 仅接受测试receipt；正式context只构造ProductionSearch100K且拒绝test receipt。

旧 `planner.CONFIG.maximum_expanded_nodes_per_candidate=20000` 作为冻结历史基础配置保留，用于原几何和SpatialSearch来源校验，不作为本轮执行额度。新下层搜索接口显式接收100000；每个实际搜索attempt必须记录并断言100000/2s/64、正式binary SHA及新预算协议。总报告和每请求wrapper分别记录 `base_planner_config_historical` 与 `effective_search_config`，明确历史20k未被执行。没有临时修改全局CONFIG绕过旧校验。

每个新commit在 `research_v2/native_builds/<commit>` 私有目录编译相同原C++。receipt保存实际源码/原ABI adapter/编译器/flags/版本及binary SHA，另由阶段source inventory绑定新预算adapter和runner/config。生产不复用test_only库，两臂共用同一生产库。不能覆盖已有未封存构建，不能自动Python回退或重试。

## 三阶段门禁与复现

冻结后从该release分别调用：

```bash
taskset -c 1 bash scripts/launch_observed_two_row_native_astar_100k_v1.sh <commit> tests
taskset -c 1 bash scripts/launch_observed_two_row_native_astar_100k_v1.sh <commit> train_preflight
taskset -c 1 bash scripts/launch_observed_two_row_native_astar_100k_v1.sh <commit> dev
```

CPU亲和性必须只有一个核，四个数值线程变量均1，GPU隐藏。每阶段独立record_job PID/日志/status/退出码，任何已有stage不重放；上一stage需实际exit0及全部artifact/source/binary哈希一致，失败保留中断请求与已消费/尚未尝试槽。不存在自动连跑下一阶段或失败候选恢复；修复须新独立协议/输出。

`tests` 必须在实际Linux同时通过旧24个native tests、新12个100k tests和全部旧/新runner、空间场、A*及已保存分析回归，无skip。JUnit必须出现精确独立的24旧native与12新native testcases，不能用其他测试凑数。新12项包括8个Python/native完整路径和确定性计数差分、真实100000节点上限（同时2s/64）、正式/测试receipt隔离、不可变额度和context恢复。两个独立实际test compiler receipts均需与正式构建cpp/adapter/compiler/flags一致。

本地Windows/MinGW实际108 tests已通过（pytest4.86秒；外层6.04秒）；这不代替冻结Linux验证，也不是任何真实100k规划结果。

## 保存与分析

保留所有原始raw/H24/open池、失败/unknown/已知类型重复、每次节点数/状态/耗时、逐父配对和完整请求分母。报告TipValid、AnyTipValid、UniqueClassifiedTipValid、已知正参考覆盖、重复和unknown；未知类型不当不存在/负例，有限参考集不叫全部可行解。

两臂重新实测IO、反投影、定位、端点连接、field、ctypes、search、路径转换、候选封存及验收全部请求walltime。ctypes准备计入2秒；field在搜索时限外，但全计入实际请求。新adapter预算标记及context构造计入请求外层；返回后额外receipt写入/source一致性核对属于阶段bookkeeping，体现在阶段总耗时，不能沿用旧20k延迟。编译、TRAIN拟合和整个阶段耗时分别记录；候选与共享费用不重复计账。100k变化仅有搜索内时限共同固定，并非固定端到端时间预算。

依实际结果区分node cap、time cap、attachment、有限无效路径及类型重复，不把更大额度称为算法收益。后续与旧20k结果比较须同时披露五倍节点上限及实际成本；本轮不会自动增加下一档或修改field。
