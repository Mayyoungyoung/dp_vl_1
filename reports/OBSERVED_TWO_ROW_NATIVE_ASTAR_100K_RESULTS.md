# 两排观察规划：100k节点强传统控制实际结果

唯一100k预算检查已实际完成。DEV两臂各144槽均96条TipValid、Any24/36；空间惩罚将已知有效不同类型总数从24增至62、已知重复从65降至34，代价是单请求中位时间从0.708增至1.221秒。它是标准传统基线的质量—覆盖—成本结果，不是新的学习方法贡献，也不是固定端到端时间优势。

## 实际来源、分母与退出

冻结运行源 `42904f0d24be0357af2257f280967383060170e0`，新family `runs/observed_two_row_native_astar_100k_v1`。只用原prefix44：注册32TRAIN父、实际31父93指令，same canonical fit `384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b`；原缺父保留。TRAIN预检固定前4父12条件；DEV固定原12父36条件，均为既有开发数据。无新增模型训练/前向/仿真/候选重试，无SCORE/CAL/LOCKED读取。

| 阶段 | PID/child | UTC开始→退出 | 实际预算 | 状态 |
|---|---|---|---|---|
| Linux tests/build | 599869/599871 | 22:27:55.740899→22:28:00.461793 | 108 tests，0 skip，含旧24+新12 | exit0 |
| TRAIN preflight | 605656/605659 | 22:36:09.466758→22:36:28.700130 | 24请求96槽 | exit0 |
| DEV | 606180/606183 | 22:37:25.547787→22:38:23.574695 | 72请求288槽 | exit0 |

GPU隐藏、CPU亲和性1核、四个数值线程变量均1，GPU小时0。每阶段单独record_job；TRAIN `mechanical_preflight_passed=true`，12首raw/H24全等。DEV此字段为false仅因为它不是TRAIN阶段；其已通过的TRAIN gate/hash引用完整，36首raw/H24全等，不是门禁失败。

两臂共同每槽100000节点、2秒、每64节点查时，K4/H24、体素.025、惩罚4、Gaussian sigma.05及所有几何/定位/验收标准保持。100k是历史20k的五倍**允许节点预算**，不能与旧20k声称同节点额度。`base_planner_config_historical`中20k仅历史值；`effective_search_config`和全部实际attempt均核验100k/2s/64。

## 真实结果

| 集合/节点额度 | 臂 | TipValid | AnyValid | 已知Unique总数/条件数 | 已知重复总数 | 有效unknown | 正参考类型Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| TRAIN12/100k | edge | 32/48=66.67% | 8/12 | 8/12=.6667 | 22 | 2 | .081944 |
| TRAIN12/100k | spatial | 32/48=66.67% | 8/12 | 21/12=1.7500 | 11 | 0 | .119444 |
| DEV36/20k（历史） | edge | 96/144=66.67% | 24/36 | 24/36=.6667 | 65 | 7 | .031944 |
| DEV36/20k（历史） | spatial | 74/144=51.39% | 24/36 | 52/36=1.4444 | 22 | 0 | .054630 |
| DEV36/100k | edge | 96/144=66.67% | 24/36 | 24/36=.6667 | 65 | 7 | .031944 |
| DEV36/100k | spatial | 96/144=66.67% | 24/36 | 62/36=1.7222 | 34 | 0 | .068519 |

100k空间臂24个有效条件均4/4有效，其中14条件有3种已知类型、10条件有2种，其余12条件没有有效候选。96有效槽类型为47 middle/middle、34 over/over、11 negative/negative、4 positive/positive。仍有34个已知重复槽。与edge相比，每个父的已知有效类型总数均增加，父内有效槽数保持；逐父保留，不只汇报总体：

| 父ID | edge/spatial有效槽 | 已知Unique总数 | 已知重复总数 |
|---|---:|---:|---:|
| 283264 | 4/4 | 1→3 | 3→1 |
| 283265 | 8/8 | 2→5 | 5→3 |
| 283266 | 8/8 | 2→4 | 6→4 |
| 283267 | 12/12 | 3→8 | 7→4 |
| 283268 | 8/8 | 2→5 | 5→3 |
| 283269 | 8/8 | 2→6 | 5→2 |
| 283270 | 8/8 | 2→6 | 6→2 |
| 283271 | 8/8 | 2→4 | 6→4 |
| 283272 | 8/8 | 2→5 | 6→3 |
| 283273 | 8/8 | 2→5 | 6→3 |
| 283274 | 12/12 | 3→8 | 8→4 |
| 283275 | 4/4 | 1→3 | 2→1 |

类型由原场景定义给出，已知类型Coverage只对采得的正参考类型计算，不是全部连续解覆盖率。unknown仍保留为有效正例，不能因空间臂unknown为0宣称它覆盖了全部可行路线。

## 旧24个cap及所有失败

20k空间臂DEV的24个节点上限失败，在100k的对应ID/slot全部返回完整raw：22条通过原TipValid，2条仍未通过语义目标容差（`283264_target1`的slot2/3）。其搜索节点20111–54816，中位26835.5；search中位23.693ms，最大47.146ms，24槽共0.603703秒。无100k节点上限或2秒超时。TRAIN旧8个cap也全变为有效路线。

DEV两臂同有44个endpoint attachment失败（11个条件×4槽），另同有4条有限但目标错误路线（`283264_target1`全部4槽）。全部有限路线都通过原tip线段检查；这里不能把attachment失败称为真实无解，或将tip通过称为全机器人执行成功。两臂100条raw、96条有效、44个NaN槽全部计入144预算。

回收早先缺失raw会改变后续空间场历史，因此逐槽转移是保存池的ID对应描述，不把后续slot当相同内部状态下的单次搜索因果实验。全144空间槽的原/新status、验收决定、raw/H24差异和完整attempt均保存。

## 20k与100k一致性及成本

edge在TRAIN48槽+DEV144槽，共192槽的完整raw、H24、open**逐值及dtype/字节全相同**；checker全部决定及实际确定性搜索计数192/192相同。只把明确预登记的`maximum_expanded_nodes`20k→100k排除出“计数相同”，搜索实际expanded_nodes仍逐槽严格相等；没有新增search。真实compiled binary同历史SHA `295a5e662d1a8726a9960c7d786006940fd1a312494bde9b442869b67f22f8d1`。

| 阶段 | edge请求median/p95/总秒 | spatial请求median/p95/总秒 | spatial field总秒 | shared TRAIN fit秒 | 阶段内部总秒 |
|---|---|---|---:|---:|---:|
| TRAIN100k | .635600/.659400/5.810502 | 1.108055/1.156534/9.654959 | 3.239619 | 2.841573 | 18.725380 |
| DEV100k | .707706/.772939/20.259991 | 1.220680/1.382798/33.997692 | 11.574150 | 2.904472 | 57.463681 |

新编译0.836195秒（测试+构建内部4.461547秒），编译/共享拟合单列，不每臂重复算。DEV edge累计expanded62582、spatial1227663；历史20k spatial实际1031441，新实际用量增约19.0%，虽然允许上限是5倍。DEV spatial全部search1.310034秒，其中kernel1.018878、ctypes准备.269293秒；field11.574150秒。field在搜索2秒时限外，但全计入请求walltime；ctypes在搜索时限内。

历史20k DEV中位edge/spatial为.626415/.926279秒，新100k为.707706/1.220680；edge路径与搜索计数没变而时间也变化，不能将全部时间差因果归于节点额度。主要同期配对比较是本100k两臂；新spatial中位约1.725倍edge。请求时钟含读取、反投影、定位、网格、field、ctypes/search、转换、预测封存、验收及context；后续额外索引/一致性bookkeeping进入阶段总耗时。没有equal-walltime或新学习方法优势宣称。

## 归档与可复核入口

[完整family](observed_two_row_native_astar_100k_v1/) 包含实际source副本、compiler/flags/build receipts、每阶段日志/退出码、每请求result/seal、全部逐槽分析、12父图和hash索引；NPZ/so保存在ignored `runs/observed_two_row_native_astar_100k_v1`，不入普通Git。原服务器789文件均SHA复核，251二进制保留ignored。

- tests report SHA `fe6fdbd3278d18962fa23d925eb81dd47aa5c0f4112c3d2ad67d1e548a296325`
- TRAIN report SHA `9fe18ec35dc65ef7bc344013269b6716bfbf7f5384a5a2d975a991b8c6b91ebc`
- DEV report SHA `dc16b6e1bfcb316836bf461cdcf8a6d3b39ce9acbe4dc87037e26b90c75d4c94`

离线入口 `scripts/analyze_native_astar_100k_saved.py` 不导入模型、planner或collection标签，只读原验收记录与sealed池；新10+旧8测试18passed。实测命令保存在LOCAL_ANALYSIS_RECEIPT，两个固定stage的所有edge候选差分均通过。[全12父图索引](observed_two_row_native_astar_100k_v1/FIGURES.md) 展示三目标、两臂、全部K4，验收几何仅用于事后绘图。没有再生成路径。

这轮应保留为更强传统观察规划对照。它仍有11个条件attachment失败、1个条件目标错误和已知重复，不能作为新核心贡献或完整机器人有效性证据；本轮未调整任何field/几何标准，也不自动增加下一预算档。
