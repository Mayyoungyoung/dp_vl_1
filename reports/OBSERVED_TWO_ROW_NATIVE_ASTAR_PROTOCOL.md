# 两臂共同原生搜索内核：固定预算性能修复

本项只修复既有传统对照的Python搜索开销，**不改变研究机制、不声称核心创新**。edge与spatial都使用同一个新编译库；不得只给spatial加速后与旧Python时延作公平比较。当前只有本地合成图测试，不是正式观测结果；服务器编译、测试与真实请求须由根线程冻结后执行。

实际已有证据是继续这项修复的原因：原空间场对照TRAIN损失的17条均为`search_time_budget_exhausted`；DEV有56槽时间失败，约7232–8128扩展节点即到2秒，而field总耗时约7.855/160秒。这些是根线程/分析线程报告的已有实际记录，本卡未追加读取DEV或重新搜索。空间场本身仍为前项标准Gaussian惩罚，引用与非新颖性边界见[原协议](OBSERVED_TWO_ROW_SPATIAL_PENALTY_PROTOCOL.md)。

## 不变的搜索问题

旧三planner、spatial v1及evaluator原文件均不修改。RGB-D可见自由图、原20mm proxy阈值、unknown规则、精确端点预测/附件、原2.5cm格点与26邻居supercover、H24表示和输出后检查全部仍由旧代码负责。

新C++17只替代`astar_virtual`的搜索循环：

- Python从原`v1.NEIGHBORS`传入26邻居的原次序、每个touched offset的原次序和原浮点length；不重新排列或重新近似对角代价。
- start/goal附件保持原Python字典插入次序与实际长度；虚拟边代价沿用原Counter。
- priority queue按`(f, insertion_counter)`顺序；counter每次push增加，包括虚拟goal。旧队列第三项坐标从不承担平局决定，因为counter唯一。
- 严格保留closed节点处理、较小`g`才更新、欧氏启发式、浮点表达式的乘加次序、节点上限检查位置和64节点计时检查位置。
- edge臂读取原无向grid edge counts；spatial臂读取原`SpatialCosts.field`，取grid edge两端均值，系数仍4。field仍由原spatial v1按全部先前完整raw路径计算，不按后验有效性筛选。
- 原始统计字段、路径及失败状态保持；新增`native`子记录说明二进制身份与封装/内核耗时，不隐去原节点、邻边、supercover、permission与virtual-edge计数。

所有生产调用固定**每次最多20000节点、2秒、每64扩展检查时间**。Python adapter从函数入口开始计时，数组转换/ctypes准备耗时传入C++并参与该2秒判断；不能把封装成本藏到计时器外。原spatial场在此入口之前计算，仍按已登记规则独立计时并进入连续请求walltime。旧64节点批次检查可能越过2秒，不宣称硬实时截止。

无Python搜索fallback、无额外候选、无修路/补采样。C++错误、哈希或ABI错误即保留失败并停止。定位/图构建失败仍保留四槽；成功返回后才进行原检查，检查结果不回馈搜索。

## 接口、二进制和来源

新增实现：`routeset/native/observed_astar.cpp`与`scripts/observation_native_astar.py`。root runner使用：

```python
receipt = native.build_library(project / 'research_v2/native_builds' / commit, commit)
library = native.NativeLibrary(Path(receipt['library_path']).with_name('build_receipt.json'))
with native.native_planner(planner, library):
    # 原spatial.one_request及其edge/spatial分支原样复用。
    # 此上下文外是原Python astar，异常退出也恢复。
    result, paths, events = spatial.one_request(..., arm='edge')
```

`native_planner`生产接口没有可放宽预算的参数或CLI。`native_search_for_testing`独立测试入口才允许无deadline或更小节点上限；它只接受显式test-only编译回执，生产context拒绝test-only库。正式执行须使用真实冻结release的生产编译身份。

服务器只用已存在的`g++`，无安装、sudo、共享环境修改或cmake依赖。编译位置必须是项目私有`research_v2/native_builds/<source-commit>/`，且与当前冻结release commit一致；不可写入不可变release源码目录。采用`-std=c++17 -O3 -shared -fno-fast-math -ffp-contract=off -fPIC`。本地Windows合成测试将`-fPIC`替换为`-static-libgcc -static-libstdc++`，不得把其二进制或时延冒充Linux正式实测。

回执保存C++/adapter源SHA、编译器可执行文件SHA、完整版本字符串及其SHA、完整flags及其SHA、实际命令、stdout/stderr、exit code、二进制SHA、编译墙钟、平台与source commit。相同已封存build只能逐项核验后复用；未封存/失败/不同身份build不覆盖或自动重编译。动态加载前再次核验源码、编译器、flags、二进制与ABI版本。

## 固定数据、两阶段和真实计时

共同TRAIN拟合仍为32注册父、31实际父、93输入，prefix44 export SHA固定`6ed7786823275f26dba38fff9039bd33127b71567de4e4c0c80a37cfd91e46f9`。仅排除原prototype的`training_seconds`后，实际prototype+workspace canonical SHA仍必须为`384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b`；所有参数、source和TRAIN行仍比较。无新增正例、模式、目标坐标或几何信息。

1. 冻结来源并用实际服务器编译器通过differential tests；编译与验证成本独立记录。
2. 固定TRAIN前4父283200–283203全部12输入，native edge/native spatial各一次：24请求、96槽。检查同原配置、同真实role、同第一条路径、计时与失败账本。不能按结果调整sigma、节点或deadline。
3. root审阅后单独决定执行预登记DEV全部36输入（283264–283275）：**两个native臂均重新评价一次**，72请求、288槽；不复用旧Python时延。每个arm保存独立完整K4池，不合并候选。

两阶段新增上限**96请求、384槽**；训练/Qwen/GPU/模拟器调用均0。合成差分测试是独立工程验证，不算观察任务候选，也不能把测试调用隐藏为实验请求。请求与槽预算由root新runner记录；本模块自身不加载图像或验收标签。

复用原`spatial.one_request`：真实TRAIN/DEV角色，白名单输入，预测池保存/hash后才读取该请求标签，所有失败/unknown/重复保留、无repair。计时必须重新实测两个native臂的IO/hash、backprojection、定位、图、field、封装/搜索、原proxy、封存与检查连续walltime；编译、库加载和TRAIN拟合另列startup/全pipeline。不能将更快的底层实现单独归为集合方法收益。

## 差分验证与停止

本地实际已有`E:/MinGW/mingw64/bin/g++.exe`完成编译，24项新测试**24 passed /2.82秒 /exit0**。使用合成图，无实际TRAIN/DEV输入、无服务器编译或搜索。覆盖：空附件、无路、对角supercover、虚拟connector、多附件并列、Counter边、连续空间field、12个固定随机图、对称绕障tie次序、四轮同历史field、真实20000节点上限和64节点时间检查、生产预算拒绝放宽、作用域异常恢复、源码目录拒绝写入、编译/加载SHA门禁。

无时间截止的差分测试逐项比较**完整cell路径和全部确定性原统计字段**，只排除实际秒数及测试deadline标记；不是只比路径长度或最终是否成功。使用原Python函数和原邻居表，测试临时deadline/小节点设置会完整恢复，不修改文件。原成本公式的double运算与禁fast-math是必要条件，不等于对所有浮点输入的形式化等价证明；正式Linux编译器必须重新实际通过这些测试，不能凭Windows测试放行。

如果实际Linux差分或TRAIN机械门禁失败，保持产物和失败记录，不放宽数值容差、重新排tie或增加候选“修好”比较。若内核等价但覆盖仍无收益，仍遵循原卡的观测自由图诊断边界；不改空间半径、不将未知空间变自由，不复活已否定辅助模块。
