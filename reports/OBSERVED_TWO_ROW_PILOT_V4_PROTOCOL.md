# v4：较低柱体、中央目标九提案预登记

状态：固定 7b496e 的唯一实跑已完成，48 项服务器测试通过，九槽得到八条有效参考、五种实际 lateral 类型、三条有效 unknown，一条 H24 类型不一致被拒绝；[完整结果](OBSERVED_TWO_ROW_PILOT_V4_RESULTS.md)。固定文件 `configs/observed_two_row_pilot_v4.json`，协议 `observed_two_row_lower_posts_central_v4`。没有额外静态配置扫描、重试或自动扩采。本地合并 **48 passed / 0.73 s**；新增七项检查覆盖唯一物理修改、原三个实体球不变/只选中央目标、九条几何 raw/H24/原 guide 坐标、unknown/重复/over 与旧门槛分离、拒绝未登记的目标/预算/策略更改。纯几何通过不是实际机器人可行性证据。

## 证据与唯一几何修改

[六个保存关节配置的静态矩阵](OBSERVED_TWO_ROW_STATIC_CONTACTS_MESH_API_FIX2.md)将外部命中定位为 Panda_gripper 与柱体；两种姿态下第一排 middle 同时命中两柱、第二排外侧各命中相应柱。没有桌面或任务球命中。该比较在新的共同静态 snapshot 中完成，六次前后恢复严格零差，但不是旧动态瞬间复现。

v4 只将四柱共享高度由 0.16 m 改为 **0.14 m**，base z=0.755 m 不变、top z=0.895 m。柱 x/y、截面、三个球的位置、entry/preparation、九个 guide 的实际坐标、原姿态策略及所有有效性阈值不变。已有静态夹爪 bbox 最低 z 比新顶面高约 8–17 mm，支持这一物理假设；不会用它代替动态整机碰撞检查。末端 guide z=0.865 m 仍低于新柱顶 30 mm，原 lateral 分类的顶部 20 mm 余量不变。

新父 `two_row_reach_283001` 仅为 DEV_COLLECTION 可采性诊断；不声称独立随机化任务分布或与旧父动态初态配对。三个实体球仍放在原三个位置并通过全部可见性检查，只预注册 **中央球 target_index=1** 的语言/监督与九个路线槽，另两球继续作为可见干扰目标。新的随机种子不被当作布局独立性的证明。

## 明确有限预算

一次 setup，沿原两段准备动作建立真实低位 snapshot；上限两次 get_path 调用。其后九个预声明 passage 序列（negative_y/middle/positive_y 的两排笛卡尔积）各执行一次 v2 的九 guide 提案，最大 **81 次路线 get_path**，总上限 **83 次**。内部 IK/OMPL 配置搜索继续如实注明未逐调用插桩，不能将一次 get_path 等同一次内部搜索。每段最多 1000 模拟步；失败立即结束该槽，不补调用、不换 seed、不 retry。

setup 失败保留一请求父/一 setup 和九个未尝试槽。每槽保存规划/执行成本、逐步碰撞、partial/错误、完整状态恢复、终点/raw/H24 审计、hash 和类型。相机/世界/全 inventory/current/cal 必须继续严格零差。九个提案只是预算，不是已知全部解。

## 新数据可采性判据与旧结果边界

本版本预先声明：同初态、同中央目标获得 **至少五个真实不同且合法的实际两排 crossing signature** 才达到本轮预算机制数据目的；统计允许既有规则下的 lateral/over 组合。只按原 `crossing_signature` 的实际轨迹计算，规则完全不变：顺序与反复穿越须一致，边界带/回退/含糊为 unknown。有效路线仍须原逐步整机碰撞、2 cm tip 全线段、3 cm endpoint、H24 不改检查/类型。unknown 有效参考保留但不计已知类型数，重复只算一个类型。

报告同时列出 lateral-only、含 over 的全部已知类型、unknown、重复和完整九槽。新判据是本版本执行前登记的数据设计，**不回写 v1/v2 的 lateral>4 原门槛、负结果或旧标签**。五个实际类型也不表示五个同伦类别或已知总解数。若不足五类、失败仍多或恢复不成立，保留本次 negative，不据此继续扫描高度。即使达到五类，也只证明一个狭窄可采性实例，正式训练仍需独立布局变化与父级划分。

输入保持 RGB/语言/depth/camera/current gripper pose/open；实体真值、guide/type、mask、future路径、准备动作只用于监督/验证。这是任务层数据准备，执行器没有新方法贡献。
