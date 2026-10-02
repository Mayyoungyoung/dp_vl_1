# 六配置静态比较：夹爪主体与柱体相交

固定 release `2c47c2a4c354ecfeb2f73b8121476f5000e82a98` 的唯一 CPU1 实跑完成。41 项服务器测试通过（0.29 s）；六个冻结配置全部应用，六次之前和六次之后的 world/inventory/RGB-D/current/camera 检查均严格零差，关节读回六次精确相同，simulation time 始终为 0。IK、路径规划、simulation-start 调用均为 0，没有新路线或训练参考。采集器耗时 23.775263 s，record_job exit0；完整进程时间、成本和矩阵见[机器可读分析](static_contacts_mesh_api_fix2/analysis.json)。

| 保存 query | 路口与姿态 | 实际外部命中 | 相交的机器人 shape |
|---|---|---|---|
| 5 | 第一排 middle，原姿态 | post 0、1 | Panda_gripper |
| 7 | 第一排 middle，竖直姿态 | post 0、1 | Panda_gripper |
| 13 | 第二排 negative_y，原姿态 | post 2 | Panda_gripper |
| 15 | 第二排 negative_y，竖直姿态 | post 2 | Panda_gripper |
| 21 | 第二排 positive_y，原姿态 | post 3 | Panda_gripper |
| 23 | 第二排 positive_y，竖直姿态 | post 3 | Panda_gripper |

六个配置均有注册柱体命中，没有桌面或任务目标等其它外部对象命中。显式 collection→单个机器人 body 查询同时显示相邻 link 1/2、3/4、5/6 和 link7/夹爪的重叠。这些单 body 查询与原 `check_arm_collision(all)` 的过滤语义不同，不能把相邻连接处重叠当作原失败的 self-collision 根因。完整 body 矩阵和补充 link 对逐项保留，不只展示柱体行。

新静态 FK 与原查询 tip 仍有 0.595–0.866 mm 位置差、0.0331–0.0430° 角度差。新旧共同状态差异也保留；本轮的严格恢复针对**新的共同 native snapshot**。所有产物均明确 `original_instant_reproduced=false`。这是一组对应静态场景中的潜在碰撞定位，不重写旧 24-query 标签，也不证明原动态瞬间已复现。

实际加载模型文件对固定 archive 的 SHA 检查通过。由于 pinned mesh API 的确定缺陷，本轮没有调用 `get_mesh_data`，runtime vertex/index/normal hash 均明确不可用；模型文件 SHA 只证明加载资产来源。[完整结果](static_contacts_mesh_api_fix2/metadata/static_comparisons.jsonl)、[来源及逐文件哈希](static_contacts_mesh_api_fix2/artifact_index.json)、[作业状态](static_contacts_mesh_api_fix2/run/static_contacts.status.json)均已保存。前两次初始化异常与原生崩溃保留在各自独立报告中。

## 唯一下一布局假设（未实施）

只将四柱共享高度从 **0.16 m 降至 0.14 m**，base 保持 0.755 m，top 从 0.915 m 降至 0.895 m。x/y 位置、横截面、路径高度 0.865 m、姿态策略、碰撞与分类阈值均不改变。这针对实际观察到的夹爪主体碰撞，而不是修改执行器或调整评测口径。

保存的六个静态配置中，夹爪 local bbox 经实际 pose 变换后的最低 z 为 0.9030–0.9119 m，比拟议新柱顶高 8.0–16.9 mm。末端 guide 在 z=0.865 m，比新柱顶低 30 mm，仍满足原 lateral 判据要求的 20 mm 顶部余量。该计算只为选择一个有依据的几何改动；bbox 和静态端点均不能证明整条动态路线安全。

最低成本验证建议：一个独立新场景/共同初态，一次 setup，预定中央目标 `[0.46,0,0.84]`，九个已声明过两排通道的序列各一个完整提案，每提案沿 v2 的九个 guide，全部失败/partial 保留、不 retry。上界为 setup 两次加路线 81 次 `get_path` 调用；内部 IK/OMPL 成本继续单列。采用新版本名，不与 v1/v2 的不同初态作因果配对；初态及每提案恢复仍需原严格 gate。只有真实仿真步机器人碰撞、tip/H24、终点检查均通过的路线才进入正参考。

主要可证伪点是降低柱顶后，原先夹爪卡住的通道是否仍在动态规划/执行中失效；若仍失败或实际通过关系未知，就保留负结果，不继续扫描高度。实际两排 crossing signature 仍依据轨迹，guide 编号不代替类型。原 pilot 的 lateral 条数门槛和结果不改变；若新数据设计将合法 over/lateral 混合关系用于预算实验，应在新版本执行前独立注册，并分别报告 lateral-only 与含 over 的已验证类型数，不将 unknown 计入，也不把九次提案当成全部解集。

## 六个静态观测

下面链接保留全部六个配置的原始渲染像素，不挑选好例：

- [query 5](static_contacts_mesh_api_fix2/metadata/query_05_static.png)
- [query 7](static_contacts_mesh_api_fix2/metadata/query_07_static.png)
- [query 13](static_contacts_mesh_api_fix2/metadata/query_13_static.png)
- [query 15](static_contacts_mesh_api_fix2/metadata/query_15_static.png)
- [query 21](static_contacts_mesh_api_fix2/metadata/query_21_static.png)
- [query 23](static_contacts_mesh_api_fix2/metadata/query_23_static.png)
