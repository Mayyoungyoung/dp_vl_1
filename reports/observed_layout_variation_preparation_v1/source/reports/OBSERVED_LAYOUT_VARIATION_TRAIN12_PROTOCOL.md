# 可变布局 TRAIN12：冻结前可执行技术协议

状态：**仅完成本地实现与纯测试，未运行服务器、模拟器或模型。** 这是 RLBench-derived 观察层采集技术小批，不是原 RLBench benchmark，不是方法贡献，也尚不能声称布局可采或解类数变化已经成立。上一版四柱 5c8 采集源和正在运行的数据不改。

## 固定范围与进入条件

实现只新增：`scripts/observed_layout_variation.py`、`scripts/collect_observed_layout_variation.py`、`configs/observed_layout_variation_train12_v1.json`、`tests/test_observed_layout_variation.py` 和本协议。读取原有严格恢复、完整世界审计、碰撞、轨迹执行、机械账本工具；旧源码不变，旧 geometry/签名/worker 不被猴补丁替换。既有 `phase_guard` 对规划 API 的受控临时拦截仍原样使用并在退出时恢复。

12 父 `layout_variation_401000`–`layout_variation_401011`，均 TRAIN，颜色 RNG 401999，一次生成，不重抽。每父 3 指令 × 9 登记槽；总 36 输入、324 槽。实际初图缺失/初始化失败也保留父和全部请求分母。历史排除表由旧公开 v4/v5/v6 的 9 条机械记录、old116 全计划、extension288 全计划组成，共 413 条；仅打开登记 JSON，不读这些旧角色的任何 raw。新内部重复组和历史 exact/1mm 命中都关闭全部相应新父，不借别的父补数。

`prepare` 在实际授权项目的 `data`、`runs` 下只扫描名称，检查父 ID/seed 占用；未通过就拒绝创建语料。当前没有运行该服务器检查，不能称 ID 已空闲。登记里 `server_id_availability_verified=false` 是本地声明；真正的初始扫描收据放在 corpus manifest 的 `presence`，不会回写登记。所有源代码、配置、机械历史登记 SHA 在 prepare 固定，worker/resume 再核。之后新 release 若这些字节不同，就不能继续这个 corpus。

## 几何与固定槽

所有柱宽 0.035m、底高 0.755m；entry `[0,0,.865]`。G0=`[(.46,-.16,.84),(.46,0,.84),(.46,.16,.84)]`，G1=`[(.48,-.19,.84),(.45,.025,.855),(.49,.18,.825)]`。

| index | N/排数 | row x | 各排柱 y | 各排高度 | goals |
|---|---:|---|---|---|---|
| 0 | 1/1 | .24 | [0] | .10 | G0 |
| 1 | 1/1 | .30 | [.04] | .18 | G1 |
| 2 | 2/1 | .27 | [-.09,.09] | .12 | G0 |
| 3 | 2/1 | .31 | [-.07,.11] | .18 | G1 |
| 4 | 4/2 | .17,.35 | [-.10,.10];[-.075,.075] | .14,.14 | G0 |
| 5 | 4/2 | .17,.35 | [-.10,.10];[-.0275,.0275] | .14,.14 | G0 |
| 6 | 4/2 | .20,.36 | [-.12,.08];[-.075,.075] | .12,.16 | G1 |
| 7 | 4/2 | .20,.36 | [-.12,.08];[-.0275,.0275] | .12,.16 | G1 |
| 8 | 6/2 | .18,.34 | [-.18,0,.18];[-.18,0,.18] | .10,.14 | G0 |
| 9 | 6/2 | .22,.36 | [-.20,-.02,.16];[-.17,.01,.19] | .14,.18 | G1 |
| 10 | 6/2 | .16,.32 | [-.16,.015,.19];[-.19,-.015,.16] | .18,.12 | G0 |
| 11 | 6/2 | .20,.35 | [-.20,0,.20];[-.16,0,.16] | .12,.16 | G1 |

4/5 同 family，6/7 同 family，各对颜色和目标相同，只有第二排开口变化。其余变化矩阵不是单因素因果实验。未来拆分必须整 family 同侧；本批没有 DEV/TEST。

一排 N 柱的意图序列为 `gap0,…,gapN,over` 循环取前 9 项，slot 0–2、3–5、6–8 的固定微变体分别为 −1、0、1。两排每排 2 柱的九对索引是 `(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,2),(over,over)`；每排 3 柱是 `(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3),(over,over)`。这是采集提案编号，不是路线类型标签。

每排在 x−.05/x/x+.05 建三个 guide；两排之间两个固定横向过渡点，再到目标；一排 4 次、两排至多 9 次显式 `get_path`。所有 guide 表先存登记，不依据验收换点。纯几何先分别检查布局物理一致性、每槽理想 tip 净空及签名；槽封闭不会使整个父被替换。当前纯检查得到全部 12 布局一致、5/7 各 6 槽预关闭，其余槽理想 tip 路径通过。这**没有**调用 IK，不是实际机器人可行性。全局显式 `get_path` 上界仍保守登记 2916；没有将理论预关闭的槽转给其他提案。

## 原严格初态与执行约束

使用现有 v4 冻结 arm/gripper q、anchor SHA、官方 Arm.py 与 task model archive SHA；新的静态 derived 初态，不复现旧动态瞬间。每父一个新进程/新场景，在本父内恢复同一内存 native snapshot。按 N 实际创建正确尺寸 cuboid，不以远移 dummy 柱伪装可变数量。

原 `phase_guard` 从环境构造起覆盖初始化、settling、restore 的 IK/path 入口；这些阶段禁止实际规划。没有 `task.reset`、`init_episode`、setup path、fallback、重采。原静态全身 arm/gripper 环境检查、每 settling 步检查、两遍 canonical restore、完整世界/对象 inventory、关节目标、RGB/metric depth、相机、当前状态严格恢复不变。FK 与 entry 差至多 1cm。实际全 N pose/bbox/goals 逐值容差 1e−6，再要求实际 1mm geometry SHA 等登记；错误不得训练。

每柱和目标至少 10 可见像素，原 RGB-D 可见柱面 1cm 几何一致性检查不放宽。每槽开始前严格恢复并核直接当前 pose/open 与 observation 一致；`pilot.execute` 保持 `ignore_collisions=False`，每 segment 最多 1000 step、每 step 检查完整 arm 与显式 gripper/environment 碰撞；没有连续碰撞或完整机器人成功概率声明。

接受仍要求实际终点误差≤.03m、raw/H24 全线段 tip 净空 .02m、raw/H24 类型一致。实际 gripper 状态与轨迹保留，后续训练仍需单独 event-segment H24/stride2/端点表示容量审计。unknown→unknown 可为正参考，unknown 不是失败/不存在标签；不按长度筛去长回环。采集成功不自动等于训练容量通过。

## 输入隔离与类型证书

模型输入只有 RGB、目标颜色语言、depth、相机参数、当前 gripper pose/open。`observations.jsonl` 只存输入索引/角色/图像/语言；`observation.npz` 只存 depth/current/camera。全 N 障碍、真目标、mask 只存 `verification_only.npz`，guide、意图、类型只存独立登记/监督。N/family/成功率不能成为生成条件。

签名由实际路径按顺序穿越各排平面的低侧开放区间或明确过顶构成；排序混乱、混合关系、边界高度/穿越含混均 unknown。它是有限穿越关系，不是全 3D 同伦分类。单柱最多 3 个本分类已知简单关系，并不证明真实总模式≤3；六柱也不自动证明≥5。只有同父同目标采到至少 5 个已检查不同关系正见证，才给“已见有效类型下界≥5”。所有参考集均不完备，真实总解数为 null。

4→5、6→7 的变化证明需要：两个实际初态都通过；开放父有真实 accepted 且第二排 `gap1` 的正路线；关闭父实际测得 bbox 的低高度带中，中间允许 y 区间宽≤0。登记预期 open=+.075m、closed=−.020m，只是几何预期。`certify` 必须等全部12机械闭合，校验 witness receipt/trajectory 字节 SHA，然后输出两对 `established` 或“未建立”，不重新规划或推断没有采到就不存在。缺初始化/见证文件仍保留 12 父/324 槽分母并输出未建立。证书只排除该低通道关系，不排除 over/unknown 或所有解。

## 预算、恢复与退出

生产入口一个 CPU affinity，OMP/MKL/OpenBLAS=1，GPU 隐藏。不能与正在运行的两个 TRAIN256 shard 等作业合计超过 CPU4。项目内部逻辑文件上限复用 8GiB、下一父预留 256MiB，纯 metadata size 计量；不删除任何旧数据。

首阶段 `pilot4` 只 indices0–3，共108槽。**不会自动串联** `pilot12`；root 冻结真实纯测试并读取首4技术结果，资源允许后才单独决定 indices4–11。若共同初态/恢复/渲染故障或首4无可行路线，先停止并分析，不修改登记重跑。45分钟软 cap 在父边界检查已封存 worker 进程墙钟和；单个不可中断规划可以越界，记录 overshoot，不启动下一父。prepare/coordinator 额外开销由外层 record_job 单列，不能把 worker/body/外层嵌套相加。

每个实际开始槽在 restore/planning 前写入并 fsync `started`；完成后同样写 `completed`。已预关闭槽不规划，明确写未尝试。首个 restore 失败保留失败 partial，并将余下所有槽未尝试封闭。任何已有父 data 或 status 都不再启动 worker：resume 只续新的父，对旧中断父生成上下界 closure；partial ledger 保守给下一槽不确定上界，不隐藏重放。未明成本中断闭合后不能自动继续下一父，须人工看原外层状态。

每父通过 `record_job --resume-strategy none` 独立运行，stdout/stderr 被该父 log 捕获；coordinator 只打印 index/ID/role/closure。收据包括源、配置、模型资产、严格恢复、实际 API 入口调用、slot ledger、原始/失败轨迹、初图与世界、实际 geometry、wall cost 和 artifact SHA。API 嵌套入口不是新增候选数，不相加伪称样本。初始配置错误或 shutdown/运行错误退出1，当前父闭合后停止stage；科学初始化 gate 失败保留0尝试/27未尝试，可由同一已授权首4流程完成其余登记父。

## 冻结后命令形状（尚未执行）

root 将固定实际 commit/export/launcher SHA，用既有独立仿真环境和 headless 环境变量执行；本文件不启动新 simulator/Xvfb，也不猜新的环境路径。

```text
python scripts/collect_observed_layout_variation.py prepare --output <fresh-corpus>
python scripts/collect_observed_layout_variation.py stage --corpus <corpus> --run-root <fresh-run-root> --sim-python <existing-sim-python> --stage pilot4
# root 单独决定后；不自动链接：
python scripts/collect_observed_layout_variation.py stage --corpus <corpus> --run-root <run-root> --sim-python <existing-sim-python> --stage pilot12
# 同 stage 可带 --resume；只续未请求的新父，绝不重复旧槽。
python scripts/collect_observed_layout_variation.py certify --corpus <corpus> --output <fresh-pair-certificate.json>
```

本地纯测试覆盖一次登记/413旧机械排除/全部分母、variable-N 与旧N4兼容 hash、实际geometry拒绝、开闭证书须有正见证、unknown 正例、闭槽/失败未尝试、严格恢复失败立即停槽、partial账本上下界、已有父不重放、分阶段门禁和源固定。模拟 API、实际 RGB-D、strict restore 与采集率必须由下一次冻结服务器测试/首4真实运行验证，不能由这些纯测试替代。

本地实际回归：新 21 项加 canonical/formal/extension 直接相关旧测试共 **114 passed，0 skipped，53.22s**；Python 编译检查通过。未运行服务器或模拟器。精确命令/源码摘要在 `.bootstrap/layout_variation_local_tests.json`，该收据不能替代冻结 release 的服务器真实测试。
