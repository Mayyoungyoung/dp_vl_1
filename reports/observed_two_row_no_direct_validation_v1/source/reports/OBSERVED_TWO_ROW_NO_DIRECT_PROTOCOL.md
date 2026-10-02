# 普通结构对照：去除路线解码器的直接 Qwen 分支

## 唯一改动与限制

本轮是普通模型的结构/容量对照，不是路线集合新机制。保留冻结真实 Qwen、`geometry.task_query/state_query/point_encoder/fusion`、spatial peak anchor、状态编码、全部查询/decoder blocks/output。只删除 `head.feature_encoder` 的可训练层，以无参数零输出替代它对 context 的加项。直接继承原 `ObservedGeometryRouteHead.forward`，所有端点、事件、几何与路线公式保持；没有 probe 的 normal endpoint 钳制，没有冻结 geometry，也没有额外候选、修复、refiner或新标签。

geometry 的 fusion 仍接收 Qwen task query，所以此对照**不是语言无关路线头、不是完全语义—几何解耦**。删除549,120参数：1,231,965→682,845，减少约44.57%。任何收益都有容量/正则化解释，不能单独归因于显式分工或报告方法新颖性。近期 cosine-last 的 TRAIN swap 干预只证明该固定模型对直接通道的依赖，不能直接推出 constant 新训练对照的 DEV 因果效果。

## 数据、训练与信息边界

唯一数据为已封存 `data/observation_two_row_prefix76_v1`：64请求TRAIN父、63有观测父、189实际输入；缺失父/3输入不替换。DEV为原反复开发的12父/36输入，**48指48次选模，不是48个DEV输入**。不读取新 extension288 的32封存DEV，也不读取旧SCORE/CALIBRATION/TEST_LOCKED raw。

输入仍仅冻结Qwen的RGB+语言特征、当前夹爪状态及当前RGB-D/相机。所有原正参考进入 saturation assignment；unknown模式不作为负例或丢弃。端点辅助 `.02`、sigma `.025`，surface peak +逐坐标 `.05m` residual、K4/H24、width128/depth2/point64、event scale `.2`、stride2均保持。checkpoint选模仍原两排 `UniqueClassifiedTipValidAtK + .05*TipValidAtK`，每250步一次，全部36 DEV条件；2cm tip/3cm目标/原type判定不变。

固定seed0、constant AdamW lr `.0003`、weight decay `.0001`、LambdaLR=1、12000×32。384,000观察抽样，1,536,000训练路径状态，48次DEV选择；同预算原constant12000控制保留。无LR/种子/架构扫参。本轮只新训去分支一臂，不再训练已封存原控制。best与last12000均保存，原规则不重选。

## 初始化、真实采样与训练证据

新模型先完整执行原构造顺序，记录原完整初态SHA，再删除直接分支。共有tensor逐字节相同，构造后Torch RNG相同；完整参数集合不同，**初始forward不相同**。生产原完整初态SHA为 `7f81eba70dfcef2a3df19ebd5661ace3881abd552a1f2d4614cf490bf153bc13`。新实际initial model hash单独保存，不能伪写成原hash。

实际抽样沿用原 loop 的持久化hash链；终值须为 `40c280ca213bfdd89502108ac4e82917cfa4d68fea7f961ad7f2d1de5c84b567`。初始Torch RNG、初始sampler、终态所有RNG/sampler、constant scheduler均与原控制严格核对；新loop没有新增随机调用。`shared_initialization.json`记录删前/删后、逐参数初值hash与数量。

所有剩余参数 `requires_grad=True`，优化器只包含剩余参数。第一实际backward为每个剩余参数记录是否收到梯度、finite/nonzero和范数，随后移除审计hook；不修改梯度。最终每参数hash变化另存，非零梯度/实际更新不足会如实呈现，不宣称每个tensor必然非零。此一次审计同步成本计入实际训练墙钟，未算成方法免费组件。

## 恢复、预测预算与成本

原模型和training/evaluation loops文件不修改。新增driver仅在 `try/finally` 范围替换模型constructor、初态审计和预声明总步数；退出后恢复原入口。新config保存model/driver/policy/source/reference SHA。新协议checkpoint完整恢复model/optimizer/scheduler/scaler/RNG/global step/真实抽样链/best/history；旧direct checkpoint、改变source/steps/预算或非constant scheduler均拒绝。checkpoint间故障记录最多250未封存更新、32,000路径状态的保守上界，不自动重试或擦掉费用。

完整训练后保留原driver封存的best TRAIN、best DEV、last DEV池及所有hash。另唯一一次固定last TRAIN诊断覆盖189请求/756路径状态，CPU执行、不重选；staging中断不能自动重复前向。所有36同图异目标控制重索引现有池，不再生成候选。原base loop的24次缓存头时延诊断/96路径状态也保留并计账，**不是完整Qwen端到端时延**；不搬用历史1500步在线结果。无新Qwen编码。

新增输出 `runs/observed_two_row_prefix76_no_direct_v1`，原constant12000 artifacts仅校验哈希。GPU1/35%、单CPU亲和，root冻结源码并完成真实Torch tests后独立启动；本地Torch缺失的skip不能冒充测试通过。

```bash
taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/REV/scripts/launch_observed_two_row_no_direct_v1.sh REV fresh
# 只有确认原PID退出/锁与实际故障状态后，可显式同源恢复；不自动再次启动。
taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/REV/scripts/launch_observed_two_row_no_direct_v1.sh REV resume
```

报告逐12父/36条件的best和last差值、TRAIN拟合、Tip/Any/knownUnique/coverage/unknown/重复、语义与碰撞及真实cost。若端点变差或容量不足，完整保留，不以“路径依赖语言降低”替代质量证据；若结果有利，也仅支持普通baseline结构限制值得研究，仍未建立集合机制贡献。
