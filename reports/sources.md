# 一手资料与基础实验边界

核对日期：2026-09-30。研究目标应明确为：给定任务与场景，在相同训练数据、相同候选数 K 下，生成更多**满足任务约束、几何有效、彼此具有实际差异**的任务层路线，并校准明确定义事件的概率：基础版是通过几何检查器，后续有固定控制器的实测结果才评价执行事件。是否使用扩散模型是实现选择，不能单独作为贡献。

## 已有工作与可主张的贡献

- [HAMSTER（ICLR 2025）](https://arxiv.org/abs/2502.05485)：高层 VLM 输出粗略 2D 末端执行器路径，底层 3D 策略执行；其核心是分层和利用廉价跨域路径数据。它支持任务层与执行层分离这一定位。
- [3D HAMSTER 论文](https://arxiv.org/html/2606.31329v1)与[官方代码](https://github.com/DAVIAN-Robotics/3D_HAMSTER)：单张 RGB-D 与指令输入，深度编码和深度重建约束支持 metric waypoint 预测。官方公开仓库仅提供高层推理；9B bf16 checkpoint，未包含底层策略。论文主要评价单条轨迹的目标位置与执行成功，没有设定“固定 K 的有效路线集合覆盖率”目标。
- [3DWay 论文](https://arxiv.org/html/2609.08224v1)与[官方代码](https://github.com/ziqin-h/3DWay)：VLM 输出对应的多视角 2D 路径，再用标定相机三角化；不能把这种表示本身称为新的贡献。可研究它与直接 3D 表示在**路线集合**质量上的差异。
- [Particle Guidance（ICLR 2024）](https://arxiv.org/html/2310.13102)：已有通过联合采样势能促进有限样本集合多样性的通用框架。仅增加路径间排斥项不足以主张通用方法创新；应作为强基线。过强排斥可能损害单条样本质量。
- [Tree-Guided Diffusion Planner（NeurIPS 2025）](https://papers.nips.cc/paper_files/paper/2025/hash/61e1b49d4f38f2126c7898b0a2dd221a-Abstract-Conference.html)：已经把 Particle Guidance 用于规划并结合树搜索。因此“扩散多路径 + 排斥/搜索”也有近邻工作。更具体的研究空间是 VLM 语义条件下的任务层路线集合、跨视角几何一致性、任务可用性和可靠置信度。

建议贡献表述：在 VLM 条件的三维任务路径生成中，通过语义约束与几何约束共同控制路线集合，在固定数据和候选预算下改善有效路线覆盖，并校准逐路线通过指定任务/几何检查器的概率。若评价固定底层控制器的执行事件，应另行收集执行标签和校准。最终新颖性仍需要进一步检索和实证，现阶段不宜声明首次。

## 开源数据与实际接入结果

优先使用[官方 3DWay-Data](https://huggingface.co/datasets/liyy4586/3DWay-Data/tree/main)，而不是把所有公开数据都判为不可用。文件列表：`rlbench.json` 约 109 MB、`rlbench.tar` 约 18.3 GB；全量包含 DROID/RH20T，约 158 GB，许可证继承不同上游。Hugging Face 自动 viewer 当前因 `image`/`images` 列差异报错，不影响直接读取文件。

已运行 `scripts/inspect_3dway_data.py`，HTTP Range 仅读取 RLBench 元数据前 **32,768 字节**，成功适配 **12 条**记录，结果保存为 `reports/3dway_metadata_sample.json`。实际字段为 `image`、`id`、`conversations`；标签是 `<ans_view1>` 与 `<ans_view2>` 中归一化到 `[0,1]` 的二维点及 Open/Close Gripper 事件。所检查的 12 条只涉及 2 个相同图像对，每个图像对只有 1 条不同路径，多行对应语言改写。此结论仅限检查前缀，**不能据此断言全数据集没有多路线**。

这些元数据样本没有相机标定字段，也没有碰撞标注、完整备选路线集合或固定控制器下的逐路线重复执行结果，因而适合单演示路径预训练/接口验证，不足以直接监督同一场景的路线覆盖、执行概率或重建 metric 3D。适配器保留 `camera_calibration: null`，不会伪造标定；也支持对已下载本地 tar 仅安全复制指定图像，不自动下载大归档。

[官方 3DWay 几何模块说明](https://github.com/ziqin-h/3DWay/blob/main/3dway_policy/README.md)还提供带标定的 `example_data`（两个图像、`camera_parameters.json` 与点云），可用于公开样本的三角重建烟测。相机变换是 camera-to-world，RLBench 示例可能有负焦距；两视角须逐点和夹爪状态对应。不能把这个示例的标定直接套给任意数据集样本。

已实际下载两个图像与标定到 `data/3dway_example`（合计 887,684 字节，逐文件 URL/哈希见 `source_manifest.json`），跳过 28.3 MB 的点云。`scripts/check_3dway_example.py` 已校验图像尺寸与标定一致，使用真实公开相机投影本次生成的 27 个三维测试点，再由双视角 DLT 重建：最大三维误差约 `1.47e-15 m`，最大重投影误差约 `4.07e-13 px`；加入每坐标标准差 1 px 噪声后的平均三维误差约 `5.27 mm`。结果在 `reports/3dway_geometry_smoke.json`。这些点明确是几何烟测生成的测试点，**不是模型预测或公开图像的真实轨迹标签**；没有验证机器人执行。

后续分层用法：用 3DWay/RLBench 做视觉语言和几何路径监督；对固定场景与同一目标重新采集多个可执行路线，形成分组验证集。只有一条人类演示时，别把未记录路线当作负例。

其他公开资源：

- [RLBench 官方环境](https://github.com/stepjam/RLBench)：适合在可重置场景中收集和验证任务演示。公开任务自带演示 waypoint，须另行生成/验证不同路线；官方示范数量不等于固定场景的多模态标注。
- [Motion Planning Diffusion 官方代码](https://github.com/joaoamcarvalho/mpd-splines-public)：已发布轨迹和预训练模型，适合作为无语言的几何规划基线；安装需要 IsaacGym 等依赖，当前并非最轻量基础版。
- [MotionBenchMaker 官方代码](https://github.com/KavrakiLab/motion_bench_maker)：公开预生成的操纵机器人规划问题，也可生成 scene/start/goal/path；需 ROS/MoveIt/Robowflex，语义任务和配套 VLM 输入需自己补充。
- [DROID](https://droid-dataset.github.io/)与[RH20T](https://rh20t.github.io/)：真实多视角演示适合泛化验证，但大文件下载成本高，公开演示描述并未保证同一初始场景的完整备选路线集合。

因此，小型受控三维数据并不是因为没有开源数据，而是为了补上**可穷举或明确列举有效路线模式的评价条件**。基础结果只说明几何覆盖机制能跑通，不能代替真实操纵任务和跨语义泛化实验。

## 冻结视觉语言基础编码器

[OpenAI 官方 CLIP ViT-B/32 模型](https://huggingface.co/openai/clip-vit-base-patch32)与[Transformers 4.44.2 官方 CLIP API](https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/clip)支持冻结图像/文本特征提取。[该版本官方 setup.py](https://github.com/huggingface/transformers/blob/v4.44.2/setup.py)明确要求 Python >= 3.8，适合先兼容已有服务器环境。CLIP 是对比式视觉语言编码器；它能提供预训练语义特征，但不能据此宣称已实现指令推理、开放世界任务规划或 VLM 泛化。应先验证语言改写、未见目标语义和视觉扰动，再换成具备任务推理能力的 VLM token 条件。

## 公平验证与主要陷阱

1. 固定训练场景、演示总数、预训练编码器、K、去噪步数和随机种子；同时报告 model forward 次数与实际时延。候选数相同不代表算力相同；超采样再筛选必须计入候选预算，或单独报告。
2. 至少比较确定性回归、独立扩散采样、Particle Guidance、多头/WTA/混合密度模型，以及所提集合生成方法。所有学习方法可见同样的路线标签；隐藏模式 ID 只用于评估时，不能只给新方法当推理条件。
3. 先判有效性，再算不同路线数：端点/任务阶段合法，整条折线段满足 clearance，长度不过度绕行；不能只检查离散 waypoint，也不能把增加抖动/绕圈算作有用多样性。
4. 主指标建议 `ValidDistinct@K`、有效模式覆盖率、`Success@K`、top-1 可用性；补充长度/平滑度、跨视角重投影误差。几何烟测的成功仅表示规定空间约束通过，不能称机器人执行成功。
5. 场景级拆分，语言改写必须跟同一场景/episode 进入同一 split；否则上面观察到的重复图像与标签会造成泄漏。
6. 普通三维空间绕孤立箱体的“左/右/上”往往是路线家族，不一定是不同同伦类。除非拓扑设定已证明，否则称路线模式/走廊类别更准确。
7. 置信度先校准为“通过指定任务/几何检查器”的概率，用独立验证集做 Brier/ECE/reliability；这个事件与“固定控制器下机器人完成任务”分别建模。后者需要实测执行标签。排序分数、VLM 自报概率、扩散 likelihood 都不能直接充当这两类事件概率。覆盖优先的集合选择分数与每条路线可靠性分数应分开。
