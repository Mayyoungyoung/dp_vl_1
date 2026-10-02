# 常规峰值anchor修复：配对验证记录

注意力在多个目标之间分散时，soft anchor会落在物体之间。新增模式只改变anchor：前向取最高attention的有效观测点，反向使用原soft期望的梯度。它不新增参数、候选、目标信息、几何修复或局部窗口；attention分布监督保持原样。默认soft保留历史行为。这是普通定位强基线修复，不是路线集合方法核心创新。

实现位于 `routeset/observed_geometry.py` 的 `anchor_mode='straight_through_peak'`，训练参数由 `train_observed_geometry.py --anchor-mode` 显式保存；模式不同的resume被拒绝。固定运行源码 `2dc026b8a34fecbb3688d26ff544165835ebd938`，旧在线9c作业及旧输出均未修改。

## 历史控制可以复用的实证

- 11项CPU测试通过，覆盖默认soft公式逐位相同、初始化相同、峰值严格是有效观测点、梯度正确传到attention、参数恢复和模式拒绝。
- 对实际产生natural64 soft控制的bb3077源码及obstacle32 soft控制的9c源码，旧/新soft/hard同seed全部1231965参数逐位一致，初始参数状态SHA `d39c101fa1e88e5c872b05b082254d121456360e149cbf887b312eefe323e59b`。两个设置各真实2条DEV的旧soft完整输出逐位相同；hard的attention分布也与soft逐位相同。
- 重放natural64的192个TRAIN指令、obstacle32的94个有参考TRAIN指令的全部1000×32抽样，最终采样器状态与实际旧last checkpoint逐位相同。采样索引SHA分别为 `fbe18e2e0d4fd7067707fd3bf7219ca075c17b3c6e58f48330719ea0f78bbb81` 与 `b95691c53daa2c5a52d2232f2e5c32184d3a63c574bf2ad635078eda8caadaf5`，数据fingerprint也一致。
- 实际new64原始数据CPU4步与2步＋resume2步的全部模型参数、优化器、采样器和调度器逐位相同。该短训练仅验证恢复，不能当方法结果。

具体命令、源SHA、PID、退出码与检查结果在 `reports/observed_peak_anchor_preflight/`。两组正式训练均从相同随机初始化开始，不从已训练soft权重warmstart。

## seed0实测与选模限制

|设置与方法|DEV-ADE选中步|选中语义|选中参考ADE|选中参考末端|
|---|---:|---:|---:|---:|
|natural64 soft|1000|19.79%|5.74cm|10.50cm|
|natural64 peak|1000|87.50%|3.86cm|6.91cm|
|obstacle32 soft|500|16.67%|18.76cm|13.72cm|
|obstacle32 peak|750|18.75%|19.56cm|10.29cm|

natural64语义为24输入/96候选，参考指标为23条有参考指令；obstacle32为12条输入/48候选。两方法同128000训练候选槽、1000步、batch32、seed0、lr3e-4、endpoint attention权重0.02。参数均1231965。

自然布局的定位与参考误差同时改善。障碍设置仍有语义与参考路径ADE的取舍，不能把选中检查点的低语义结果隐藏：last1000时soft语义43.75%、peak75.00%，但参考ADE分别21.30cm/22.33cm，比各自best差。后续将**双方所有seed**统一在1000步做额外 `fixed_step_1000` 探索性评价；这不替换原DEV-ADE选模主表或改变测试标准。完整路径障碍检查另由统一tip协议报告，语义命中不是全路径有效。

seed0完整结果在 `reports/observed_anchor_peak_v1/{natural64_seed0,obstacle32_seed0}/`，严格配对核验和best/last结果在 `reports/observed_anchor_peak_seed0_analysis.json`。真实pt/npz位于本地及服务器的 `runs/observed_anchor_peak_v1/`，SHA索引已保存，不进入普通Git。每组hard实际约54秒，均exit0。

同两设置的soft/peak seed1、2已由主任务排队执行，结束后将按三种子完整结果更新判断，不能从seed0外推稳定收益。
