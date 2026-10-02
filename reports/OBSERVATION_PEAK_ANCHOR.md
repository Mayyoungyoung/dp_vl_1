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

自然布局的定位与参考误差同时改善。障碍设置仍有语义与参考路径ADE的取舍，不能把选中检查点的低语义结果隐藏：last1000时soft语义43.75%、peak75.00%，但参考ADE分别21.30cm/22.33cm，比各自best差。下文已完成双方所有seed的 `fixed_step_1000` 探索性评价；这不替换原DEV-ADE选模主表或改变测试标准。完整路径障碍检查另由统一tip协议报告，语义命中不是全路径有效。

seed0完整结果在 `reports/observed_anchor_peak_v1/{natural64_seed0,obstacle32_seed0}/`，严格配对核验和best/last结果在 `reports/observed_anchor_peak_seed0_analysis.json`。真实pt/npz位于本地及服务器的 `runs/observed_anchor_peak_v1/`，SHA索引已保存，不进入普通Git。每组hard实际约54秒，均exit0。

## 三种子实际完成：原DEV-ADE选模

两设置的soft/peak seed1、2共8个新训练均已完成、exit0，加上经过历史一致性审计的seed0构成12个真实训练。每一配对使用相同初始化seed、数据、抽样、参数量、1000×32训练曝光和选模规则。以下为seed0/1/2的均值±**样本标准差（ddof=1）**；不是三个独立测试集，也不是置信区间。

|设置与方法|严格语义候选正确率|参考ADE（cm）|参考末端误差（cm）|选中步，seed0/1/2|
|---|---:|---:|---:|---|
|natural64 soft|17.36±13.19%|6.47±0.70|12.15±1.48|1000/1000/1000|
|natural64 peak|83.68±6.62%|4.55±1.22|7.94±2.62|1000/1000/1000|
|obstacle32 soft|19.44±4.81%|18.38±0.53|11.94±2.57|500/750/500|
|obstacle32 peak|28.47±34.38%|19.30±1.02|11.03±3.11|750/500/1000|

natural64逐seed语义为soft `[19.79,29.17,3.13]%`、peak `[87.50,76.04,87.50]%`；三个配对的语义、参考ADE、参考末端误差均改善。该证据支持修复soft均值落在多个目标之间的定位缺陷。obstacle32逐seed为soft `[16.67,25.00,16.67]%`、peak `[18.75,0.00,66.67]%`；peak的seed1发生明显退步，三个seed的原选中参考ADE均变差，不能称为障碍路线上的稳定整体提升。

完整配对条件、逐seed指标、训练成本与源文件SHA见 `reports/observed_anchor_peak_three_seed_analysis.json`；原始训练报告位于 `reports/observed_anchor_peak_replication_v1/{natural64,obstacle32}/{soft,peak}_seed{1,2}/`。数据和候选预算没有增加，新增训练没有把多个随机重复的候选合并。

## 双方统一1000步：已实测的探索性敏感性分析

固定评价源码 `94de7dbc53058a0761fdb042053473734d19e652` 实际在CPU1加载全部12个 `last.pt` 重新生成预测，作业 `observed_anchor_fixed_step1000_jobs/paired_twelve_cpu` 已exit0。结果分别保存为新输出，原best及历史summary没有覆盖。全部12臂的语义判定与训练时last评价一致，参考误差只有CPU/GPU浮点级差异。

|设置与方法|严格语义候选正确率|参考ADE（cm）|参考末端误差（cm）|
|---|---:|---:|---:|
|natural64 soft，固定1000|17.36±13.19%|6.47±0.70|12.15±1.48|
|natural64 peak，固定1000|83.68±6.62%|4.55±1.22|7.94±2.62|
|obstacle32 soft，固定1000|39.58±5.51%|19.99±1.18|7.52±0.45|
|obstacle32 peak，固定1000|72.22±4.81%|20.66±2.19|7.59±0.84|

障碍固定1000步的逐seed语义为soft `[43.75,41.67,33.33]%`、peak `[75.00,75.00,66.67]%`。语义提升与平均参考ADE/末端误差恶化并存，说明定位阈值、参考路径拟合与选模之间存在取舍。该协议在观察开发结果后补充，仅用于敏感性分析；不能拿它替换不利的原选模结果，也不能称为预注册最终确认。`evaluation_protocol='observation_eval_v2'` 保持3cm与目标身份标准，另设 `checkpoint_selection_protocol='fixed_step_1000'` 区分检查点协议。

实际12组预测、逐场景JSON、checkpoint/source SHA和CPU运行记录分别见 `runs/observed_anchor_fixed_step1000_v1/`、`reports/observed_anchor_fixed_step1000_v1/`、`reports/observed_anchor_fixed_step1000_jobs/`。统一tip碰撞评价另附，不由终点语义替代。

## 证据边界与下一次实际检验

这12个训练重复使用natural旧8个DEV父（24语义输入、23参考输入）和obstacle旧4个DEV父（12语义/参考输入），均已多次参与方法选择。三种子减少初始化偶然性，不能消除小开发集与反复选择的偏差；两个设置也都仍是颜色目标派生任务。当前结论是常规观测定位基线得到修复，尚未证明开放词汇、机器人全路径执行或集合方法的新颖性。

后续使用 `configs/observation_partition_reservation_v1.json` 显式导出192 TRAIN＋16 fresh DEV_MODEL，先检验现存64父模型泛化；不读取保留的DEV_SCORE、CALIBRATION、TEST_LOCKED内容来继续选方法。固定步数与原选模两套结果都会保留，最终确认协议需在读取锁定测试前另行冻结。
