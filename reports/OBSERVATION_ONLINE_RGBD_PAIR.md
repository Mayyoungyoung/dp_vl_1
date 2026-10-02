# 同起点在线RGB-D冻结/LoRA配对：均保留step0

真实Qwen在线继续训练没有改善本轮开发指标。冻结骨干和LoRA两臂均由原DEV参考ADE选中共同的step0；实际额外训练中8个LoRA参数张量全部得到非零梯度并更新，但这些更新没有带来有效收益。原共同模型继续作为最强方案保留。

|设置|选中步|选中语义|选中参考ADE|最后1000步语义|最后参考ADE|最后参考末端|
|---|---:|---:|---:|---:|---:|---:|
|在线冻结Qwen＋RGB-D|0|19.79%|5.74cm|16.67%|7.13cm|12.88cm|
|在线LoRA Qwen＋RGB-D|0|19.79%|5.74cm|12.50%|7.29cm|13.17cm|

语义为全部24条DEV/96候选，参考指标仅23条有参考输入；严格身份和3cm标准不变。两个最优检查点的全部预测坐标逐元素差为0，选中参考末端均10.497cm。最优LoRA checkpoint为零B初始adapter，不得把最后步的adapter更新事实算到最优权重上。

共同起点是新64 TRAIN父训练的完整RGB-D辅助模型（best SHA `281aa924fd78207ad55795214a232ce87aab7000a8b2da8ea81c805f06d9a5f8`）。已核验两个配置的初始化状态hash、数据fingerprint、相机点输入、候选数、学习率、processor、seed和曝光完全一致。每臂新增1000更新×4真实在线Qwen请求，4000观测请求/16000候选槽；共同预训练128000槽及0.01126 GPU小时单独记录。两臂均训练全部1231965个geometry/集合头参数，LoRA额外114688参数，只在末两语言层q/v投影中形成8个A/B张量。两臂使用相同0.02 endpoint attention监督，没有使用过时隐藏缓存或答案token。

冻结和LoRA实际循环耗时361.95秒/405.25秒，含setup GPU占用0.10386/0.11372小时；峰值allocated显存4124.39/4141.84MiB，均在授权范围。最优权重的实际单请求全Qwen＋RGB-D＋头中位约69.61/69.57ms，页缓存可能已热；该计时不含检查、评分或机器人执行。

两臂均来自固定release `9c19288b0e2fb86b6bea42ac23758314134872c0`，实际exit0。原始JSON、完整最后/最优adapter审计、逐场景输出、模型初始来源及日志在 `reports/observed_online_geometry_v1/`；配对配置逐字段核验在 `reports/observed_online_geometry_pair_analysis.json`。实际pt和npz已下载到本地 `runs/observed_online_geometry_v1/`，也保留服务器同名路径，二进制SHA见各run的 `binary_artifact_index.json`，不进入普通Git。

本轮不继续通过更换seed寻找LoRA正结果。独立attention诊断已经表明跨目标软均值会损害定位，因此后续先验证相同编码器下的普通峰值anchor修复；该诊断及修复不回写本轮历史预测，也不包装成集合方法创新。
