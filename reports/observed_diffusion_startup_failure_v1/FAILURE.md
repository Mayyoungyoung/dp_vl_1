# 观测扩散首次入口失败（保留）

source `0f1d5bfbd4ff788a6ea311339adfe77449db9634` 的独立臂在读取普通控制收据时退出1。真实`composite_training_receipt.json`的抽样链位于`budget.actual_index_chain_sha256`，driver误读取顶层键。此前35工程测试没有覆盖真实收据结构；通过测试不代表真实数据入口已成功。

失败发生在模型构造、配置保存和调用journal建立之前：实验forward/optimizer均0，未生成权重或候选。已有数据读取和GPU运行环境初始化仍计入这次外层job墙钟/GPU占用，不把它视为免费或质量结果。原服务器目录完整保留，不覆盖或伪造resume。

最小修复只读取并验证原budget字段，增加原已完成收据fixture及结构拒绝测试；科学配置、数据、48次选择、12000步曝光均不变。修复后必须新commit、实际Torch验证和独立输出family，两个正式臂均使用同一修复来源。原失败不进入MAIN质量行。
