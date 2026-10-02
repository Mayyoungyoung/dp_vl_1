# 六任务完整 TRAIN：普通头第三个学习曲线点

`6464b3fb329704a99d8f9076dd6a8c5ac6821363` 的真实 Qwen 缓存、1500 步普通自由终点头和最近参考检索已全部 exit 0；实际 launcher/config 固定于 `98f818eb97458dd5b1340cbd17528645cc3a9ad7`。108 父快照保留 96 请求 TRAIN 父中的 1 个零参考父；训练只有 95 个正参考父、381 条正参考语言输入。DEV 仍为完全相同的 12 父、48 指令。

所有三点都用相同 head、seed0、1500×32、K4、grounding=0、task→positive-parent→language 采样和宏参考 ADE 选模。增加父场景的同时增加已见任务颜色/variation，不能称为纯 IID 样本数效应。只跑了一个真实训练种子，DEV 已多轮使用。

| 训练请求父数（正参考父） | Head best step | DEV best ADE / 末端误差 cm | DEV last1500 ADE / 末端 cm | 同数据检索 ADE / 末端 cm |
|---|---:|---:|---:|---:|
| 12（12） | 500 | 17.121 / 24.498 | 17.216 / 24.493 | 18.372 / 26.442 |
| 48（48） | 1500 | 12.944 / 19.700 | 12.944 / 19.700 | 15.393 / 25.895 |
| 96（95） | 750 | 10.000 / 17.434 | 10.244 / 18.751 | 13.890 / 23.145 |

96 请求父的 head best TRAIN ADE/末端误差为 3.528/5.542 cm；TRAIN 报告保留全部 384 语言输入、96 父，其中只有 381 输入/95 父有参考。不能将零参考三条输入当成无效解，reference 指标仅在可评价部分计算。DEV reference 分母始终为 48，semantic 分母为 0。best 的事件状态准确率 99.074%、压缩开合序列 100%；这些指标不判断事件发生位置或任务执行成功。

| 任务 | 96 父 head ADE / 末端 cm | 96 父检索 ADE / 末端 cm |
|---|---:|---:|
| pick_and_lift | 9.969 / 7.570 | 25.724 / 27.881 |
| pick_up_cup | 8.648 / 9.425 | 9.424 / 14.046 |
| push_button | 7.630 / 12.583 | 11.975 / 22.135 |
| reach_target | 15.007 / 31.631 | 17.519 / 34.655 |
| slide_block_to_target | 12.701 / 32.970 | 13.712 / 32.486 |
| take_lid_off_saucepan | 6.047 / 10.427 | 4.983 / 7.666 |

相对 48 TRAIN 父，12 个 DEV 父中 8 个 ADE 改善、4 个变差，全部保留于 `paired_parent_60_to_108.json`。cup 两父、reach_282017、lid_312017 变差；其中 lid_312017 ADE 6.444→9.432 cm、末端 10.573→18.058 cm。并非所有任务都随数据增加单调改善。主要未解决问题仍是 reach/slide 超过 30 cm 的末端误差，不能用整体平均改善掩盖。

新增 432 请求真正 Qwen 编码耗时 30.131 秒（含加载 4.056 秒），峰值 4,279,430,144 bytes；head 完整训练/评价 78.262 秒，峰值 950.855 MiB，仍为 1,231,965 参数、48,000 指令曝光/192,000 路线状态。检索全流程 CPU 2.739 秒，缓存后的检索耗时不等于完整 Qwen 推理延迟。

真实性复核已完成：48 条 DEV 输入/监督逐字段不变、60 个原图/当前观测/参考 SHA 与旧数据相同、48 个共享真实 Qwen 特征 NPZ 字节一致；全部 432 新缓存生产者/训练读取者哈希一致。新 best/last checkpoint、预测 NPZ 与检索预测的实际同步文件 SHA 均符合原运行 summary。复核入口为 `scripts/analyze_multitask_learning_curve.py --expanded-prefix 108`，机器证据位于 `reports/observed_multitask_prefix108_v1/learning_curve_analysis.json`。二进制在本地 ignored `runs/` 与服务器实际 run 保留，JSON 索引入 Git。第三点初始化 CPU 重建已在固定6464下 record_job exit0：60/108两次初始化完整参数与post-init RNG一致，参数SHA仍为7f81eba70dfcef2a3df19ebd5661ace3881abd552a1f2d4614cf490bf153bc13，实际receipt在 `observation_multitask_init_audit108_v1/`。这是按历史源码与记录seed重建，历史并未保存初始checkpoint。

下一步保留普通头的全部结果，单独检验已由 TRAIN 事件支撑审计支持的常规注意力辅助，固定既有 weight .02、sigma .025。辅助不改变网络参数/forward、不移动实际末端标签，也不删除难例；首先验证初始化、采样流与恢复兼容，再作同预算配对。它是增强定位基线的候选修复，不是任务路径集合的核心创新。当前参考重建仍不认证语义、碰撞、独特路线类型或机器人执行。
