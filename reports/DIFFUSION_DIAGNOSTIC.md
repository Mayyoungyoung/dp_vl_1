# 扩散epsilon基线：固定去噪诊断

固定4f348c73aaf1411e9c02ba04806a5134326651e8、CPU1、CUDA隐藏，record_job实际exit0；诊断主体耗时3.711秒。使用first32 TRAIN与全部128 DEV，不训练、不修复、不筛除候选。

| checkpoint | step | DEV t99 epsilon MSE | raw x0 RMSE | clip比例 | DEV t0 x0 RMSE | t0打乱条件RMSE |
|---|---:|---:|---:|---:|---:|---:|
| independent_best | 3000 | 0.006135 | 158.935 | 0.9879 | 0.00972 | 0.02381 |
| set_diffusion_best | 2000 | 0.009489 | 197.666 | 0.9898 | 0.01139 | 0.02101 |
| set_diffusion_last | 3000 | 0.004836 | 141.116 | 0.9852 | 0.00865 | 0.02154 |

终端alpha_bar=2.42854071e-7，epsilon误差到x0的放大因子约2029；当前预测几乎全部被clip到±2。真实CPU采样的首步同样约98.5–99.1% clip，因此不是仅对带答案noised reference才出现的现象。TRAIN也出现相同模式。

条件并非被完全忽视：低噪声重建约0.009–0.011米，打乱条件后变为约0.021–0.024米；中噪声也变差。此证据支持先检查标准参数化的数值/优化问题，而不支持盲目添加条件模块。它没有证明终端误差是全部失败的唯一原因。

| checkpoint | 实际路径z MAE | 最后去噪RMS | 线段-膨胀障碍距离为0比例 |
|---|---:|---:|---:|
| independent_best | 0.021658 | 0.023550 | 0.687500 |
| set_diffusion_best | 0.040609 | 0.019004 | 0.792969 |
| set_diffusion_last | 0.015286 | 0.022950 | 0.740234 |

实际采样仍是K4/40步，没有额外路线。这里CPU的torch随机流与之前GPU训练时DEV评价的同整数seed不保证逐元素相同，故CPU诊断指标不覆盖或替换原GPU主结果。障碍距离是精确非负线段-AABB距离，碰撞为0，不解释成穿透深度；完整碰撞和z数据保存在npz。

独立best与last同3000步且所有模型权重逐元素相同，last只记录对best的引用，避免重复计算。集合best2000与last3000分别诊断。

下一次实际实施决定：标准v预测、直接x0/epsilon DDIM重建，保留原cosine、40步、K4、同父/目标/噪声和3000步，从头配对训练两臂。epsilon保持默认且历史实现不动。只改变标准参数化，不称方法创新，也不使用更有利的数据/验收规则。若后续有效性仍低，继续据证据诊断收敛，不能据这次弱epsilon结果宣称扩散已被充分战胜。

原文与官方公式核验见 [DIFFUSION_PARAMETERIZATION_REVIEW.md](DIFFUSION_PARAMETERIZATION_REVIEW.md)。PG仍未执行；当前主要瓶颈是有效性，不是有效路线之间的重复。
