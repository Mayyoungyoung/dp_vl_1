# 同 checkpoint 直接 VLM：独立4次与整集合K4

2026-10-02。状态：入口已实现，新增7项纯生成/预算/后处理测试通过（0.35秒）；**尚未实际自回归运行，不能报告质量或时延结果。** 使用 [训练协议](VLM_ROUTE_SFT_TRAINING_V1.md) 所得同一完成作业的原best或last检查点，先核验训练summary、checkpoint SHA、训练配置、数据来源索引和实际输入/LoRA helper源码。两方案只改变请求K与调用次数，训练权重完全相同。

## 固定生成与信息边界

全部24 DEV指令、8父，包括无成功参考指令。生成入口先核验注册父/角色和当前输入哈希。supervision manifest仅提供当前observation文件的位置与身份；routes、semantic_targets、route_types、verification_only字段不进入生成记录，也不打开其原始文件。模型输入函数只接受RGB、当前depth、语言、相机参数和当前夹爪pose/open；`prepare_prefix`根本不接收路径答案。LoRA装载后全部参数冻结，运行真实Qwen自回归，不用冻结hidden cache。

| 方案 | 每场景调用 | 解码设置 | 最大新增token总额 | 名义候选 |
|---|---:|---|---:|---:|
| independent4 | 4次各K1 | 每次temperature0.7/top_p0.9/top_k0，单序列 | 4×512 | 4 |
| whole4 | 1次K4 | 同上 | 2048 | 4 |

关闭隐含top_k截断；不加beam、多返回序列、重试、rerank或几何修复。base seed0，每个repeat/scene/method/request由固定SHA256组合导出并记录独立解码seed。`--repeats`默认1；扩大采样重复时，每个repeat分别保留完整K4结果并独立评价，最后仅平均指标，不合并候选池。采样重复不是训练种子。每repeat共120次实际请求；两方案合计新增token上限98304，实际token另计。

每次独立请求都重新读RGB/深度、processor处理、视觉/语言编码；KV缓存只在该次自回归请求内部使用。请求计时从读入前开始至解码和严格parse完成，包含CUDA同步；四次独立编码全部纳入同一场景的时间。模型加载与首次请求保留，没有把缓存吞吐当单请求时延。场景时间还包含逐请求journal写入。外部几何检查另计，报告generation+checker的实测分项和，明确不是联动部署的单次计时，也未包含不存在的学习评分器。

## 失败与候选预算

`requests.jsonl`逐次保留原始生成文本、prompt/output token、是否达到token上限、解码seed、异常、format错误与真实可数路线条数；`per_scene.json`保留请求次数、格式通过、候选预算和计时。异常不重试，所请求槽均按失败保留；中途失败导致不可知的部分token不伪填为0，设置`token_counts_incomplete`，保留错误。

缺失/格式错误/重复均占候选槽。若任一请求过量生成，实际charged slots增加，**整个场景不能被当作固定K4比较**：用于严格K4汇总的四槽均为NaN失败；正常请求的原始解析另存nominal数组和文本，仅供审计，不从较大池挑选结果。这是预声明的保守规则，不改变既有几何标准。报表同时展示实际charged、超预算场景数和名义96槽分母，不能把实际多于96槽写成真正K4成功。

## 独立后处理

`scripts/analyze_vlm_route_sft.py`只在完整generation summary存在且全部产物哈希匹配后运行。它先重解析所有raw文本并逐元素比对NPZ、核验实际有限调用次数/seed/预算，再打开DEV物理几何与语义标签。训练source index必须仍匹配config fingerprint；export manifest必须匹配生成时SHA；所选DEV verification/observation当前SHA必须匹配原export。沿用 `observed_obstacle_tip_eval_v1`：3cm正确目标、5mm起点、原2cm完整线段箱体避碰、正确reach事件。未知有效路线保留有效但不虚构类型；没有reference的指令仍进入语义/tip分母。输出逐候选、逐场景、逐父配对差以及每repeat和平均结果。SelectedTipValid为空；tip检查不包含整臂、桌面、IK或机器人执行。

7项测试覆盖有限4+1调用、repeat不合池、缺失/异常/超额槽不筛除、拒绝答案字段/已有输出、raw文本重parse拒绝改写数组、均值不相加Unique和不伪造Selected、深嵌套JSON失败预算、来源/verification改写拒绝。首次后处理单测遇到旧几何helper的同目录import失败，已仅在新入口补同目录搜索路径，原几何实现未改。peer审查复现深嵌套JSON的RecursionError，新增generation-only safe_parse将其保存为NaN失败；原训练serializer未修改。最终7项全部通过。真实Qwen生成还需固定release后的独立record_job，纯fixture不证明质量。

## 固定release后由root启动的命令模板

```bash
# GPU1、CPU1、35%内存：无resume；输出必须新建，失败记录不得覆盖。
python -m scripts.evaluate_vlm_route_sft \
 --checkpoint /home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_v1/seed0/best.pt \
 --output /home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_autoregressive_v1/best_seed0 \
 --seed 0 --repeats 1

# GPU隐藏、CPU1：全部生成完成后独立验收，无模型前向。
python -m scripts.analyze_vlm_route_sft \
 --generation /home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_autoregressive_v1/best_seed0 \
 --output /home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_analysis_v1/best_seed0
```

实际run_id、固定源码与launcher哈希、PID和退出码必须由root的record_job记录；本worker没有启动GPU。中断时保留既有请求日志/已写结果，不能偷偷复用不完整输出或把重跑成本丢弃。
