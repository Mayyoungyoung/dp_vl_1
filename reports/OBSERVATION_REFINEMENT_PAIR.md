# 一次草稿更新：局部/全局池化对照

两臂已从不可变 `5501c5f464b61ae1a34e34c5381bbded306bf4ed` 实际完成3000×32、seed0训练并exit0。运行目录 `runs/observed_refinement_reserved96_v1/{global_seed0,local_seed0}`，完整checkpoint、最终/草稿预测已同步本地，JSON/log/协议同步reports同family。下表来自原始summary；独立草稿复算和可见点支撑审计另行固定源码执行，不以这些待执行步骤作为已完成证据。

local/global使用同96 TRAIN父、285有参考训练指令、同24 DEV（23有参考）、相同新增4995参数和零初始化、相同3000×32×K4最终候选曝光。总参数均1,236,960。读取尺度只由先前固定TRAIN审计决定：Gaussian sigma0.10m、草稿自身弧长前50%、逐坐标tanh bound0.10m。两臂在一次请求中均产生4条完整草稿和4条最终路径，共8个完整路径状态；训练每臂768000个完整路径状态。原单次peak和四次A*仅为更低完整生成数的参照。

| 原tip选模结果 | 选中步数 | TipValid@4 | AnyTipValid@4 | UniqueClassifiedTipValid@4 | 严格语义 | 参考ADE |
|---|---:|---:|---:|---:|---:|---:|
| global best | 2750 | 50.00% | 66.67% | 1.083 | 80.21% | 17.498cm |
| local best | 1000 | 51.04% | 62.50% | 1.167 | 62.50% | 18.687cm |
| global last3000 | 3000 | 52.08% | 70.83% | 1.000 | 80.21% | 18.050cm |
| local last3000 | 3000 | 43.75% | 58.33% | 0.875 | 80.21% | 17.922cm |

原选模分数为 `UniqueClassifiedTipValidAtK + 0.05*TipValidAtK`。local best相对global仅增加0.083个分类有效类型，Any降低4.17百分点、语义降低17.71百分点；last则TipValid降低8.33百分点、Unique降低0.125。仅凭这一种子不能称局部读取修复成立，不能只报告best微小增益而隐去最后一步的负结果。

global/local实际训练耗时145.069/201.797秒，local增加约39.1%；峰值allocated显存952.45/1250.10MiB。真实Qwen缓存共享、费用单列。尚未测此新增结构的完整在线Qwen单请求延迟，不能用旧普通头的在线计时替代它。

原单次peak best的TipValid52.08%、Unique1.125、Any62.50%；同数据传统观测A*v2为87.50%、1.250、87.50%。局部读取没有在当前证据下提供明确的质量—覆盖—成本优势；这些不同完整路径状态数与运行成本的参照不构成等预算胜利。Tip仍仅认证新增箱体的tip线段及原目标/起点/事件标准，不认证整臂、桌面、IK或执行。

下一步已准备独立只读入口 `scripts/analyze_observed_refinement_pair.py`：完整重建local/global参数初始化、实际采样流末态、配置与hash，复算保存draft与final的全部24DEV指标，保留原peak为低成本参照。另有 `scripts/audit_observed_refinement_support.py` 对实际草稿query报告可见点nearest和sigma/2sigma支持计数，不读取真实箱体/目标/参考，不把支撑计数解释成自由空间或安全。只有实际执行后才追加其结论；若不足以解释并修复收益缺口，就不继续增加模块。
