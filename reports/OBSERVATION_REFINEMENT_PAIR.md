# 一次草稿更新：局部/全局池化对照

两臂已从不可变 `5501c5f464b61ae1a34e34c5381bbded306bf4ed` 实际完成3000×32、seed0训练并exit0。运行目录 `runs/observed_refinement_reserved96_v1/{global_seed0,local_seed0}`，完整checkpoint、最终/草稿预测已同步本地，JSON/log/协议同步reports同family。随后固定分析源码 `d30506dfaf5673f6cb8af085340b8417c14271c7` 实际完成11项测试（新分析3、支撑3、selection5含eval_every恢复拒绝）、完整pair/草稿/原peak/planner复算和两臂支撑审计，均exit0；没有追加GPU训练。

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

独立复算通过全部实际checkpoint/NPZ SHA、完整refiner参数初始化重建、同输入配置、actual sampler最终状态和两臂768000完整path-state曝光核验。原始选模与best/last不变，24条DEV（含无参考指令）均保留。低成本原peak额外核验anchor_mode与best/last原summary SHA，不能误接soft控制。草稿指标只用于事后解释，没有拿草稿替换原final选模结果。

| 草稿→最终路径 | TipValid@4 | UniqueClassified@4 | 新增有效候选 | 被破坏的有效候选 |
|---|---:|---:|---:|---:|
| global best | 48.96%→50.00% | 1.167→1.083 | 2 | 1 |
| global last | 50.00%→52.08% | 1.042→1.000 | 3 | 1 |
| local best | 50.00%→51.04% | 1.125→1.167 | 1 | 0 |
| local last | 46.88%→43.75% | 0.958→0.875 | 0 | 3 |

每行均为同checkpoint实际保存的全部96草稿与96最终候选，未重新生成/筛选。local best唯一小增益对应1条新增有效候选；local last则没有修好候选并破坏3条。局部/全局草稿之间本身也存在差异，因为点编码器与原头继续联合训练；跨臂最终差异不能全部归因于更新这一单步。端点和事件在该步保持不变，语义准确率也不变。

实际前缀查询支撑审计只读取相同stride2 RGB-D点及保存草稿，不打开验收箱体/目标/参考。local best的1087个eligible查询有77个在2sigma=0.20m内无可见点，last为1072个中的100个；nearest中位11.39/10.78cm。实际修正欧氏范数中位2.66/2.64cm，逐坐标最大值6.86/7.61cm，均未触及10cm逐坐标上限。global best/last逐坐标最大4.71/4.40cm。Gaussian并未硬截断，缺邻近点也不等于自由空间；这些数据不支持直接把失效归因于半径过窄或上限饱和，更不能根据本DEV再调尺度追逐结果。

当前判断：这个最小局部读取方案没有表现出稳定、实质且抵偿计算成本的优势，不作为核心方法，也不因一项best小增益立即堆更多更新层。保留原单次peak和强传统观测规划器作为后续参照。若继续这一方向，必须先给出新的可证伪原因并有TRAIN证据支持，不能把增大半径/位移上限当作已经被本诊断证明的修复。

实际证据：`reports/observed_refinement_analysis_v1/analysis.json` 与 `actual_predictions_index.json` 保存完整来源、同初始化/流审计、逐父/逐指令差异、draft/final失败及低成本参照；`reports/observed_refinement_support_v1/{global,local}.json` 保留全部2112内点query/阶段的支持与实际更新值；`reports/observed_refinement_analysis_jobs` 保存11测及三个实际分析job的命令、PID、退出码。所有二进制仍在本地及服务器 `runs/observed_refinement_reserved96_v1`，未进入普通Git。
