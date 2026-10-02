# Multigate扩散强基线：实现前协议

状态：新增实现已完成，固定提交 `f78460795b2658c1767848533d9117a29ccc64c7` 上必要CPU测试实际通过（5 passed in 2.75s），尚无新扩散训练结果；GPU启动由根线程控制。现有普通集合回归是强基线，新扩散不是预设主方法。

## 复用边界

原样复用 `routeset.models.RouteDenoiser`、`routeset.diffusion.DiffusionSchedule`、直线残差编码及multigate完整线段评价。不得直接调用历史 `routeset.train`：它使用小写train/val划分、缺完整恢复，`common.balanced_targets`还没有过滤新数据的padding(mode=-1)，会把零填充路线抽作正例。历史实现保持不变；新模块/脚本显式处理padding、父split和新协议。

## 第一组配对实验

- 数据：现有 `data/multigate_v1_partitions/development.npz`，SHA `f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057`。仅TRAIN/DEV_MODEL，出现locked分区即拒绝；不重新造有利场景。
- 两臂：independent、set_diffusion，均真实训练；width192/depth3/H24/condition34/AdamW lr3e-4/weight_decay1e-4/clip_grad1。总参数形状相同，独立臂未用attention参数数目必须另报active count。
- Ktrain=4、batch64、3000步，每臂768000个完整路线目标槽，与已有3000步K4普通回归曝光一致。TRAIN父与变体均匀采样。两臂使用相同父场景、参考选择、扩散时间和噪声随机流，记录实际曝光哈希。
- 每个场景仅从mask为真且mode非负的正例中选目标，先无放回覆盖已知类型，少于K时再合法重复，最后随机置换槽位。模式ID仅用于训练参考平衡，绝不作模型条件。不把已知正例数量当真实解数。
- 100步cosine正向噪声，标准epsilon MSE；不对独立高斯噪声做Hungarian重配，也不混入新方法独占的几何损失或修复。集合臂学习有相关性的K个训练目标；独立臂看到完全同一目标批。
- 每500步固定DEV K4/40步DDIM/固定独立评价seed选优。最后对best在K1/2/4/8各进行3次随机重复，分别评价再平均；绝不把重复的候选合并。K≠4是输出数量转移，不能写作训练预算条件化。
- 保留best/last及模型、优化器、调度器、全部RNG、父sampler、目标sampler、global step、history、目标槽/参考曝光。新增输出+锁、配置严格校验、GPU1/35%/CPU1、run_id/PID/log/exit/恢复命令。

公平边界：同数据、信息和目标槽，不等于相同训练损失；回归可在整个正例池内自由匹配，扩散要拟合已采样条件分布，这是模型范式的真实差别，必须披露。各臂训练墙钟与GPU小时分别报告，不能仅给步数。3000步后若DEV仍明显上升，允许一次根线程依据证据安排的配对收敛检查，不能把尚未收敛的弱扩散当最终强对照。

## 唯一推断适配：Particle-Guidance-inspired RBF

[Particle Guidance原文](https://arxiv.org/html/2310.13102)及[官方代码](https://github.com/gcorso/particle-guidance)已核验；其核心是联合粒子势的梯度进入逆扩散。这里仅适配固定RBF势，不称原文机器人复现、DINO核复现或学习势复现。

令每条噪声残差展平为d维向量，a为cosine累计alpha，h_t²=a h_0²+(1-a)。定义

`E_t(X) = sum_{i<j} exp(-||x_i-x_j||²/(2 d h_t²))/(K-1)`，K=1时E=0。

解析排斥方向 `r_i=-grad_i E_t=sum_{j≠i} kernel_ij (x_i-x_j)/((K-1)d h_t²)`；epsilon改为 `eps'_i=eps_i-gamma sqrt(1-a) r_i`，再按同一DDIM公式更新、使用同一x0 clip。只用当前噪声候选，不读取模式、参考、目标类型或几何验收标签；固定端点由共同解码器处理。输出正好K，模型调用仍是40次，势计算开销完整计时。

gamma=0是原始采样；K1必须严格无作用。当前仅实现适配器与零强度/K1、解析梯度测试，**不启动强度网格**。先完成基础训练、检查收敛，再根据证据决定是否在DEV对gamma∈{0,0.1,0.3,1.0}、固定h0=0.2做有限网格；若执行，将使用同一独立checkpoint、相同初始噪声、相同K/步数。欧氏排斥可能只造成抖动或无效路径，只有真实通道类型UniqueValid/ReferenceCoverage增加且质量取舍可接受才保留。无额外神经模块、无数据修复、无大池筛选。

## 最小必要测试和证伪

1. padded或mode=-1正例永不抽中；R<K时合法重复，R≥K时目标类型无重复；独立与集合sampler同seed逐元素相同。
2. 新采样器gamma0及K1与历史DDIM同噪声逐元素相同；解析排斥与autograd对照、候选置换等变、K1/2/4/8有限输出、forward调用数等于指定步数。
3. 新trainer连续4步与2步中断恢复到4步全权重严格一致；不同data/lr拒绝；locked和父泄漏拒绝。只跑涉及变更的测试，CPU1时隙由根线程协调。
4. DEV报告Valid/AnyValid/UniqueValid/ReferenceCoverage及逐父结果；无训练评分器则不造SelectedValid概率。完整线段碰撞、候选数、每请求head+传输+checker实测p50/p95分别报告，明确受控几何而非VLM端到端。

停止规则：若合理配对收敛后扩散仍逊于普通集合回归，保留结果和最强回归；若PG网格收益只来自无效欧氏排斥或不胜同噪声gamma0，则删除该推断适配。这里补的是可信强对照，绝不通过重命名把它变成新颖贡献。

## 新增文件与待运行命令

- `routeset/multigate_diffusion.py`：仅TRAIN/DEV的严格加载、父/目标/噪声独立流、可恢复滚动内容哈希、固定RBF适配；历史models/diffusion/common不修改。
- `scripts/train_multigate_diffusion.py`：真实epsilon训练、best/last完整状态、严格resume、不同K三次独立评价、父均值、生成+传输+checker单请求计时。K1/2/8明确标未见K推断；没有硬把固定K4回归改成K8对照。
- `tests/test_multigate_diffusion.py`：5项有关实现的测试，包含真实最小训练与恢复，不是指标达成证据。

以下训练命令为待执行模板，必须在根线程固定的immutable release及record_job内执行，不能据此称已训练。第一条CPU测试已经真实执行：

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m pytest tests/test_multigate_diffusion.py -q
# 每臂分开record_job；GPU队列统一调度。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.train_multigate_diffusion \
  --data /home/wzy/dpvlm/route_set_v1/data/multigate_v1_partitions/development.npz \
  --output /home/wzy/dpvlm/route_set_v1/runs/multigate_diffusion_v1/seed0/independent \
  --arm independent --steps 3000 --batch-size 64 --candidates 4 --width 192 --depth 3 \
  --lr 0.0003 --seed 0 --eval-every 500 --diffusion-steps 100 --sampling-steps 40 \
  --eval-candidates 1 2 4 8 --final-repeats 3 --latency-requests 20 --threads 1 --device cuda
# 第二臂仅把arm/output末尾改为set_diffusion；相同dataset/hash与所有其余超参数。
# 中断恢复重用原recorded command加 --resume；核验实际PID后才处理陈旧锁。
```

真实预算由summary记录。本组每臂768000目标槽；训练参考池访问量和滚动stream哈希必须最终相同。模型的active参数数目、训练/评价总耗时及GPU小时另报，不用步数代替成本。最终3次随机重复分别生成K个、分别评价再平均；日志与每次predictions保留重复和无效输出。中途checkpoint选择只用单个固定DEV噪声seed，最终重复是同DEV诊断而不是新测试。

2026-10-02 CPU验证记录：`reports/multigate_diffusion_v1_validation/targeted_f784607.{log,status.json}`。record_job PID227087、child227088，10:16:29至10:16:32 UTC，exit0、CUDA隐藏、CPU1；结束后CPU槽已归还其它研究诊断。连续4步与2+恢复至4的模型参数/调度器/噪声/损失严格相同，两真实训练臂的实际stream hash相同。未运行无关全仓测试。

固定launcher：服务器 `research_v2/incoming/launch_multigate_diffusion_v1_f784607.sh`，SHA `ea4f7826cb997adb916ac46e9b250ccad340a6e662fe0f01c0d32e83bf473e09`；已上传并通过 `bash -n`，准备时尚未启动GPU。launcher核验源码、数据、GPU UUID、pytest结果和新输出，逐臂记录PID/log/exit/恢复命令，完成后再核验实际同stream/目标槽/参考池访问。运行后以真实日志与状态为准。

首启失败保留：根线程启动上述launcher后在preflight退出1，报 `Immutable source hash mismatch: scripts/train_multigate_diffusion.py`，尚未进入任何训练臂。实际审计发现Git blob为LF、服务器固定导出为CRLF，8个所用源码逐字节归一化后全部等于Git blob。`reports/multigate_diffusion_v1_validation/release_hash_audit.json` 同时保存实际部署raw hash和Git blob LF hash；没有修改release，也没有因为这项包装错误重跑已在实际release通过的CPU测试。

新的独立launcher是 `research_v2/incoming/launch_multigate_diffusion_v1_f784607_v2.sh`，SHA `21b69c4245af343330294065d2dbd1a45b222e33876c2d0d60aa0ccdc308cb18`，同时严格核验部署原始字节和归一化Git blob。旧launcher与失败事实保留，训练/评价配置完全不变。已通过shell语法和双端hash检查，GPU启动仍由根线程执行。
