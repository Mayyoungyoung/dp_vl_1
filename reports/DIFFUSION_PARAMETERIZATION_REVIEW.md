# 扩散参数化诊断与常规修复边界

这是强基线训练诊断，不是新的任务路径机制。3000步epsilon基线明显偏弱，不能仅据其失败宣称普通集合回归在充分训练的扩散上取胜。

## 已核验的原文与官方实现

[Progressive Distillation, ICLR2022，原文第4节](https://arxiv.org/html/2202.00512)直接分析低SNR时epsilon输出的小误差经除以alpha放大为x0误差，并比较x0、混合参数化和v预测。v的定义是 `v=sqrt(a)*epsilon-sqrt(1-a)*x0`，据此 `x0=sqrt(a)*x_t-sqrt(1-a)*v`。论文虽以蒸馏为主，也测试未蒸馏模型的参数化；本项目不会把采用v预测当创新。论文链接了[作者Google Research实现](https://github.com/google-research/google-research/tree/master/diffusion_distillation)，本次网页访问受限，未声称运行或完整核验作者仓库。

[Common Diffusion Noise Schedules and Sample Steps are Flawed, WACV2024](https://arxiv.org/html/2305.08891v4)第3.2和第6节强调：零终端SNR下epsilon目标退化，v预测应直接转换为x0进行采样，不能先转epsilon再用除以趋零alpha的公式，否则重新引入数值问题。我们现有100步cosine已从最后时间开始、终端alpha很小但非零；是否存在严重误差放大要看实际诊断，不能直接把论文的Stable Diffusion问题当成本项目已证实原因。

已逐公式读取[Hugging Face Diffusers官方v0.35.1 DDIM实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py)：epsilon、sample、v_prediction分支明确分开；v直接产生x0与epsilon，然后执行同一DDIM更新。若诊断支持改参数化，将新建适配实现与两臂同预算重训，不改历史DiffusionSchedule，不从旧epsilon权重硬切输出语义，不改变数据或几何验收。

## 已预先规定的诊断

固定first32 TRAIN及全部128 DEV，读取independent/set_diffusion的best和last；同一步且权重逐元素相同则明确复用诊断。t=0/10/25/50/75/90/95/99下给同一合法参考加同噪声，比较epsilon MSE、x0重建误差、clip比例；将整条条件向量循环打乱，量化场景条件的实际作用。TRAIN和DEV分别记录，不将DEV参考反传或用作推理修复。

同时对原始40步DDIM做被动hook，保存每步raw x0 clip、最后去噪幅度、最终z误差、完整线段到clearance膨胀障碍的最小欧氏距离。后者是精确非负距离，碰撞路线为0，不称为穿透深度。返回的候选与原始sampler相同，不加入新候选或隐藏重试。

脚本：`scripts/diagnose_multigate_diffusion.py`。已从固定4f348c73aaf1411e9c02ba04806a5134326651e8用CPU1实际执行，3.711秒、exit0；结果见 [DIFFUSION_DIAGNOSTIC.md](DIFFUSION_DIAGNOSTIC.md)。当前该诊断脚本仅解释epsilon权重，遇到v配置会显式拒绝，避免输出语义混淆。

## 证据后的决策规则

若终端时间x0误差/clip爆发，而低噪声重建和条件扰动显示模型确实学到了条件，则优先一次标准v预测参数化对照；仍保留原cosine/40步/K4/数据/目标槽，避免同时改变多个机制。若低噪声或条件使用也明显欠拟合，则先处理收敛/条件学习，不能把问题全归因于终端。任何修复都是强基线常规工程，必须真实重训两臂并报告成本。

PG当前不扫强度。有效率恢复后才考虑同独立权重、同噪声、同K4/40步下gamma=0与一个预先选定非零强度的最小筛选；只有有效通道覆盖提升才考虑进一步预算曲线，不以轨迹欧氏分散冒充路线类型增加。

## 已批准的最小常规修复

终端DEV重建RMSE实际达到141–198，98.5%以上坐标被clip；低噪声重建和条件shuffle显示已学习条件。因此下一次只改标准v目标和直接x0/epsilon DDIM计算，仍用同cosine100、采样40、K4、3000步、原父/目标/噪声随机流，独立/集合两臂从头训练。新增 `routeset/diffusion_parameterization.py`；历史DiffusionSchedule不改，默认epsilon采样严格委托原实现。

v修复不自动获得PG兼容性：现有PG适配将模型输出解释为epsilon，故显式拒绝nonzero PG+v；zero strength仍可直接使用v schedule。v上的score引导将来若确有必要需单独推导与测试。当前目标是恢复可信的多样化生成对照，不增加研究创新主张。

固定 `7f27f93514850973fba93260e8569f2a7fec27ea` 上CPU1必要测试已实际完成：9 passed in2.84s，record_job PID250791/child250792，10:55:22–10:55:25 UTC，exit0。包含t0/99 oracle重建、默认epsilon逐元素保真、v连续4步与2+恢复严格一致、跨参数化resume拒绝及PG保护。测试日志和实测部署/Git blob哈希在 `reports/multigate_diffusion_v2_validation`。

新训练launcher已固定为服务器绝对路径 `/home/wzy/dpvlm/route_set_v1/research_v2/incoming/launch_multigate_diffusion_v2_7f27f9.sh`，SHA `d048ae9618c87d1ce4090b7ce8b9905c049ef4cd2873536bf83bcce15b368fbe`。准备时通过语法检查但未由子任务启动GPU；根线程统一启动。其新输出为 `runs/multigate_diffusion_v2/seed0/{independent,set_diffusion}`，每臂768000目标槽，最终要求两臂实际stream相同且与旧epsilon实验相同。旧epsilon结果完整保留。

## 实际执行后的判断

根线程随后启动并完成v两臂，均exit0；独立70.072秒、集合95.011秒。K4三次独立采样均值：独立Valid24.02%/UniqueValid0.78385，集合24.61%/0.796875。v相较epsilon的UniqueValid差分别为−0.04427、+0.16667；四臂父/参考/扩散时间/噪声stream逐程序验证完全相同。详细配对区间、逐K结果和预算索引见 [DIFFUSION_PARAMETERIZATION_RESULTS.md](DIFFUSION_PARAMETERIZATION_RESULTS.md)。

此修复对集合臂有收益，但没有解决碰撞，因此终端数值误差不是全部失败的充分解释。v集合每4个候选仍平均2.98958个碰撞、0.1875个有效重复；当前不扫PG。继续训练只可作为有上限、单独披露额外曝光的收敛诊断；不可将当前结果包装成方法贡献或已充分证明扩散上限。
