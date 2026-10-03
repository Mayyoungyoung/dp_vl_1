# 观测扩散：封存结果的只读配对分析

这是预定普通强基线的结果核验，不是新机制或额外实验。分析脚本为 `scripts/analyze_observed_two_row_diffusion.py`，测试为 `tests/test_observed_diffusion_analysis.py`。仅打开既有结果、权重的字节/hash、实际调用账本和作业状态；不加载模型/权重张量，不调用原始观测/参考loader、几何checker、搜索或forward，不读取reserved或新DEV。

固定实际训练源码为 `6e0203ba1335f9fa9975c523c657959f8bc9ab60`。独立与集合两臂使用相同composite108的285个TRAIN输入/1663条全部正参考、36个原DEV输入，以及原ordinary的384000输入抽样、1536000路径训练状态。unknown正参考不删除。独立/集合只是普通候选交互对照；相同路径训练曝光不表示相同全正例监督访问、计算量或时长，不能凭优劣证明确定性MSE平均导致碰撞。

运行前读取两臂下列共10份stage状态，全部completed之后才读质量池：`train`、`repeat1`、`repeat2`、`fixed-last-train`、`denoising-diagnostic`。仍在运行、paused、失败或缺产物则拒绝最终分析，保留原件；不补采样、不改选择标准、不跨stage自动启动。

核验内容：

- 完整policy/data/cache/source SHA；ordinary实际receipt及旧完整池，源权重只hash。新模型原ordinary随机初始模型/RNG证据、各共享geometry/feature/state张量hash与两臂整个初始state一致，绝不把训练权重当初始化。
- 两臂最终all-positive stream整对象相同；285/1663、384000draws、1536000states、实际ordinary index chain、参考抽样频数及共同reference/permutation/t/epsilon联合hash全部保留。单独和集合参考抽样目标相同；不能假称它与ordinary的全正例匹配损失相同。
- 12000次geometry、12000次denoiser和12000次optimizer，以及48×36次DEV geometry、48×36×40次denoiser。核实际append-only ledger的完整hash/调用顺序，并核每池的前后高水位；失败调用不会消失或被静默重放。
- 48个250步间隔原池、原评分 `UniqueClassifiedTipValidAtK + .05*TipValidAtK`、strict earliest-tie best和固定last12000，逐个seal/receipt/预测hash一致。这里只验证已选checkpoint，绝不从repeat或teacher重选。
- repeat0直接使用原best/last选模池；repeat1/2分别seed300001/300002，原repeat0为300000。每池36请求/K4，核保存的actual initial noise数组、每请求40个实际DDIM timestep调用及全部失败槽。best=last只允许原driver明记的零调用复用。
- 对每个repeat分别算TipValid/Any/已分类Unique/已知参考覆盖/有效unknown/已分类重复/语义/碰撞/事件/ADE。三个repeat只平均各自指标，并报总体标准差；不能合成K12或跨重复候选上限。仍是一个训练seed，不是三训练种子。
- 配对按相同输入/父身份，保留逐父和逐条件差值、胜平负以及语义正确×tip-clear四格；unknown类型不是失败或“没有有效解”。固定last285单独分旧189/新96/全部285，普通控制也取实际固定last训练池，不用best TRAIN替代。
- teacher限定最后权重、前6个TRAIN输入、t=0/25/50/75/99，总6geometry+30denoise/120中间状态、0optimizer/DEV。核相同参考索引和噪声hash、保存30个x0张量与原行，逐t报告x0 MSE/xyz RMSE/端点误差/事件MAE。这是正参考加噪的条件重建诊断，不是正常生成结果或视觉定位证据。

成本以该family实际`job_records/*.status.json`为主：每臂pause/resume/全部独立stage的真实outer时间含已终止失败作业；各body、检查、已选池和40步采样时间均嵌套展示，不重复相加。repeat0已在48次选模训练成本内；历史Qwen缓存和普通baseline成本是已发生共享/对照成本，不重新算作两臂新编码。另存在0f的模型构造前启动失败，保存在独立`reports/observed_diffusion_startup_failure_v1`，不藏掉；本脚本实验family的outer合计不包含该历史独立失败family，最终项目总成本由root另计一次。采样计时只包含已缓存Qwen条件下输入/geometry/40DDIM，不是完整在线Qwen端到端时延。

输出为`analysis.json`、无重复stream大数组的`histories.json`、全部选择池逐条件结果、收敛图与逐产物hash索引；输入文件hash末尾再次核验未变。每父三目标的全部数值均留存；图只表示原loss/48历史趋势，不以最后一次改善无限续训。原大NPZ与checkpoint不复制进普通Git。

冻结并通过纯CPU测试后，由root在所有stage真实完成时单独执行，例如：

```bash
python -m scripts.analyze_observed_two_row_diffusion \
  --runs /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_diffusion_v1_fixed_receipt \
  --ordinary-run /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_composite108_v1 \
  --training-source /home/wzy/dpvlm/route_set_v1/research_v2/releases/6e0203ba1335f9fa9975c523c657959f8bc9ab60 \
  --output /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_diffusion_analysis_v1
```

这是待执行命令，不能当作已完成结果。测试使用临时合成封存池验证未完成stage前置拒绝、重复/未知/NaN分母、原48选择和早平局、实际噪声错配、40次调用不足、receipt边界篡改及零Torch/模型导入；不会为测试打开真实reserved数据。
