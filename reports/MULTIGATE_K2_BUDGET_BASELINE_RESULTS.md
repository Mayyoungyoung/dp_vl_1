# 真正K2普通基线：未适配截断的缺口基本消失

2026-10-03。**普通saturation K2训练已经消除绝大多数所谓共同失效机会，不开发joint-risk新模块。** 固定原静态DEV选出的best500，TRAIN单关闭等父AnyValid为99.6528%，比K4未经适配的前两槽83.6635%高15.9893pp；距特权几何最远对/DPP/响应最优100%仅0.3472pp。全部16次可解关闭失败来自原集合已含无效候选的8父，没有两条原本都有效却共同失效的事件。

这是一个真实训练过的普通小预算基线、单训练种子和已复用开发数据结果；不建立新颖性、独立测试、观测语义或机器人执行结论。

## 训练、选择与预算保持

完全使用既有`routeset/train_v2.py`，source `5bb9087c9ca511c2a68a4da08798e95e8c6d4cdc`；注册launcher实际重新核验服务器历史K4 config。K2、H24、saturation、width192、depth3、3000×64、lr3e-4、seed0，静态DEV每500步按`UniqueValid+.05*Valid`严格提升选模。原模型/匹配/checker未改，没有新风险目标、几何修复或额外候选。

训练实际192000输入曝光、384000梯度路径槽、1175991正参考池访问；K4历史768000路径槽，不能称同候选预算。新模型1170690参数，比原K4少384个query参数；不是完整初始化逐元素配对。

| 静态DEV128 | best500 | last3000 |
|---|---:|---:|
| Valid@2 | 100% | 100% |
| AnyValid@2 | 100% | 100% |
| UniqueValid@2 | 1.9375 | 1.9375 |
| ReferenceCoverage@2 | 50.3364% | 50.3364% |

best在最早达到同分的500步保存，之后同分不替换；不是根据TRAIN关闭结果改选。last原权重和历史逐步评估均保留，本轮关闭审计只用了注册的best，没有重评/挑选更有利last。

## 同一TRAIN关闭协议

768TRAIN父，3806次关闭，3424可解/382无解；有条件分母720父。原7b496e2审计的全部参考选择、closure和模型/checker/helper源码hash核验一致，未重新选tie或改变标准。

| 原参考类型数 | 父数 | actual K2变化后AnyValid | 旧K4前两槽 | 几何最远/DPP/响应最优 |
|---|---:|---:|---:|---:|
| R<2 | 48 | null，全部关闭无解 | null | null |
| R=2 | 95 | 100% | 74.2105% | 100% |
| R>2 | 625 | 99.6000% | 85.1004% | 100% |
| 全部可评价父 | 720 | 99.6528% | 83.6635% | 100% |

actual K2相对joint4等父差−0.3299pp；相对旧prefix2的逐父正/负/平为526/7/187。旧joint4真实生成4条、prefix2也由4条产生；新K2真正只生成2条。参考控制读取R条特权正例池，不能看作同信息K2生成对照。

静态TRAIN Valid99.4792%、AnyValid100%、Unique1.927083：1536候选中8条原本无效。关闭后的全部16个可解失败分布在8父，逐项检查均为原集合含无效候选；原两条均有效同类型共失效0、原两条均有效不同类型共失效0。因此剩余缺口不能作为继续共同失效模块的证据。完整失败清单保留在[analysis.json](multigate_k2_saturation_budget_check_v1/analysis.json)。

## 实际资源与复现

- 13项服务器测试0.82s通过，固定wrapper和全部依赖原始SHA保存；不是仅本地测试。
- training record PID438120/child438128，2026-10-02 16:49:02.158139至16:50:56.703650UTC，exit0。进程114.546s，原trainer内部记录110.899s/0.030805GPU小时；前者含更多进程启动成本，二者分别保留。
- 实际GPU1 UUID `GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab`、35%显存上限；训练峰值PyTorch allocation49.583MiB。OMP4/Torch4、显式taskset与affinity只允许逻辑CPU0，不能将其wall time与历史4核结果直接比较。
- 原trainer静态DEV测得best单请求头前向中位1.491ms，只含受控小头；不含Qwen/观测处理/checker，不是机器人端到端延迟。
- 后续CPU关闭审计内部5.615s、GPU小时0（CUDA_VISIBLE_DEVICES=-1），实际生成1536条完整K2路线；所有原closure、无解分母和参考控制保持。
- 训练结束即通知GPU释放，之后CPU审计也退出。没有额外训练种子或参数搜索。

best SHA `bb151fdc3e083434d88711f9a9a479a3fc7e59b0522ebff801a1c1be924c8dad`；last SHA `60abd5a5c03e4bac81edefcdbadb968504b37b5467ac8b715fd3377a809b73c5`。静态DEV预测SHA `b948d37c6ff88b38cdf0f800e25c6baf52af1d513ace52f4dda6a3e3044061aa`；TRAIN预测SHA `7fec854abaea7c738a7760357eefdb66d1f26f3ee3a84085ac984547b1116a06`。四项均与服务器原文件逐项一致，二进制在ignored `runs`中，普通Git仅保存索引。

实际入口（已执行；原输出存在，不可直接重复）：

```bash
PYTHONPATH=/home/wzy/dpvlm/route_set_v1/research_v2/releases/5bb9087c9ca511c2a68a4da08798e95e8c6d4cdc \
  /home/wzy/dpvlm/route_set_v1/.venv/bin/python \
  scripts/launch_multigate_k2_saturation.py \
  --source 5bb9087c9ca511c2a68a4da08798e95e8c6d4cdc \
  --registration configs/multigate_k2_saturation_budget_check_v1.json
```

执行cwd为同source release。完整wrapper/命令/恢复路径见[validation索引](multigate_k2_saturation_validation_v1/artifact_index.json)、[launch receipt](multigate_k2_saturation_budget_check_v1/launch_receipt.json)、[20个实际文件索引](multigate_k2_saturation_budget_check_v1/artifact_index.json)、[TRAIN审计](multigate_k2_saturation_budget_check_v1/train_closure_audit/summary.json)。本次已完成，无需恢复或继续训练。

## 决定

停止当前双墙单门关闭方向的新机制和追加调参。真正K2训练证明零适配prefix不足以支持之前的假设；几何控制和普通生成器都已接近或达到上限。

后续只读方法判断依托新的真实两排多模式数据及其普通观测基线，保留R>K、R<K、未知类型与采集不完整边界；不为制造机会给双墙添加人工risk groups。本轮并未授权或实现下一核心模块。
