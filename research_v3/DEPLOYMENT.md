# V3观测输入推理

可复现演示固定使用完整集合匹配seed0 last1200，不按DEV挑选最佳种子。
推理只读真实Qwen缓存、RGB-D、相机、当前状态和指令；不读真值目标、
障碍框、参考路径。完整评分器独立固定，不能把新生成器替换到它内部。

在授权服务器使用已冻结源码110abd9c048eb758fbd42ff09fc3e52505e3fcc8：

```bash
cd /home/wzy/dpvlm/route_set_v1/research_v2/releases/110abd9c048eb758fbd42ff09fc3e52505e3fcc8
bash scripts/launch_research_v3.sh --id user_v3_demo_NEW_ID -- \
 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.predict_research_v3 \
 --bundle /home/wzy/dpvlm/route_set_v1/runs/paired_modes_v1/reliability/R1_seed0/deployment_seed0/planner.pt \
 --checkpoint /home/wzy/dpvlm/route_set_v1/runs/research_v3_v1/frequency_set_matching/last.pt \
 --manifest /home/wzy/dpvlm/route_set_v1/data/paired_modes_v1_v2/export/observations.jsonl \
 --id paired_family_520128_open_target0 \
 --observation /home/wzy/dpvlm/route_set_v1/data/paired_modes_v1_v2/parents/DEV_MODEL/paired_family_520128_open/paired_family_520128_open/observation.npz \
 --qwen-cache /home/wzy/dpvlm/route_set_v1/data/paired_modes_v1_v2/export/qwen_cache \
 --output /home/wzy/dpvlm/route_set_v1/runs/research_v3_v1/user_v3_demo_NEW_ID.json --k 4
```

必须使用新job id和新输出文件，且确认当前没有项目作业占用活动锁。
输出8×24×3米制世界坐标路径、8×24事件和八个独立q，以及四个所选索引。
K4仍生成内部M8；不将q乘积解释为集合成功率，也不将上层几何有效等同于机器人执行成功。

真实公开CLI验证位于服务器runs/research_v3_v1/public_cli_v1，以及本地
F:/dpvlm/runs/research_v3_checkpoint_20261009/runs/research_v3_v1/public_cli_v1。
生成器SHA0ffc30128082061cb7e1cff0e66068b2ab63a9014007cab567b1270e943fbf09；
固定完整评分包SHA5718fd973d1908307987cdac06c2886f4670fba7ed06cc1ababe9fe76323d4ca。
原始优化器/RNG/sampler在各训练输出的recovery.pt与last.pt中保留。
平均惩罚生成器在旧评分分布的三个重训q门槛失败；随后新配对布局评分数据实验
已全部完成，三个普通q门槛通过。上述旧演示仍保留为历史复现入口。

新增的强普通基线已实际通过独立CLI重放，预先指定seed0，没有挑最佳评分种子。
使用冻结源码目录
`/home/wzy/dpvlm/route_set_v1/research_v2/releases/9bbc0213a542815cd767716ddc5e5869f5d19d4c`，
保持上述观测参数，将bundle改为
`/home/wzy/dpvlm/route_set_v1/runs/research_v3_v1/matched_q_paired_v1/reliability/mean_seed0/calibration_seed0/scorer_bundle.pt`，
checkpoint改为
`/home/wzy/dpvlm/route_set_v1/runs/research_v3_v1/safety_mean/last.pt`。
仍使用全新的job id和output路径，并经过相同资源包装器。

该bundle SHA为2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e；
proposal SHA为ff2dfbc5463a38e9acb2af740e9b605cbfda5d1317cc146f5f2c4e65c2a7a60c。
本地闭合快照保存完整checkpoint、训练RNG/优化器和CLI_RECEIPT.json。
实际路径/事件/选集精确一致，q最大差1.1921e-7；2.947秒计时不含在线Qwen。
新q仍只是上层几何事件的概率估计，未验证新的独立TEST或机器人执行。
