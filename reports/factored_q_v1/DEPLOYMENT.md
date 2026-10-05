# 双评分多路径原型的使用

研究演示固定使用生成器seed0、评分器seed0，不按开发集最优种子选演示。完整比较见RESULTS_REPORT.md。该原型提供用户要求的两项分数和乘积；是否优于单q基线必须单独看实验结论。

## 模型与输出

本地包：`F:/dpvlm/runs/factored_q_v1/conditional_endpoint_g0_s0/deployment_v2/planner.pt`。服务器同名相对路径位于`/home/wzy/dpvlm/route_set_v1`。原始全路径任务头的九个模型保存在`conditional_g*_s*/deployment_v2`，修正后的九个模型保存在`conditional_endpoint_g*_s*/deployment_v2`。

输入是真实Qwen缓存特征、观测RGB-D对应的世界点、RGB、有效深度掩码、相机投影信息和当前状态。包包含路线生成头、评分头、归一化与校准；完整Qwen编码器仍通过既有环境/缓存提供。它不是一个脱离特征编码环境的独立VLM权重。

每次输出8条24点三维路径、路径事件、逐路线`q_task`、`q_feas`、`q=q_task*q_feas`和所选路线索引。最高乘积的路线排在选择首位；默认演示返回4条候选，保留8条完整结果。q_task表示任务匹配概率估计，q_feas表示任务成立条件下的上层几何可行概率估计。所有路线分数不归一化为和1，所以多条合理路径可以同时高分。

API：

```python
import torch
from routeset.factored_q import load_factored_planner
planner = load_factored_planner(bundle_path, device="cuda").eval()
# observed_inputs: features/current/world_xyz/rgb/uv/depth/valid_mask
with torch.inference_mode():
    prediction = planner(**observed_inputs, return_k=4)
```

命令行入口是`scripts/predict_factored_planner.py`，参数包括`--bundle --manifest --id --observation --qwen-cache --output --k`。实际运行过的完整命令保存在`conditional_endpoint_g0_s0/public_cli_deployment_v2/command.json`，输出在同目录`prediction.json`，核验在`receipt.json`。重跑请使用新的输出路径。此CLI拒绝TEST_LOCKED。

服务器运行应从不可变发布目录`research_v2/releases/8f2739c8dbfd86a330d8f303fdeedda4ca27a822`启动，使用既有`.venv/bin/python`、GPU1、四线程和现有显存上限设置。完整实验的启动器及每项实际命令见`runs/factored_q_v1/jobs/*/receipt.json`；不要重新运行已完成的队列覆盖产物。

## 恢复与范围

每个训练目录保留best、last、rolling recovery以及config。训练状态包含model、optimizer、scheduler、RNG、实际sampler/data-order摘要、配置与归一化。恢复入口为`python -m scripts.run_factored_q fit --arm <arm> --generator-seed <g> --scorer-seed <s> --resume`；只用于未完成的相应运行，完成的last存在时程序拒绝覆盖。

本轮没有训练底层执行器，没有把末端路径通过检查器等同于整机成功，也没有输出未经训练的模式频率π。评分器使用监督标签，不是RL价值函数。当前任务事件摘要只适用于本实验的reach语义，不能直接声称支持复杂接触操作。
