# 上层多路线规划器接口

默认演示模型为 `R2_seed0`，对应真实训练后的生成器、独立训练的 q 评分器和独立温度校准。组合模型已通过保存、重载以及独立 CLI 进程验证：八条路径、事件、q 和选择索引均与保存结果逐元素完全一致。实际 CLI 用时2.938秒，包含子进程启动和路线模型加载、复用真实 Qwen 缓存；这不是完整在线 Qwen 延迟。证据位于本地 `runs/paired_modes_v1/reliability/R2_seed0/public_cli_seed0/receipt.json`。

## 现有文件

本地目录：`F:/dpvlm/runs/paired_modes_v1/reliability/R2_seed0/deployment_seed0/`。

- `planner.pt`：生成器、评分器、训练集归一化参数和温度。
- `example.npz`：八条路径、事件、q 和选出的四个索引。
- `example_observed_inputs.pt`：真实观测得到的示例张量，可用于接口重放。
- `manifest.json`：生成器、评分器、校准和组合模型 SHA，以及重载验证结果。

生成器完整恢复点位于 `runs/paired_modes_v1/R2_seed0/last.pt`；q 的完整恢复点位于 `runs/paired_modes_v1/reliability/R2_seed0/q_seed0/last.pt`。部署模型与训练恢复点用途不同，后者保留优化器、调度器、随机状态和采样状态。

## 输入与输出

观测清单每行只包含 `id,parent_id,split,image,instruction` 五个字段；当前 CLI 使用绝对图像路径。RGB 对应的观测 NPZ 需要米制 `depth`、3×3 `camera_intrinsics`、相机到世界的 `camera_extrinsics`（3×4或4×4）、当前 `gripper_pose` 和 `gripper_open`。坐标系、深度尺度和训练观测格式必须一致。

冻结 Qwen 特征来自实际运行 `scripts.observation_cache_qwen`，不是占位向量。快照固定为 `89644892e4d85e24eaac8bacfd4f463576704203`，缓存记录清单与图像哈希。更换图像、指令、清单或骨干参数时，不能继续冒用旧缓存。现有1,440个请求的缓存已实际生成完毕。

输出 JSON：

| 字段 | 含义 |
|---|---|
| `paths` | 8×24×3，世界坐标米制路径点 |
| `events` | 8×24，对应路径上的夹爪状态输出 |
| `q` | 8个独立有效性概率，不要求和为1 |
| `selected_indices` | 从这八条中选出的 K 个索引 |
| `selected_paths`, `selected_q` | 对应子集 |
| `internal_candidates`, `returned_candidates` | 分别为8和K |

K=1取最高 q；K>1先取最高 q，再优先从 q≥0.5 的候选中选平均路径距离≥4cm的路线，不足时按 q 补齐。阈值只是固定展示规则，不是有效性证书。指定K=4仍然生成八条路线，不能宣称只用了四条生成预算。接口不暴露内部未训练的 pi 分支。

## 在已有服务器环境运行

先确认当前研究队列已经结束，避免与它并发。下例使用一个新 job id 和新输出文件；重复演示应更换二者，旧输出不会被覆盖。脚本自动限定 GPU1、35%显存和四线程，不需要安装或修改环境。

```bash
cd /home/wzy/dpvlm/route_set_v1/research_v2/releases/c25f18ea192954c4eee1e45f9ce8de631f865c69
bash scripts/launch_paired_modes_gpu.sh --id user_demo_R2_001 -- \
  /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.predict_paired_planner \
  --bundle /home/wzy/dpvlm/route_set_v1/runs/paired_modes_v1/reliability/R2_seed0/deployment_seed0/planner.pt \
  --manifest /home/wzy/dpvlm/route_set_v1/data/paired_modes_v1_v2/export/observations.jsonl \
  --id paired_family_520128_open_target0 \
  --observation /home/wzy/dpvlm/route_set_v1/data/paired_modes_v1_v2/parents/DEV_MODEL/paired_family_520128_open/paired_family_520128_open/observation.npz \
  --qwen-cache /home/wzy/dpvlm/route_set_v1/data/paired_modes_v1_v2/export/qwen_cache \
  --output /home/wzy/dpvlm/route_set_v1/runs/paired_modes_v1/user_demo_R2_001.json \
  --k 4
```

该命令的模型输入没有目标真值、障碍框或参考轨迹。它复用真实冻结 Qwen 缓存，因此计时包含路线模型和评分器，不是重新加载和运行 Qwen 的完整在线延迟。底层执行器可读取任一路径及其事件；本项目不在该接口中执行机器人动作。

q 表示本轮定义的上层目标、起点、事件、柱体余量和最低工作空间高度联合有效性。逐路线 sigmoid 不代表各路线错误在统计上独立：同一次目标识别错误可能使八条路线一起失败，不能把这些 q 相乘当作集合成功概率。完整机器人、控制器误差和未建模障碍不属于当前标签事件。具体概率质量及高置信度失败案例见 [RESULTS_REPORT.md](RESULTS_REPORT.md)。
