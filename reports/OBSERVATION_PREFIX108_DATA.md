# 六任务 prefix108：完整请求 TRAIN 与固定 DEV

在扩展原始数据/模型使用前，`configs/multitask_prefix108_registration_v1.json` 于 2026-10-02 14:25:04 UTC 固定每任务 TRAIN 父索引 0–15、原 DEV16/17，共 108 请求父。原采集已在运行；没有为注册重采样，也不替换失败父。

17 项新旧导出与事件测试已在固定 `6464b3fb329704a99d8f9076dd6a8c5ac6821363` 通过。正式导出随后在同 source 实际完成：PID 377758/377759，14:42:15–16 UTC，CPU1，exit 0。流程包含实时批内/跨批机械门禁、全部请求父封口校验、严格复位与当前观测/参考/源码哈希核验、失败分母保留和原子输出。未读取 SCORE/CALIBRATION/TEST_LOCKED 原始图像、路径或验收结果。

实际 108 请求父、324 次已记录尝试、321 条成功正参考。0 setup failure，**1 个 TRAIN 父零参考**：`pick_up_cup_322013` 三次均报告 `Could not get a path for waypoint 1`。它的初始观察和三条语言输入仍保留；采集规划失败不等于不存在有效解，不给负存在标签。全部原始尝试、失败及 PID/日志均随 snapshot/源数据保留。

| 分母 | TRAIN | DEV_MODEL | 总计 |
|---|---:|---:|---:|
| 请求且有当前观察的父场景 | 96 | 12 | 108 |
| 有正参考父场景 | 95 | 12 | 107 |
| 观察/语言输入 | 384 | 48 | 432 |
| 有正参考观察/语言输入 | 381 | 48 | 429 |
| 独立记录正参考 | 285 | 36 | 321 |

DEV 的 48 条输入和监督元数据与 prefix60 逐字段相同。父场景数量和已见任务 variation 同时增加，保持旧的非 IID 开发解释；不能把语言改写数或重复使用参考数称为独立场景/轨迹。

- 数据目录：`/home/wzy/dpvlm/route_set_v1/data/observation_multitask_prefix108_v1`
- snapshot manifest SHA：`b0f014ca2294ba863d1b75fc2e65d45f1c2e650687f09f320fd10f68230e998c`
- observations SHA：`faf8bfe503d5458883725a7bf991221f78b14a5f318426b8bc8e77a6355cdce1`
- supervision SHA：`71c0addab71f2f5ef0fa8818f9bde66aad8c3332cc72f94d55714df5f56a3dbe`
- 完整证据：`reports/observation_multitask_prefix108_v1/snapshot/`、`reports/observation_multitask_prefix108_export_v1/`。

下一训练配置/启动脚本为 `configs/observed_multitask_prefix108_v1.json` 与 `scripts/launch_observed_multitask_prefix108_v1.sh`。普通头保持相同参数、seed0、1500×32、K4、grounding=0、free_offset、task→有正参考 parent→language 抽样、宏参考 ADE 选模，无新模块。训练源码固定 6464，其八个模型/训练初始化依赖与旧 6bc/2f7 相同。新缓存必须真实编码全部 432 请求；训练仅使用 381 个正参考语言输入，但不从数据中删除另外三条。额外编码和采集成本单列。新事件辅助暂不并入此学习曲线。此文件只记录已完成导出与已准备配置，训练状态以随后实际 job 为准。

后续实际状态：同配置的432请求真Qwen缓存、1500步普通头及检索均已完成exit0；第三点完整结果见 `OBSERVATION_MULTITASK_PREFIX108_RESULTS.md`。此处的数据分母与原导出记录保持不变。
