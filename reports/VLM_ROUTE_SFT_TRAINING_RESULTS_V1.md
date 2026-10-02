# 直接 VLM SFT 首轮真实训练结果

2026-10-02。**正式1500步完成exit0，原DEV token-NLL规则选中step1500；自回归路线质量尚未测。** 不能用教师强制NLL下降宣称有效候选更多、语义定位改善或机器人任务成功。

固定源码 `ba984062b6c672bb7e4c2572b2112994bbbd0c95`，按 [预声明训练协议](VLM_ROUTE_SFT_TRAINING_V1.md) 使用新96 TRAIN/8 DEV、真实固定revision Qwen3-VL-2B、last2 q/v LoRA、K1/K4交替。实测数据仍有3条TRAIN和1条DEV无参考指令；它们保留在清单中，没有被监督为无解。后续生成必须评价全部24 DEV。

| step | 固定DEV教师强制token NLL |
|---:|---:|
| 0 | 0.588299913 |
| 250 | 0.471902846 |
| 500 | 0.456782045 |
| 750 | 0.451384550 |
| 1000 | 0.444539060 |
| 1250 | 0.439973237 |
| 1500 | **0.436726732** |

训练结束到达预定预算，没有因best位于末尾自动续训。NLL在所有定期检查下降，说明该序列监督得到优化；它混合了坐标、事件、JSON标点和结束token，且K4权重按其实际token量进入均值。这个读数不能区分格式学习与真实规划质量，也未证明收敛。

实际TRAIN1500请求、3750目标路线槽、1,160,434 prompt token、1,273,081监督token、2,433,515全序列token。7次固定DEV检查共322请求、805路线槽、272,979监督token。DEV没有梯度更新，但其编码/NLL计算已纳入实际成本。与回归头比较不能只列相同步数或同K；已有基线曝光远大于这组3750目标路线槽，输入depth表示也不同。

内部总耗时556.786846秒，含数据/模型startup、训练、教师强制DEV；GPU占用墙钟0.154663小时，峰值allocated显存4,833,612,288 bytes。上述不是自回归响应时延。8个adapter张量全部有实际非零梯度、最终哈希全部相对初始化改变；被冻结LoRA base投影哈希未变。完整逐请求ledger、history、gradient audit、配置和record_job状态均保存。

测试与失败都保留：初始 `.venv-qwen` 命令因未安装pytest退出，**没有执行测试**；随后使用已有 `.venv` Python3.8/torch2.4.1，15项测试实际通过（2.83秒），包括正式shared training loop连续4步与2→resume的adapter/optimizer/scheduler/RNG/sampler/history/曝光逐元素相等。未安装或修改共享环境。随后真实Qwen正式训练使用原 `.venv-qwen` 并完成。

服务器目录 `/home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_v1/seed0`，三个小二进制也已同步本地ignored `runs/vlm_route_sft_v1/seed0`，未入普通Git。20份产物逐文件SHA与服务器相同，见 [artifact_index.json](vlm_route_sft_v1/artifact_index.json)。

- best1500：`675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f`
- last1500：`d604b5a213bdf281e7976b460b7ceb2fc428488610b1b84670ddca04711cb00c`
- initial adapters：`966970d0a68904a6828e036fbf8476e9c5956378843224a388aa4579d78bcf8b`

下一项实际研究决定是按 [固定自回归协议](VLM_ROUTE_SFT_EVALUATION_V1.md) 比较同一个best checkpoint的独立4次K1与一次wholeK4，严格保存格式错误、超额槽和全部编码成本。生成结果出来前，不追加训练或从NLL推断方法优势。
