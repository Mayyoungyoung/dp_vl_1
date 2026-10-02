# 直接 VLM 路线 SFT：首个正式训练协议

2026-10-02。**源码已实现；正式训练尚未启动，tiny恢复测试待固定release后执行。** 这是一项补齐强基线的实验，不主张LoRA、SFT或整集合序列本身有新颖性。此前6bc2b8真实预检的4次优化、8个adapter更新与4.84GB峰值只证明接口/梯度/资源可行；不能替代本次训练或自回归评价。

## 固定训练条件

数据只来自隔离的 `data/observation_obstacle_reserved_development_v1`。入口先只解析observation manifest，核验TRAIN父272000–272095及DEV_MODEL父272096–272103的完整身份与角色，随后才加载原始图像/状态/参考；不读取后续锁定父。所有轨迹/图像/当前状态文件、两个manifest均进入数据指纹。3条TRAIN和1条DEV无参考指令保留清单，不虚构负例；没有正参考的指令不进入监督或NLL分母。后续自回归评价仍须覆盖全部24 DEV指令。

模型固定为官方 Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`，同revision processor，transformers4.57.1/torch2.4.1。RGB与当前depth各一图，processor min/max_pixels均65536；depth使用现有高/低字节毫米编码。语言、当前pose/open、相机参数之外不加入目标、完整障碍、未来路径、采集模式或验收标签。`read_observation`按字段白名单读取NPZ，`prepare_prefix`不能接收reference字段；教师强制构造另用显式答案参数。完整序列的实际processor token前缀必须逐token等于无答案生成前缀，且全置-100。

首轮默认seed0、1500优化步、batch1、AdamW lr1e-4/weight_decay0、恒定学习率、clip1、last2语言层q/v投影LoRA rank8/alpha16，bf16冻结基座与FP32 adapter。保持model.eval关闭dropout，autograd正常启用；新增adapter实际训练。损失使用已通过真实预检的64-token chunk causal CE，同数学目标，不截短序列或路径。严格GPU1/35%显存/CPU1，根线程负责启动。

每步先均匀选择一个有正参考的TRAIN父，再均匀选择其中有正参考的语言指令。奇数步K1、偶数步K4：K1均匀采一条正参考；K4先包含该指令全部已采正参考，再均匀有放回补到4，并随机置换整集合顺序。未知类型正参考同等保留。当前最大参考数3；若未来超过4，本协议明确拒绝，不能暗中选择子集后仍称全参考。1500步产生750个K1及750个K4监督请求，合计**3750目标路线曝光**；实际token、模型前向和读入量独立记账，不能与回归头只按步数比较。

## 选模、恢复和成本

每个有参考DEV指令固定一个K1和一个K4答案实例，由独立固定 `dev_plan_seed=200000` 在训练前生成并落盘；不同训练seed也不改变此计划。当前23个有参考DEV指令共46请求。step0及每250步进行教师强制评价，全部监督token NLL之和除以总监督token数，严格更小才选best，另保留last。这是token加权语言建模指标，不是路径有效性、覆盖或执行成功；K4答案token较多会自然占更大权重，此选择规则预先固定。

检查点只含小adapter、AdamW、恒定scheduler、global step、Python/NumPy全局/目标Generator/Torch/CUDA RNG、独立parent sampler状态、best记录、history、gradient audit、曝光、耗时、源码/数据/固定DEV计划哈希，不保存完整基座。每25步及选中best时原子写last，最多initial_adapters/best/last三个小文件。`--resume`仅恢复同一输出中断状态，拒绝换数据、lr、步数、DEV频率、目标顺序seed或实际源码。`--stop-after`只用于保持原总步数的可审计恢复测试；它不会触发额外DEV选模。达到1500的完成作业不能再次resume。

每个输出目录有独立exclusive `train.lock`，记录PID与commit；即使两个不同record_job run_id也不能并写。finally清理本进程持有锁；强制终止残留锁必须先核验记录PID已退出再显式处理，不能自动覆盖活动锁。每个成功TRAIN/DEV请求追加 `requests.jsonl`，保留步号、父/指令、K、抽到的参考索引、prompt/全序列/监督token、loss与实际耗时。若作业在两次checkpoint间中断，超出原checkpoint的已记录请求不会从预算消失；恢复后重放按额外曝光计入总表。只允许完整journal；尾行损坏要显式保留/修复，不能默默删除记录。失败或正在进行但未写完的请求仍以record_job原始日志为限，不能声称其成本精确为0。累计耗时计每段模型/数据加载与执行，并保留额外已记录重放成本。`gpu_hours_reserved=elapsed_seconds/3600` 包含startup与教师强制DEV，是占用墙钟时间，不是自回归时延或按GPU利用率归一化的时数。

## 已验证与待验证

本地12项纯测试实际通过（新增7 + 既有序列5，0.57秒）：K4包含所有已采正例、K1均匀、父→语言采样、显式Generator恢复、无参考边界、观测白名单/答案前缀、父泄漏与配置拒绝，以及exclusive锁/异常清理、原始内容读取前的父角色拒绝。新增 `tests/test_vlm_sft_training.py` 将直接调用正式训练循环，以tiny因果模型检验连续4步和总步数仍为4的2步→resume逐元素相同，包括adapter/optimizer/scheduler/RNG/sampler/history/曝光；此测试不冒充真实Qwen模型恢复。冻结前不启动服务器计算或GPU。

待root冻结后CPU1测试入口：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /home/wzy/dpvlm/route_set_v1/.venv-qwen/bin/python -m pytest -q \
 tests/test_vlm_sft_data.py tests/test_vlm_sft_training.py \
 tests/test_vlm_sft_loss.py tests/test_vlm_route_serialization.py
```

实际训练命令应由root在固定release下record_job包装，fresh output，不在旧预检目录执行：

```bash
python -m scripts.train_vlm_route_sft \
 --observations /home/wzy/dpvlm/route_set_v1/data/observation_obstacle_reserved_development_v1/observations.jsonl \
 --supervision /home/wzy/dpvlm/route_set_v1/data/observation_obstacle_reserved_development_v1/supervision.jsonl \
 --model /home/wzy/dpvlm/route_set_v1/data/qwen3-vl-2b-instruct-89644892 \
 --output /home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_v1/seed0 \
 --steps 1500 --batch-size 1 --seed 0 --dev-plan-seed 200000 \
 --eval-every 250 --checkpoint-every 25 --lr 0.0001
```

## 随后的自回归对照（入口尚待实现）

同一个best checkpoint：独立4次K1与一次K4，均temperature0.7/top_p0.9，seed0；K1每次max_new_tokens512、K4为2048。每次仅传 `prepare_prefix` 的观测输入，未使用教师强制答案。各重复分别独立评价再平均，不合并池。每条原始输出、token、截断/格式失败、缺失/额外路线槽全部保存；严格parse和原Tip检查在生成后独立处理，无修复/重试/从额外候选挑K。独立方案4次读取/processor/视觉与语言编码全部计入端到端延迟；模型加载与首请求另列。正式SFT和自回归结果未测前，不称其为已收敛强基线或方法正结果。
