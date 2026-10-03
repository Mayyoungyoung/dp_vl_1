# 仅 TRAIN 的普通扩散 Min-SNR 加权诊断协议

状态：实现准备；尚未服务器测试、读取实际 PT、训练或生成。研究依据见 [FIT_AUDIT](OBSERVED_DIFFUSION_FIT_AUDIT.md)。这是单一标准优化修复，不是新核心方法。旧 6e 扩散源码、配置、权重和结果保持不变。

## 固定科学比较

同一个已完成的 independent last12000 分成两个新的输出 lineage：`uniform` 保持旧逐元素 x0 MSE 的精确 reduction；`min_snr_5` 对每个输入的 4×23×4 MSE 乘 `min(alpha/(1-alpha),5)`，再除以这 100 个固定权重的算术平均。只改这个时间步权重，grounding loss 和 .02 系数、观测白名单、预测 anchor、末端 .05m 范围、事件 .2、所有网络参数、constant LR 3e-4、beta／t／epsilon、40 DDIM／K4 都不改。

只使用 x0 版本，不能误用 epsilon 版本的 `min(SNR,5)/SNR`。平均归一化是本项目明示的常数尺度控制，不是作者原样代码；实际 100 权重和 SHA 写入新 config。gamma 不扫。高 t 权重很小，必须报告高噪声恢复或自由生成退化，不能仅报低 t 下降。

## 原始权重与恢复的实际门禁

唯一 parent 为 source `6e0203ba1335f9fa9975c523c657959f8bc9ab60` 的 independent last12000，PT SHA `53b8550cc950be68c39fbabc46efbf7a1143429c4f13a5251c37da9632deaa8d`。config／summary／原 ordinary 源文件索引 SHA 固定在新增 JSON 中。

实际 PT 尚未在本轮打开。冻结源之后先且仅先独立执行 `--stage inspect-parent`：CUDA 隐藏、CPU1，核 PT SHA 后 `torch.load(map_location='cpu')`，不构造模型、不读取数据。实际 keys 必须包括 model、optimizer、scheduler、rng、stream、stream_audit、journal、step、history、best、实际曝光及 pending_gradients。检查 scheduler.last_epoch=12000，四流实际为 parents／positives（抽样和置换共用）／times／noise，训练 RNG 为 NumPy generator、NumPy global、Python、Torch CPU、CUDA。任一字段缺失或不符就拒绝，不猜测或补造。原 summary 的流／journal 必须等于 PT。

每个 run 必须绑定该同源实际 inspection receipt。共有模型、AdamW 动量及 step、constant scheduler、训练 RNG、四个抽样 RNG 全部真实恢复；恢复后再次对五类状态逐字节 digest，与原 PT 对照并写 `restore_receipt.json`。没有 new Adam，也不加载原 best。两臂新 journal 从空开始；旧已完成的曝光、成本和 48 DEV history 仅作 lineage，不能算为新增调用或新增选模。

## TRAIN 数据身份，不重新读取 DEV 数组

仍为固定 composite108 的 285 实际 TRAIN 输入／1663 正参考，缺失父不替换。先核原 observations、supervision、cache_config、export、quality、cache receipt 及原 `peak_seed0/source_hashes.json`。只解码所选 TRAIN 的监督行，只打开其图像、当前状态、Qwen NPZ 和全部正参考；每项实际 SHA 必须等于原索引和已完成的 cache receipt。

TRAIN compact loader 复用原 H24 event resample 和缓存输入白名单。随后保留原 321 索引空间，36 旧 DEV 仅为空数组占位；旧 parent stream 的 TRAIN 索引位置不改变，实际抽样只能落在 TRAIN。原 full dataset fingerprint 明记为历史 identity 引用，不伪称本次重读了 DEV。重新计算的 TRAIN target bytes／refs／canonical IDs 必须让原 `PairedPositiveStream` identity 原样恢复，否则停止。观测几何只从 TRAIN RGB-D 与当前状态调用原 `read_geometry`，训练 point encoder 仍每步真实执行。

post-generation checker 使用仅含 TRAIN 行的独立 metadata view。其 scoped verifier 只返回已经固定的原 export label-hash 索引；原 two-row checker 逐个验证实际 TRAIN 验收文件 SHA 和原 2cm／3cm 标准。数据和 labels 不进入额外的 model conditioning；unknown 正例不删、不变负。0 DEV forward、0 DEV 选模；新 DEV／reserved 不得读取。

原 evaluator 为兼容仍输出 `selection_score`／历史 selection protocol 字段；本入口不调用 `accept_selection`，这些字段不构成新选模机会。所有本轮对照固定使用 step12500。

## 调用和预算

每臂只增加 step12001–12500：500×32=16000 输入抽样、64000 正目标槽。训练 journal 逐 batch 记 geometry／denoise／optimizer；新段独立 digest 覆盖 indices、reference indices／permutation、t、epsilon 实际字节。两臂 final cumulative stream 和 new-only stream 均必须完全相同。

完成 500 步后在同一臂内依次运行已批准的两个固定诊断，没有 DEV checkpoint selection：

| 阶段／每臂 | geometry 调用 | denoise 调用 | optimizer | 计费状态 |
|---|---:|---:|---:|---:|
| 新训练 500×32 | 500 | 500 | 500 | 64000 采样正目标槽 |
| 原首6 TRAIN × 5 t | 6 | 30 | 0 | 120 中间路径状态 |
| 全285 TRAIN，K4／40步 | 285 | 11400 | 0 | 1140 最终候选；45600含最终的去噪更新状态 |
| 合计 | 791 | 11930 | 500 | 分项保留，不混作1140×额外候选 |

teacher 的原正参考选择／NumPy seed400000、t=[0,25,50,75,99]不变，新增 body RMSE 为排除末端的后22点 XYZ；原后23点 XYZ RMSE、末端欧氏距离、event MAE 全保留。自由池固定 seed300000、原全部285输入、无筛选、重试、修路或合池；首点硬约束和端点预测规则原样。所有原 metrics/per_scene/NPZ 保存，额外 Qwen 编码为 0。teacher 不作为生成成绩。

每臂 360 秒累计上限，含脚本加载、校验、训练、两诊断和检查；每次新模型／optimizer 调用前以及更新边界检查。单次 CUDA 操作不能中断，最后的 CPU 封存亦可能使实际 walltime 超过边界，必须记录 overshoot，不能声称严格实时截止。各 attempt 的外层实际 elapsed 累加用于恢复，不能用 checkpoint 时间回滚成本。主体／stage 计时嵌套于该时间，原12000历史成本不重复记。实际 record_job 外层（含解释器启动）另记，不能与内部时间再相加。

## 暂停、失败与零隐形重放

每25步保存 model／AdamW／scheduler／训练 RNG／四 stream RNG／journal／incremental digest 全状态。`--stop-after` 是绝对 logical step 的管理边界，不进入科学 policy；如12002、12025、12500均可完整保存并返回 paused，不生成 completed summary。恢复命令显式去掉 stop-after，仍有相同剩余500步总预算。

硬崩溃后 journal 比 checkpoint 多出的真实 issued 调用不回滚。即使已计算但未保存也不能静默重做；缺最后 attempt 成本／状态时 fail closed。失败调用只保证已发出、保守计费，不能称全部已完成。完整 teacher 或完整自由池有严格 receipt／model identity／journal／必要文件 hash 时可零 forward 复用；残留 `.staging` 保存 NaN、partial rows／journal／失败说明，自动重复被拒绝。

输出含每次 attempt PID、状态、退出对应异常、精确恢复 argv、最后 checkpoint、partial 和 source identity。record_job 的真正退出码／launcher SHA 由 root 的冻结 launcher 记录；不得把本脚本 status 代替外层退出证明。运行期间不覆盖源或 shell。

## 比较、判别与停止

`--stage compare` 是后续独立纯 CPU 读取阶段，不导入模型、不加载 PT 张量、不读取原 label/raw；它只 hash 两臂 final PT 和复核两个封存池，核共同 parent、initial components、实际抽样链和相同 teacher/free noise。保留所有285逐输入配对、完整指标，以及 t0/25/50/75/99 teacher 数值。

最低继续条件是 **t0 与 t25 的 body RMSE 均下降，且全285 TRAIN自由 TipClear 与 TipValid 均上升**。这只是进一步审阅资格，不是统计显著或方法成功声明。仍必须查看语义、已知 Unique、unknown、coverage、ADE、逐父分布及高 t 退化。只改善 teacher 或只改变端点／unknown构成不足以升级。未满足就停止此唯一加权方向，不扫 gamma／seed／长度。即便满足，也不能将本次从旧权重额外续训的 TRAIN 诊断加入 MAIN 正式结果；正式强基线比较须 root 另定同预算协议。

## 冻结后入口（尚未执行）

所有命令须由 root 填入实际 immutable release、record_job、CPU1／GPU1 UUID／35% 上限和真实文件路径；下面只列分阶段 API，不自动部署或启动：

```text
python -m scripts.train_observed_diffusion_minsnr_probe --stage inspect-parent \
  --parent-run <original-independent> --output <fresh-inspection.json>

python -m scripts.train_observed_diffusion_minsnr_probe --stage run --arm uniform \
  --parent-run <original-independent> --parent-inspection <inspection.json> \
  --ordinary-source-hashes <original-composite/peak_seed0/source_hashes.json> \
  --data <fixed-composite108> --quality-audit <fixed-quality.json> --output <fresh-uniform>

# 同一批参数，第二次明确 --arm min_snr_5 --output <fresh-weighted>
# 管理性暂停后，仅同输出 --resume；不保留 --stop-after。

python -m scripts.train_observed_diffusion_minsnr_probe --stage compare \
  --uniform-run <uniform> --weighted-run <weighted> --output <fresh-pair.json>
```

本地纯测试和实际 Torch 工程验证必须区分。本机未安装 Torch，实际恢复／梯度／权重测试由 root 冻结后在既有 `.venv` CPU1 执行，0skip才可视为通过；不新装共享环境。
