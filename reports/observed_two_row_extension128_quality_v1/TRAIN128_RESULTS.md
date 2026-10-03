# TRAIN128 质量与完整归档结果

128 个已注册 TRAIN 父场景、384 条指令的 3456 个请求槽已全部闭合并完成固定源码质量分析：2248 条接受、1208 条失败。新增 64 父场景贡献 1117 条接受参考，接受率 64.64%，与前 64 父的 65.45% 接近。全部失败、unknown 正参考、长回环和唯一已知类型重复均保留；本轮没有训练或方法收益结论。

| 范围 | 父 / 指令 / 请求槽 | 接受 / 失败 | 已知 / unknown 正参考 | 已知 R>K4 条件 | 严格恢复 |
|---|---:|---:|---:|---:|---:|
| 前64（indices 0–63） | 64 / 192 / 1728 | 1131 / 597 | 666 / 465 | 50 | 1728/1728 |
| 新增64（indices 64–127） | 64 / 192 / 1728 | 1117 / 611 | 644 / 473 | 36 | 1728/1728 |
| 累计128 | 128 / 384 / 3456 | 2248 / 1208 | 1310 / 938 | 86 | 3456/3456 |

统计来自原 `analysis_run/train128/analysis.json` 和完整 `all_requested_slots.json`，按原槽身份一次分组；`TRAIN128_SUPPLEMENT.json` 保存精确数值和分母。没有其他、未尝试或缺图槽，没有 model-use blocked 父。每条件接受参考 2–9 条。累计 unknown 占接受参考 41.73%，新64为42.35%；unknown 不是无效或负例。R 是已采到且可分类的不同类型数，不是真实全部解数。

新增条件中 `400067_target1`、`400070_target2` 的两条正参考及 `400078_target0` 的四条正参考均为 unknown，不能报告为没有有效路径。唯一已知类型重复发生在 `two_row_reach_400075_target1`：5条正参考含2条unknown、2种已知类型，slots4/7均为 over/middle；实际图显示一条局部抬升、一条大侧向回环。这是同类不同曲线，保留原监督，不据此推断数据重复污染。

| 接受路径统计 | 前64 | 新增64 | 累计128 |
|---|---:|---:|---:|
| 长度中位数（m） | 1.375553 | 1.379498 | 1.377395 |
| 长度 P95（m） | 3.059244 | 3.235640 | 3.199317 |
| 最长（m） | 6.569143 | 6.621788 | 6.621788 |
| 长度 >4m 条数 | 25 | 19 | 44 |
| 端点误差中位数（mm） | 1.303494 | 1.309958 | 1.307401 |
| 最大端点误差（mm） | 3.583517 | 2.413423 | 3.583517 |

新增最长路线为 `two_row_reach_400087_target0` slot6，6.621788m，trace SHA `84cb9e3be72f5ab149c3098406a6fe710c2f66860823929e48b4b94ec4dd479a`，是被原检查接受的 unknown 大回环；邻槽7也为大回环但原判失败，两者均展示。没有按长度或视觉简洁度筛除正例。20个预注册颜色均出现，精确指令频数在 supplement；它们仍是同一窄几何范围的两排派生任务，不是新增任务或 OOD 证据。全部3456槽的事件转移数为0，本数据不提供交互事件多样性证据。

新增611失败的首错误为529次原 endpoint/raw/H24/type 联合验收失败和82次模拟运动中的 robot collision。重叠诊断分别为 raw-tip 350、H24-tip 348、type-changed191、endpoint>3cm82、robot collision82，不能把它们相加当失败总数。累计首错误1039+169=1208；累计重叠诊断 raw-tip712、H24-tip694、type-changed357、endpoint>3cm169、robot collision169。失败不会因重新投影或某一项通过而升级为接受。

新增64父的实际视觉复核已全部完成：`VISUAL_QA_CONSTRAINT_UPDATE_64_95.json` 记录本agent逐页看过 indices64–95 的32张父联系表（96目标×9槽XY/XZ）及原分辨率front page1/2；`QA_ROOT_096_111.json` 记录 root 看过96–111的16张父联系表及front page3；`QA_LOCAL_112_127.json` 记录 controlled_data 看过112–127的16张父联系表及front page4。`VISUAL_QA_COMPLETE.json` 核验三份实际receipt与全部64张父联系表、4页front的SHA及完整覆盖，没有将分配或生成库存当成实际查看。后两位审阅者以2048×1072显示父联系表，原2770×1450版本保留。所有已查看图均保留三目标/九槽、失败partials和unknown，未见缺面板或图像损坏。部分原全槽图的宽公共坐标轴造成小字标题/轴标签重叠，完整2730×1300原图仍保留；联系表只能检查呈现和记录覆盖，不能替代3D或全臂安全判断。

`qa_views/` 有全部新增64张父联系表和4页原分辨率front；原192张九槽图和64张front完整保留。`VISUAL_QA_PENDING.json` 是生成时的初始库存（viewed=false），最终实际查看范围以独立QA receipts为准。前64的256张原图与先前TRAIN64归档逐字节相同，计数回归也相同；旧视觉复核引用先前报告，不宣称本轮重新目视旧64。

采集 source `5c8f8e4f5cd478c793a0e0d9640005deaf700973` 的追加64 session 自01:16:26至02:45:10 UTC，两个shard均exit0，外层5324秒。新增64 worker elapsed合计10606.038528秒，内含 planning1289.927328秒和simulation2725.412851秒；这些是嵌套墙钟记录，不能相加成CPU/GPU成本。新增14955次实际get_path，累计29879次；不能把3456候选槽当作仅3456次底层规划调用。无新采集由本归档启动。

全128只读质量分析 source `1a3eef1fb12d55e98d4d188a091ea40ea62c0a02`，PID720255/child720256，自2026-10-03 02:47:03.411313至03:02:14.984262 UTC，exit0；主体909.658842秒、外层911.572949秒，嵌套不能相加，GPU小时0。实际冻结wrapper `source/extension128_quality_1a3eef1.sh` SHA `24c8af0f0e950c16c9ce3105b0ff50a18e25d9f59218d15e8bb888afa7921ad0`。成功后才启动本地只读归档/补充统计，没有新模型forward、search、仿真或reserved原始读取。

归档2242个服务器原件，共139441807字节，全部server SHA逐项验证。`SERVER_ARCHIVE_INDEX.json` 含原服务器恢复路径，`SYNC_RECEIPT.json` 含tar和旧图字节回归，`LOCAL_ARTIFACT_INDEX.json` 索引原件与本地补充产物。完整128父status/log共256文件、采集session和双shard状态、注册/closure、collector/analyzer真实源码与wrapper均归档。归档tar SHA `36778f183ed253f87581c13c210e39ee684a7ff0ae3955c87f7b621ba2265ce8`。

关键文件SHA：

- 原analysis JSON：`0145fc5d01f2a8d59f9e735ba650803a33f0fe003b93246605578be862a73f74`；其中analysis_source_sha256是源码值 `758235704e344b0f34cce414b0bacb8f28027ef6b1eaef6f96e21279838bfcba`，二者不混用。
- 注册JSON：`574167e8a818dc6ee3d6ac197c6436e8031489306b2f8514466bd764f0cac70e`。
- 完整槽JSON：31231795字节，SHA `74c649eeb6f2c21ee82b8e1f81a0e29900c19dc6ea0bc5591f81b84b55fefa10`，原件 `/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_extension288_quality_v1/train128/all_requested_slots.json`；本地同路径后缀按精确规则ignore，不进入普通Git，未删除或从统计中遗漏。

实测命令已保存在 `analysis_run/train128.status.json`：固定release下 `.venv/bin/python -m scripts.analyze_two_row_extension_train --corpus /home/wzy/dpvlm/route_set_v1/data/observed_two_row_extension288_v1 --train-parents 128 --output /home/wzy/dpvlm/route_set_v1/runs/observed_two_row_extension288_quality_v1/train128`，CPU0、隐藏GPU，fresh-only，不应重跑覆盖。归档的两个本地helper也在source/保留；刷新索引使用 `.bootstrap/extension128_quality_supplement.py --root-confirmed-archive-complete --index-only`，不会重新读取服务器raw或生成轨迹。

该批只证明已完成采集在既有验收规则下的质量和可追溯性；仍需独立训练器重采样/端点容量门禁才能用于新训练。不能从本结果推断完整连续机器人执行安全、完整解集、学习收益或论文方法优势。当前composite108训练人口不变，新DEV32继续封存。本归档没有启动采集或扩训；root读完质量与完整QA后决定按原5c8源码恢复train256、仅追加128–255，具体启动与退出另记，不能由此报告推断已采集成功。
