# 两排观测任务：独立288父同分布扩规模预登记

目的：检验更多独立TRAIN父是否缓解现有63个可观测TRAIN父上的过拟合。cosine控制已改善TRAIN拟合但未改善DEV；本次增加数据不构成新方法、不改变任务层定义，也不是OOD。旧116语料及其源代码、配置、角色和所有失败保持不变。本登记不授权模型训练或开启新DEV内容。

## 固定登记与读取边界

一次性登记288父：index0–255为TRAIN，seed400000–400255；index256–287为DEV_MODEL，seed400256–400287。新布局RNG固定399999，按index递增，每父先12个几何uniform再官方20色中无放回选3色；不按结果或稀有颜色重采。每父同一初始RGB对应3目标语言，每目标9个预声明提案，共864请求条件、7776路线槽。每槽至多9个引导段规划API入口，路线层最大69984段；内部OMPL配置搜索不称为一个额外完整路线。所有失败、unknown、近似重复与未尝试槽保留。

TRAIN固定启用完整注册前缀32/64/128/256，对应96/192/384/768个请求输入；实际缺图或零参考不得补造、替换父或在分母中消失。旧12DEV是已经反复使用的开发证据，必须标reused。新32DEV在全部256TRAIN闭合后才单独采集，raw图像/路径/验收结果随后继续封存，待另一个预先冻结的模型/机制确认协议才允许分析。名字DEV_MODEL不使它自动成为日常调参池；也不把它称TEST_LOCKED或OOD。

几何范围严格同旧窄ID设置：row x分别[.135,.145]、[.335,.345]m；每排两柱y分别[-.115,-.105]/[.105,.115]及[-.155,-.145]/[.145,.155]；柱尺寸不变且高度.14m；goal x[.45,.47]、三目标y[-.175,-.145]/[-.015,.015]/[.145,.175]、z=.84。固定canonical关节/夹爪与已核验模型资产和Arm源SHA，entry声明[0,0,.865]，真实FK仍逐父核验。没有执行setup路径的主张；初始化新增IK/path预算均0。

输入仍只能RGB、语言、实际depth/camera、当前gripper pose/open；目标/柱真值、模式、规划引导、掩码仅采集或评价标签。复用 `collect_two_row_canonical_repair.run_collection` 原字节，包括完整机器人离散步碰撞、初始可见性、几何和RGB-D审计、每候选同native snapshot的严格恢复、tip/H24检查及实际穿越类型规则。没有放宽门槛或给模型额外几何。离散仿真步碰撞与tip线段检查不等于全身连续执行证书。

## 历史机械排重和一次注册

排除表 `configs/observed_two_row_extension288_exclusions_v1.json` 包含128项：公开v1/v2/v3配置各1、原v4/v5/v6九项、旧正式116父全部注册元数据。包含16个旧LOCKED父的仅ID、角色、精确/1mm几何hash与登记来源SHA，不打开旧锁定图像、路径、结果、closure或成功率。已知配置几何哈希不读取验收数据。entries canonicalSHA `fbad0c35f270b064f091d9c32d09f5f51f54c7bc5aff06ce7457918435ee99fa`。

哈希定义复用四柱中心/半尺寸和三目标中心；不含颜色、seed或机器人FK微漂。新内部同角色/跨角色重复组全部新成员关闭；与任何历史几何碰撞的新父关闭，27槽仍登记未尝试，不替换、不修改旧角色。实际采集的1mm几何仍须匹配本父注册哈希，后续全局机械门禁还查实际重复组；失败不冒充独立可用父。

已在本地完成一次固定288纯注册，生成 `configs/observed_two_row_extension288_registered_v1.json`，SHA `574167e8a818dc6ee3d6ac197c6436e8031489306b2f8514466bd764f0cac70e`；288个理想tip引导/H24预检通过、0历史/内部注册重复。仅是采集前检查，没有运行模拟器、IK、路线搜索或训练，不证明实际机器人可达。首次直接文件启动缺PYTHONPATH在import处退出1、未创建登记文件；随后使用 `python -m scripts.register_two_row_extension ...` 成功，未改变随机种子/参数或重采结果。

服务器已只读扫描全部data目录名，400000–400287无匹配；local configs同范围和399999也无旧记录。seed不碰撞不是布局独立证明。注册与机械排重才决定采集资格；新corpus输出目录必须fresh，禁止旧数据覆盖。

## 五个独立采集阶段与恢复

`tests`、`prepare`和每个采集阶段都是独立launcher动作，任何动作都不会自动启动下一阶段。prepare只核验/保存整个注册和来源manifest，不初始化模拟器。root读门禁后分别启动：

| 动作 | 此时允许的注册父 | 启动前必须已闭合 |
|---|---|---|
| train32 | TRAIN0..31 | 无 |
| train64 | TRAIN0..63，仅采未闭合 | 全TRAIN0..31 |
| train128 | TRAIN0..127，仅采未闭合 | 全TRAIN0..63 |
| train256 | TRAIN0..255，仅采未闭合 | 全TRAIN0..127 |
| dev32 | DEV256..287 | 全TRAIN0..255 |

因此较快shard不能提前进入新DEV，不设置隐性等待循环。前序父的失败闭合也计入闭合；不能等待一个父成功才推进。每stage两条shard分别CPU2/3单核、所有线程1、GPU隐藏、软件渲染；全288按偶奇分片各144父。通过PID锁拒绝活跃重复job。每个父单独record_job，原始stdout/stderr（包括DEV结果）进入日志文件，协调器只输出ID/role/closure，不打开raw结果。

源文件、配置、注册字节与所有复用旧依赖均入manifest；resume必须字节一致。来源只新增文档的另一release若采集依赖字节相同可通过，不凭不同路径冒充新实验。父配置先冻结；parent worker启动前验证官方颜色表、v4 canonical来源joint与模型资产SHA。

遇已闭合父：核机械receipt/ledger/artifact hash与role/index后跳过。遇已有父data或status但无closure：按已发槽账本闭合为中断，不重放任何已发或剩余槽；不因进程退出再从头采同父。worker启动前失败也写0已尝试、27未尝试closure。runtime/shutdown错误exit1并停止该shard；修复/恢复只能继续之后未发父，原失败保留。正常严格初始化门禁关闭属于已完成负采集，不伪造轨迹。部分账本行保留attempt上下界。

## 资源与内部预算

旧116两CPU外层墙钟9604s，含启动/检查/协调，不是实际CPU利用率；原数据909,823,031B、run日志13,420,691B（2026-10-02只读`du -sb`实测）。按相同分布线性粗估：新288约6.62h双CPU墙钟、13.25核预留小时；前32/64/128/256 TRAIN累计约44min/1.47h/2.94h/5.89h，最后32DEV约44min。实际随机规划耗时可能不同，逐父实际wall必须记录，不承诺这些数。

原字节数线性估计新288数据约2.26GB、日志约33MB。root已授权本轮额外内部预算8GiB；代码按新corpus+run根的逻辑文件bytes核计（仅stat，不读内容），每发新父前为两个并发父共预留512MiB，每父关闭后再检查。超限暂停exit3、保留所有数据，不能删除失败腾预算或自动扩quota；此为父边界预算保护而非操作系统实时磁盘配额。snapshot/cache/归档另行计入本轮总预算，不借由路径移出规避限制。当前disk约1.2T free，不据此提高CPU/GPU配额。

这次登记不决定模型训练预算。后续32/64/128规模普通对照必须分别声明固定总曝光和按每输入曝光对齐的区别，不能再把明显欠拟合基线当机制收益。任何组合旧TRAIN+新增TRAIN的快照须另行冻结完整父选择与去重门禁；本入口不会自动训练。

## 实测/启动命令边界

本地纯注册/测试已运行；服务器新tests/prepare/采集尚未运行，只有root在新commit部署后可以执行。不可变release用root实际SHA，不填猜测提交：

```bash
bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/REV/scripts/launch_two_row_extension288_v1.sh REV tests
bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/REV/scripts/launch_two_row_extension288_v1.sh REV prepare
# root 独立读 prepare 后，才发第一个采集阶段
bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/REV/scripts/launch_two_row_extension288_v1.sh REV train32 fresh
# 后续各阶段由 root 逐次明确启动；中断同阶段也使用 resume
bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/REV/scripts/launch_two_row_extension288_v1.sh REV train64 resume
```

独立输出为 `data/observed_two_row_extension288_v1` 与 `runs/observed_two_row_extension288_v1`。每session保存不可变LF launcher/wrapper、source hashes、完整命令、PID、status、起止与退出码。Xvfb只清理当前wrapper实际拥有的子进程。旧116十个Python源、原配置和两launcher在实现前后逐SHA核验，无源修改、无新server job、无自动DEV或正式扩大。
