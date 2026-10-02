# TRAIN 路径局部观测支撑审计

2026-10-02；固定源码 `5501c5f464b61ae1a34e34c5381bbded306bf4ed`，CPU1/CUDA隐藏，正式自测和审计均实际exit0。**33个首次碰撞位置全部有10cm邻域可见点，普通局部几何读取有可用输入；这不等于已经证明局部更新能改善路线。** 已把结果交给普通local/global匹配对照，未实现新分配器或额外候选模块。

**弧长口径必须区分：旧121/164是首次相交线段起点的弧长比例，只是接触位置的下界；本报告定位闭箱精确接触点。** 原诊断JSON和原数值保留。固定8父内，同一33个碰撞在前25%范围，首相交线段起点计21个、精确接触弧长计17个、H24索引比例计26个；三者不能替换。

## 固定范围、数据和评价条件

仍用旧32 TRAIN 的预声明前8父272000–272007、全部24条指令。23条指令有40条正参考，另1条无正参考的指令仍计入预测检查。每条正参考按自身完整路径弧长在0–25%取17个等距位置（含起点），共680点；不按帧号或H24节点数取前四分之一。

预测来自旧32 peak seed0 last1000真实TRAIN文件，SHA256 `58fb8e350c044f93f8aaf8f7824e7acb0c58c9ab7df992ad097da68396d70b62`，没有新前向或训练。全部24×K4=96候选均检查；33碰撞、63无该箱碰撞。33处来自15条指令/7父，全部目标正确。真箱体仅作为评价标签，按原2cm膨胀AABB精确定位首次接触，逐项核验首次相交线段与原冻结诊断一致。箱体、目标真坐标和模式编号不进入点云构建或支撑函数。

点云直接调用训练器使用的 `backproject_rgbd`，保留相机焦距符号、采样像素偏移、float32坐标和有效深度mask。每父完整有效点50176个；原head stride2有效点12544个。所有半径均**以诊断查询点为中心**，不是以当前末端为中心。预声明半径0.05/0.10/0.20m，无DEV择参。

可见性仅按查询投影与同一当前深度比较：比实测表面前2cm以上、表面±2cm、遮挡后方、画外、无效深度和相机后方分别记录。前方标记只是深度射线代理，不代表整机器人安全。没有附近可见点，也不代表自由空间。

## 主要实测支撑

下表采用**模型实际stride2点云**。查询点同路径/同父相关，数量不是独立场景样本数。

| 查询 | 最近有效点距离中位/P90 | 5cm有点 | 10cm有点 | 20cm有点 | 5/10/20cm点数中位 |
|---|---|---:|---:|---:|---|
| 40正参考早段，680点 | 11.35/14.83cm | 99/680，14.56% | 231/680，33.97% | 655/680，96.32% | 0/0/112 |
| 33预测首次接触 | 2.22/3.55cm | 32/33，96.97% | 33/33，100% | 33/33，100% | 22/72/336 |

完整点云的正参考覆盖为112/237/655，碰撞处为32/33/33；使用4倍像素只小幅改变这些支撑率，不能把不足简单归因于stride2。按父等权后的stride2覆盖为正参考14.66%/34.30%/96.76%、碰撞点97.62%/100%/100%，与查询级判断一致；不据8父估计最终泛化或训练种子不确定性。

正参考680点中643在实测表面前方、32被当前深度遮挡、5在画外。32个遮挡点在10cm内均无可见点、20cm内全部有点；这表示附近仍有可见上下文，不能将遮挡空间填成已知自由。5个画外点在20cm内均无支撑。33个首次碰撞查询全部投影在实测表面前方；它们是膨胀箱的2cm安全边界接触位置，并非必须落在可见表面像素上。

首次碰撞距当前末端中位21.36cm、最大33.84cm；因此从当前末端固定取10cm球与“围绕草稿节点取10cm上下文”是不同操作。

| 归一化进度阈值 | 精确接触弧长 | 首相交线段起点弧长下界 | 接触H24索引比例 |
|---|---:|---:|---:|
| 0.25 | 17/33 | 21/33 | 26/33 |
| 0.50 | 30/33 | 30/33 | 30/33 |
| 0.75 | 33/33 | 33/33 | 32/33 |

正参考JSON中的vertex比例对应原始记录帧；预测对应H24，报告明确标记，不能用它们直接横向比较。

## 由TRAIN证据固定的下一步与边界

根线程已锁定一次普通对照的 `sigma=0.10m`、`prefix_fraction=0.50`、`bound=0.10m`，后续不从DEV扫值。sigma是高斯标准差，**不是10cm硬截断**：多数正参考早段离可见表面超过10cm，而20cm范围有96.3%支撑；sigma0.10允许利用这部分上下文。读取仍只是成熟局部聚合操作。prefix取草稿自身归一化弧长前半段，覆盖本探针30/33个首次碰撞；不改成固定H24前半节点或从真实碰撞mask决定更新位置。

**支撑距离不能证明修正幅度足够。** 当前实现 `bound*tanh` 的0.10m是逐坐标上界，欧氏位移最多约17.32cm；这个选择保留为两臂共同预声明条件，不能宣称TRAIN审计已证明可修复全部碰撞。固定起点/终点、同参数global普通适配器和同训练曝光对照的实际结果才会检验局部读取解释。该审计未训练权重，没有checkpoint；没有新核心机制有效性声称。

## 实际作业、索引与复现

自测PID305260/child305273，12:21:19.533388–19.997192 UTC，exit0。正式PID305276/child305277，12:21:20.057619–22.898440 UTC，exit0；进程墙时2.841秒，内部统计/IO0.9503秒（进程还含Python/torch导入），GPU小时0。CPU已释放，无本worker后台训练。

[summary](observed_local_support_train8_v1/summary.json)、[逐查询支撑](observed_local_support_train8_v1/per_query.json)、[三种进度口径与父等权统计](observed_local_support_train8_v1/comparison.json)、[作业命令与状态](observed_local_support_train8_v1/jobs/audit.status.json)、[图](observed_local_support_train8_v1/support_and_progress.png)、[哈希索引](observed_local_support_train8_v1/artifact_index.json)均已归档。713个完整查询坐标（680参考+33接触）保存在服务器及本地 `runs/observed_local_support_train8_v1/queries.npz`，SHA256 `f3e4beac818335b5b142deb7a3887c118b9b8a7e0669483cd80c4304870c9237`；二进制不进普通Git。复制后实际校验NPZ/JSON哈希及逐点对应通过。

summary SHA256 `edeaf01636c91018872c6c2844c60bad4eb161ded4a189a436a4eebb33568107`；实际部署脚本SHA256 `67350b08f0c66ae7f22e836fdabdbfaa3750eb6cb2637f31ba2e49119acee9b0`，换行规范化Git内容SHA `c1be847e5b9df59f4dcd13e0b5eea311b2be9cfccaaf54b9c060efd0a3f0632b`。5501包含可选refiner，整个geometry模块hash与旧训练版本不同，但调用的backproject原文未修改；实际源hash已在summary记录，不假装使用旧模块hash。

在固定release目录下使用服务器 `.venv/bin/python`，CUDA隐藏且OMP/MKL/OPENBLAS/NUMEXPR线程均1，按以下实测命令复现；须换全新output，没有resume，也不重跑训练：

```bash
python -m scripts.audit_observed_local_support --self-test
python -m scripts.audit_observed_local_support \
  --data /home/wzy/dpvlm/route_set_v1/data/obstacle_learning_curve_new32_v1 \
  --output /home/wzy/dpvlm/route_set_v1/runs/observed_local_support_train8_v1_reproduction \
  --diagnostic /home/wzy/dpvlm/route_set_v1/runs/observed_obstacle_train_geometry_v1/diagnostic.json \
  --diagnostic-sha256 2e3420bb5e99b61f63c481aec381d7d665ae6fcf62ac94e9ef69831354642ebf
```

本地归档验证与图复现入口：`python reports/observed_local_support_train8_v1/reproduce_archive.py`。
