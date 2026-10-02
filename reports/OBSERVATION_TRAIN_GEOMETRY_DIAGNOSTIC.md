# 障碍TRAIN参考与预测几何一致性诊断

固定 `3fc6fd77c060bfb5b94bca3466be56837afb4fe4`，CPU1实际完成helper自测和全部32 TRAIN父诊断，均exit0；主诊断耗时4.677秒。只按预先父272000–272031过滤后解码监督记录，没有打开DEV原始图像、轨迹或标签文件。此次模型诊断仅seed0，不外推三种子结论。

使用原 `observed_obstacle_tip_eval_v1`：目标身份与3cm末端、5mm起点、恒定reach事件、每条tip线段对实体箱体膨胀2cm后的精确相交检查。函数直接复用现有evaluator与collector几何helper，不改阈值、不修复或筛掉输出。它仅检查新增箱体和tip，不认证整臂、其他环境物体、IK或机器人执行。

## 181条正参考不存在此处的训练/评价冲突

96个TRAIN指令中94个有正参考，共181条；另2个无参考指令仍进入模型预测评价。全部181条原始记录路线和训练器实际使用的 `resample_event_segments(..., horizon=24)` 都通过原TipValid，语义、起点、事件、有限性与碰撞门各0失败。没有raw有效但H24失效的例子。

collector保存的 `xyz_24` 也181/181满足2cm检查，与训练H24坐标最大绝对差仅5.959e-8m。70条参考的通道类型未知，但它们仍是有效正参考，没有被判为无效或拿来虚构类型数量。该结果排除了这批TRAIN中的安全余量矛盾及H24重采样破坏正例；不证明所有未来任务或轨迹表示都不存在此问题。

## 预测失败来自路线本身，定位改善后仍明显

每臂下列分母均为96指令×K4=384候选，包含2个无参考指令。best为原DEV参考ADE选择的soft500/peak750，last为双方统一step1000；best直接读取原保存TRAIN预测，last从相同固定checkpoint在CPU重新前向，没有训练。四组均有限、起点正确、事件正确。

|模型/检查点|语义正确|tip线段安全|TipValid|语义正确但碰撞|
|---|---:|---:|---:|---:|
|soft best500|44/384（11.46%）|153/384（39.84%）|19/384（4.95%）|25|
|soft last1000|256/384（66.67%）|209/384（54.43%）|135/384（35.16%）|121|
|peak best750|101/384（26.30%）|288/384（75.00%）|84/384（21.88%）|17|
|peak last1000|338/384（88.02%）|220/384（57.29%）|184/384（47.92%）|154|

每个门独立统计，允许同时发生身份与碰撞失败。peak last的338个正确目标候选中仍有154个碰撞（45.56%），所以后续问题不能继续只归因于终点定位。best与last的取舍也说明单纯较短或偏向错误目标的路线可能更少触碰箱体，不能只报TipClear。

在94个有参考TRAIN指令（376候选）内，soft last为256语义正确/204 tip安全/135 TipValid；peak last为338/212/184。与整体结论一致，不能通过删去无参考输入制造提升。全部94子集和2无参考子集分别保存在诊断JSON。

## 首次碰撞位置与下一步可验证假设

使用同一碰撞helper逐段定位最早失败线段，同时按预测路线自身累计弧长归一化，不用参考路线位置替代。peak last的164个碰撞候选中121个的**首次相交线段起点**在前25%路程；仅限目标正确的154个碰撞候选，其中115个符合这个线段起点条件。它是精确接触进度的下界，不能写成121个实际接触点都在前25%。逐候选首次线段索引、端点和`normalized_arc_before_segment`已保存；这些是原384个输出，没有追加候选或挑选成功子集。

12:24 UTC澄清：早先文字把线段起点简称为首次碰撞位置，不够准确。此次只修正文案解释，原JSON、碰撞检查、TipValid及选模标准完全保留。后续固定8 TRAIN父的局部支撑审计另计算了AABB精确首次接触：33个碰撞中17个接触点位于前25%、30个位于前50%。该8父子集不能直接与上述32父总数比较；新的prefix=.50由这个TRAIN精确接触诊断预声明。

当前证据支持下一轮集中测试“路线中间段如何保留绕障关系”的机制，同时保留已修复的共同观测定位编码。原集合回归已用正例集合匹配，不能把Hungarian或多个输出头重新命名为创新。新机制应只使用同一RGB-D/语言/当前状态与参考路径监督；若引入额外几何损失或检查，必须让比较方法获得同等信息并计入预算。此次诊断只确定需要解决的失败位置，没有宣称已确定唯一成因或实现新方法。

完整181条raw/H24结果、四组1536候选独立失败门、模型/预测/原始TRAIN文件hash与运行源码见 `reports/observed_obstacle_train_geometry_v1/diagnostic.json`；last的真实TRAIN预测见 `runs/observed_obstacle_train_geometry_v1/`。实际命令、PID、exit与日志在 `reports/observed_obstacle_train_geometry_jobs/`。命令：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CODE_COMMIT=3fc6fd77c060bfb5b94bca3466be56837afb4fe4 \
/home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.diagnose_observed_train_geometry \
  --root /home/wzy/dpvlm/route_set_v1 \
  --data /home/wzy/dpvlm/route_set_v1/data/obstacle_learning_curve_new32_v1 \
  --output /home/wzy/dpvlm/route_set_v1/runs/observed_obstacle_train_geometry_v1 --seeds 0
```

运行目录为该commit的immutable release。重评另用新output，不能覆盖实际诊断记录。
