# 训练末端监督的传统颜色观测定位对照

本实验暴露了当前学习头的定位缺口。在同一组已经反复用于开发的8父场景、24条指令上，传统RGB-D颜色原型定位达到严格语义79.17%或83.33%，高于当前普通/grounding辅助集合头。它仅输出一个观测表面点，不产生完整路线，不能据此报告Valid@K、机器人执行率或候选路线覆盖收益。

## 固定机制与信息边界

`scripts/observation_prototype_grounding.py` 在测试前固定所有参数。对TRAIN有成功参考的指令，从参考路径末端4cm邻域选取当前图像可见RGB-D点，学习每个完全相同instruction的颜色及平滑chromaticity中位原型、颜色尺度、训练99%距离阈值和可见点云尺度。父场景/指令的统计等权，不因某父有更多示范而增加权重。4cm是训练正例像素邻域，不是修改3cm成功标准。

预测函数只接收当前RGB、米制深度、相机内外参和instruction。颜色阈值分割后，8连通区域按颜色似然和训练可见尺度软先验评分；返回最佳区域中距离三维中位数最近的一个实际观测点。没有使用测试目标、仿真几何、实例mask、未来路径、可达性答案或GT裁剪。未知instruction显式拒绝且计入失败。本轮训练集覆盖20条颜色指令，DEV未知指令为0；它是封闭词表定位，不是开放词汇方法。

预算为一个端点提案，所有拒绝也计入分母。图像内部连通区域是视觉定位假设而非隐藏生成路线，但逐请求区域总数、候选区域数及处理耗时均报告，不把CPU视觉处理当成零成本。没有Qwen编码步骤，也不把它伪装为同骨干路线方法。

## 实测结果

|训练来源|TRAIN父/有参考指令|DEV语义分母|严格身份且3cm命中|目标误差均值|CPU单请求中位/P95|
|---|---:|---:|---:|---:|---:|
|旧32父数据的TRAIN|24 / 71|24|19/24，79.17%|33.50cm|12.49 / 16.06ms|
|新32 TRAIN父快照|32 / 96|24|20/24，83.33%|21.22cm|10.73 / 17.62ms|

两次均没有拒绝，平均目标误差包含全部24条预测；有参考末端误差分母23，分别22.03cm与22.04cm。旧训练版本漏掉无参考DEV的cyan目标，新训练版本将其定位正确，因此语义命中增加，但参考末端均值几乎不变。这也说明不能因为没有参考路径而删除该语义评价样本。

两次输入均按同图三目标进行48个有向语言替换诊断：换用新指令后，严格语义分别79.17%/83.33%；保留旧指令预测再按新目标验收仅2.08%。这说明传统定位使用了语言对应的颜色信息。该诊断复用同图独立请求的实际预测，没有合并候选池。

旧版本每图总连通区域数为1–226，新版本1–42；最优区域可能是非目标物体的小块颜色。两版本共同失败的4条为teal、silver、gray、silver；teal输出位于远处（目标误差2.693m），银灰色输出误差0.58–0.71m。旧版本另有cyan错误2.974m。失败不能被高命中率掩盖，也没有用DEV目标范围过滤这些点。当前CPU方法的语义优势与大误差尾部同时成立。

## 可复现证据与成本

固定release：`7474782f903ed104fd590b6bc3181da1e3675175`；实际脚本SHA256：`fdeafdbc22f2e0351477e11ea597400d3037d5571369c150a1c70879a1f0a565`。`--self-test`实际通过两个颜色指令、未知拒绝、负焦距相机与拒绝分母检查。

服务器run：`runs/observation_prototype_v1/{old32,new32}`，均exit0，GPU隐藏、BLAS/OMP等CPU线程1。旧版本fit1.299秒，进程wall2.01秒/user1.64秒/system0.25秒；新版本fit1.836秒，wall2.81秒/user2.08秒/system0.29秒。总实际CPU4.26秒（约0.00118 CPU小时），GPU小时0，峰值RSS均约67MiB。计时包含读取RGB-D、反投影、分割、区域统计与端点生成；不含未实现的路线规划、检查或执行。页缓存可能已热。

逐场景预测、训练原型、训练正例数量、所有manifest/图像/当前深度/参考文件SHA见 `reports/observation_prototype_v1/{old32,new32}.json`。run PID、source/launcher SHA、实际命令、状态、退出码和资源记录同步在 `reports/observation_prototype_jobs/`。

复现（输出需新路径，脚本拒绝覆盖原结果）：

```bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/home/wzy/dpvlm/route_set_v1
SRC=$ROOT/research_v2/releases/7474782f903ed104fd590b6bc3181da1e3675175
$ROOT/.venv/bin/python "$SRC/scripts/observation_prototype_grounding.py" \
  --data "$ROOT/data/observation_derived_reach32_20261002" --output /tmp/prototype_old32_reproduction.json
$ROOT/.venv/bin/python "$SRC/scripts/observation_prototype_grounding.py" \
  --data "$ROOT/data/observation_learning_curve_new32_v1" --output /tmp/prototype_new32_reproduction.json
```

当前决定：保留为认真比较的传统观测定位基线，不能把颜色任务中普通空间attention的收益当成方法核心。其与Qwen集合头共享训练终点监督来源，但结构、骨干、候选数量和输出能力不同；只能直接比较定位指标及成本。后续完整路线方案必须进一步证明几何有效性、多路线覆盖和更复杂语言任务的价值。
