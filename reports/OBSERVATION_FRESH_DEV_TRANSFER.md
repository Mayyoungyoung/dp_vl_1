# 旧64父模型在新16父DEV上的实际泛化

预注册natural父262192–262207导出为DEV_MODEL后，固定 `7f27f93514850973fba93260e8569f2a7fec27ea` 的 `evaluate_observed_fresh_dev.py` 在CPU1加载旧natural64的soft/peak三种子原best权重，对全部48条新指令实际完成评价、exit0。原权重由旧8父DEV的参考ADE选择，此处不训练、不resume、不用新DEV重新选checkpoint。训练dataset fingerprint保持原值，新评价fingerprint另存，源checkpoint SHA前后不变。

新DEV的48条输入均有参考，语义与参考分母都是48；每条K4，192候选/seed。输入只有真实冻结Qwen的RGB＋语言特征、当前末端状态与观测RGB-D/相机；目标标签只用于原 `observation_eval_v2` 的身份＋3cm检查。DEV_SCORE、CALIBRATION和TEST_LOCKED没有进入本次导出、缓存或评价。

|原best模型|严格语义候选正确率|AnySemantic@4|参考ADE（cm）|参考末端误差（cm）|
|---|---:|---:|---:|---:|
|soft，三seed|28.47±3.18%|29.86±1.20%|6.60±0.28|12.08±0.37|
|peak，三seed|72.05±3.14%|73.61±5.24%|6.21±0.14|11.25±0.61|

均值±样本标准差（ddof=1）；三个seed使用同16个新DEV父，不是三套独立测试。逐seed语义soft `[31.25,29.17,25.00]%`，peak `[72.40,75.00,68.75]%`，三个配对语义与参考ADE均改善。新开发场景支持定位修复的收益能迁移；仍不意味着开放词汇、路线类型覆盖、完整路径碰撞有效性或机器人执行已经成立。峰值模型新DEV语义低于反复使用的旧DEV均值83.68%，不能隐藏这一差距。

## 全部失败的事后分解

`analyze_observed_fresh_dev.py` 只读取已保存预测，再读取评价标签分类，逐条重算后与原48×6语义结果完全相同。全部288个场景/模型结果、144个配对差异及每个候选的错误类型保存在 `reports/observed_natural_fresh_dev_diagnostic.json`，没有删除失败或挑选种子。

|模型|成功候选，seed0/1/2|最近目标身份正确但距离>3cm|最近目标身份错误|
|---|---|---|---|
|soft|60 / 56 / 48|85 / 102 / 101|47 / 34 / 43|
|peak|139 / 144 / 132|1 / 8 / 4|52 / 40 / 56|

每个单元的每个seed分母均192。soft→peak减少了正确目标附近的偏移错误，但最近目标身份错误反而略增，不能把收益描述为颜色识别能力普遍改善。此为端点几何分解，不据此推断未查看的图像原因。

三个peak种子都没有任何候选通过的指令共4条，全部列出：

|输入|指令目标|三seed目标中心平均距离（cm）|
|---|---|---|
|derived_reach_262192_target1|green|27.06 / 26.76 / 26.96|
|derived_reach_262194_target2|azure|63.51 / 62.61 / 63.18|
|derived_reach_262196_target0|violet|35.97 / 34.25 / 34.29|
|derived_reach_262201_target1|gray|50.48 / 50.80 / 50.01|

这些失败的全部12候选各自最近目标身份都错误。此处中心距离使用独立语义目标，不能与表中匹配参考末端误差混为同一量。

## 可复现产物与后续实际队列

六个源checkpoint的路径/hash、原训练commit、独立评价commit、输入/缓存hash、全部预测和逐场景结果见 `reports/observed_natural_fresh_dev_v1/` 与对应 `runs/observed_natural_fresh_dev_v1/`；命令/PID/退出码在 `reports/observed_natural_fresh_dev_jobs/`。正式命令为：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CODE_COMMIT=7f27f93514850973fba93260e8569f2a7fec27ea \
/home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.evaluate_observed_fresh_dev \
  --root /home/wzy/dpvlm/route_set_v1 \
  --data /home/wzy/dpvlm/route_set_v1/data/observation_natural_reserved_development_v1 \
  --reservation configs/observation_partition_reservation_v1.json \
  --output /home/wzy/dpvlm/route_set_v1/runs/observed_natural_fresh_dev_v1
```

运行目录为上述commit的immutable release。复评必须使用新的output路径，原产物不覆盖。

同一新192 TRAIN父、576监督指令的soft/peak seed0从头配对现已实际完成：3000×32、K4、lr3e-4、grounding权重0.02，原DEV参考ADE每250步选择，保存best与last3000。两臂内部训练预算均384000候选槽；相比旧64增加三倍数据也增加三倍步数，使每监督指令平均曝光仍约166.67次，不能称为旧学习曲线的等总预算点。独立结果与最后一步回落见 `OBSERVATION_RESERVED192_PAIR.md`。旧64固定传统原型在同新48指令上也已实际完成，83.33%严格端点成功，高于同数据神经头；全部8失败和单端点范围见 `OBSERVATION_PROTOTYPE_FRESH_DEV.md`。
