# 两排普通模型固定末步训练拟合诊断

首轮best500的TRAIN TipValid仅24.48%，但这不能代表last1500的训练拟合；末步saturation loss已经下降到约0.000279。先做一次可证伪检查，再选择学习机制。

入口 `scripts/audit_two_row_last_train.py` 固定原last1500、全部16TRAIN父的48条件、K4/H24，在CPU执行48个新头请求/192条路径状态。使用真实冻结Qwen旧特征，不重新编码、不更新参数、不重选checkpoint、不修复或筛选预测。原development loader会读取已授权TRAIN/DEV export，但新forward只覆盖TRAIN，锁定数据不在export中。

原两排评价器重算末步TRAIN全部有效性，并保存完整预测、逐条件判定和hash。best500 TRAIN池直接读取原保存结果，不再生成。对两个固定池分别复用原saturation精确分配，核对loss数值，记录每个候选匹配到的原参考hash、逐顶点偏差、最大偏差和端点偏差。所有unknown/长弧仍可参与匹配；匹配本身是诊断，不能作为forward输入。

若last TRAIN几何仍差但匹配误差很小，优先检验回归误差对狭窄净距的影响；若last TRAIN良好而DEV差，则先判断训练覆盖与泛化，不能把best早停不足当结构失效。只有有效重复出现且占用预算，才继续考虑去重补全。原始best/last DEV结论不因该诊断改写。
