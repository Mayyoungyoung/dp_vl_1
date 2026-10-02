# 两排 prefix44：服务器验证与真实准备结果

固定源码 `9e0094aff152d48cd34eadd24e554a05f0cf4a0f` 的CPU准备已于2026-10-02 19:45:28 UTC完成。预登记32TRAIN＋原12DEV全部闭合后导出；没有替换失败父。服务器140项相关测试实际通过，17.16秒、exit0；[验证日志及状态索引](observed_two_row_scaling_validation_v1/artifact_index.json)保留原命令、PID和源码。准备阶段CPU affinity=[0]、线程1、GPU隐藏。

| 分区 | 请求/有观测父 | 请求/实际图文输入 | 请求/完成路线槽 | 已接受正参考 |
|---|---:|---:|---:|---:|
| TRAIN | 32 / 31 | 96 / 93 | 864 / 837 | 556 |
| 原DEV_MODEL | 12 / 12 | 36 / 36 | 324 / 324 | 205 |
| 总计 | 44 / 43 | 132 / 129 | 1188 / 1161 | 761 |

缺失父 `two_row_reach_283220` 的[原始机械closure](observed_two_row_prefix44_preparation_v1/missing_parent_283220/closure.json)明确记录：27个请求槽，started=completed=0，attempted上下界均0，unattempted上下界均27；未保存初始观测，因此三个请求输入不伪造。实际几何1mm指纹与登记相同，但 `model_eligible=false`。worker的`completed/exit0`表示这次门禁流程已结束，不能解释成路线采集成功。机械记录不足以区分具体哪项初态验收未通过，本次不为推断原因额外读取或重跑该父。原18.287秒worker成本保留。

TRAIN审计覆盖全部93个实际输入与556条正参考，没有零参考的已记录输入。终点在stride2真实观测点逐坐标±5cm范围内的容量检查为556/556，最大最近点L∞距离1.784cm；模型event-aware H24的原两排TipValid为556/556，事件转换数变化0/556。出口保留221条unknown类型正参考（39.75%），并不把unknown判为无效或删除。

| 全部TRAIN正参考长度 | raw | 模型H24 |
|---|---:|---:|
| 均值 / 中位数 | 1.532 / 1.363m | 1.466 / 1.321m |
| 95分位 / 最大值 | 2.971 / 6.482m | 2.816 / 6.036m |
| 超过2m / 超过3m | 94 / 28条 | 77 / 24条 |
| 最高世界z | 1.832m | 1.828m |

长弧、unknown和失败分母全部保留；未截断、筛选或改变2cm tip/3cm目标标准。通过审计只说明本批正参考在既定表示与tip代理下可用，不证明学习泛化或完整机器人执行。导出record进程2.878秒，TRAIN审计进程3.619秒（审计内部2.059秒）。

`fixed_dev_identity.json`按原36个DEV ID核对观察、语言和监督行完全相同，并核对其输入、验证标签、全部正参考及导出源SHA。本文只归档这一已有receipt和准备元数据，没有重新打开DEV图像/路线数组来作方法选择，也没有读取锁定raw。GPU重编码数组一致性、初始化与训练结果属于之后的独立阶段，本文不提前记为完成。

完整证据见[准备索引](observed_two_row_prefix44_preparation_v1/artifact_index.json)及[TRAIN汇总](observed_two_row_prefix44_preparation_v1/quality_summary.json)。18个服务器小文件在复制前后双读SHA并与本地逐字节校验一致；传输归档1,167,360字节，SHA256 `c1d13915c70aba63ce547a00b438be580db74bc44a124e18a3908c5d9d90d9bb`。

- export manifest SHA256：`6ed7786823275f26dba38fff9039bd33127b71567de4e4c0c80a37cfd91e46f9`
- TRAIN quality SHA256：`4e613bd8908f4bc0fd110bb96334aa4d89c65047b6453af2956e7bc9382f2548`
- 固定DEV身份receipt SHA256：`c89803af22f58e862cb9f477a46f50083934222cd16dc444aeb1a01e13904c15`

导出时机械gate为54父闭合，duplicate/blocked集合为空，仅283220初始观测不可用；这不是完整116父已完成或最终无重复的声明。准备通过后，root已另行启动普通32父训练；本次归档不启动GPU/模拟器、不改全局结果表、不提交。
