# Extension288 固定 TRAIN 前缀全槽质量审计

本入口独立于已经运行的 collector，不修改其源、配置或验收规则。`scripts/analyze_two_row_extension_train.py` 仅允许完整注册 TRAIN 前缀 32/64/128/256，分别对应 96/192/384/768 请求条件、864/1728/3456/6912 路线槽；不能传任意父子集、成功父列表、旧16前缀或新DEV32。

机械模式随时可记录已闭合父的预算上下界、缺失初始输入和仍未闭合父的完整请求预算。未闭合父的槽不被提前当作最终失败或未尝试。此模式只读注册、manifest、closure 与机械 receipt/ledger 的 hash，不读取 raw，也不授权训练或开启新 DEV。

全槽模式先完成全部选中 closure/role/identity/hash 和全局机械重复门禁，之后才读取这一个 TRAIN 前缀的 raw。任一前缀父尚未闭合则在创建输出前拒绝。不按结果选父、替换失败、筛除 unknown、删长弧或根据 observed 输入数重定请求分母。缺图与零参考仍列入全部父、三个条件和每条件九槽。所有实际输入保持同父同图不同语言原字节。

复用原 `analyze_two_row_formal_train.py` 的受限 TRAIN 路径读取器、逐槽账本解释、原 raw/H24 复核、实际穿越分类和全九槽图。无新 inference、IK、模拟器或路线规划。每个父三个目标全部画图，已失败或 partial 路径保持失败，即便重算 tip 代理通过也不升级。unknown 是已验证正参考但不虚构为新已知类型；参考类型/条数不是总解数。

报告保留非互斥失败原因、端点/raw tip/H24 与类型变化、完整调用/耗时、缺失观察数、已知 R 直方图、已知 over/重复和 unknown、逐槽与逐条件记录、所有所读 TRAIN 文件 SHA。全局机械 gate 可检查封存新 DEV 的 mechanical closure/hash；禁止打开其图像、指令、路径和 outcome。每次输出必须 fresh，并保存新旧 helper 的实际 source SHA。

实际执行待 root 冻结新 source 后再授权，CPU1/GPU隐藏、record_job；不在等待前缀期间启动隐性轮询或自动接续采集/训练。CLI 全槽模式固定输出所有图，`plots=False` 仅为纯测试内部调用。示例：

```bash
python -m scripts.analyze_two_row_extension_train --corpus DATA --train-parents 32 --output FRESH_ANALYSIS
python -m scripts.analyze_two_row_extension_train --corpus DATA --train-parents 32 --output FRESH_MECHANICAL.json --mechanical-only
```

本地新16项与历史12项合跑：27 passed、1 skipped，19.62s；唯一 skip 为 Windows 无符号链接权限的历史逃逸检查，Linux 冻结后的实测应覆盖。没有读取实际新 TRAIN raw 或任何封存 raw。此时只准备分析能力，不预写采集效果。
