# 去 direct 分支普通对照：实际服务器验证

固定源码 `1a3eef1fb12d55e98d4d188a091ea40ea62c0a02`；记录作业 `targeted_tests` 在 wzy3090 实际完成 exit0。PID 628230/child 628232，2026-10-02T23:28:58.397779+00:00 至 2026-10-02T23:29:21.280989+00:00，外层 22.883210s；pytest 22.07s。

118 tests 全通过、0 skip、0 failure。新入口15项中5项实际使用 Torch，覆盖生产尺寸共有初始化/RNG/参数数量、原forward/标签边界、剩余模块梯度及更新、scoped入口恢复，以及未修改真实训练loop的连续/中断恢复完整状态等价和实际采样链。原constant/convergence/two-row与新旧TRAIN审计回归一并执行。

GPU隐藏、单CPU亲和由实际wrapper断言；GPU小时0。测试包含合成fixture更新，不是实际研究训练或机器人预测。真实 no-direct12000 另由root启动；此归档不声称其完成或产生收益。

原始log/status/registry/JUnit、实际wrapper与相关冻结源码共34文件逐字节远端SHA核验，见 `REMOTE_ARTIFACT_INDEX.json`。未重跑测试、未产生额外真实数据forward、未读取封存新DEV或旧reserved raw。
