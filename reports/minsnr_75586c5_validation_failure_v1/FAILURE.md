# Min-SNR 实际CPU测试失败，真实训练未启动

固定75586c573d8d67b3a3a0d33ee2a027538a1bacad在原.venv/CPU0/CUDA隐藏进行真实16测试：14通过，2失败，0skip。失败为两个arm的synthetic连续4步与2+2恢复后的完整state digest不等。pytest13.65秒、child15.670754秒、runner15.744744秒嵌套，外层时间见原status；0真实数据/PT inspection、0正式训练或GPU。五个Torch测试实际执行，不能用本地5skip代替通过。

原日志/精确JUnit/源身份/状态全部保持。Root未绕过检查，未触发inspect-parent或任一真实arm。下一步先定位具体模型/optimizer/RNG/stream字段差异；任何修复都使用新source和fresh验证输出，失败目录不覆盖。当前未断言科学loss或恢复实现有错，也未将全部state差异当浮点容差问题。
