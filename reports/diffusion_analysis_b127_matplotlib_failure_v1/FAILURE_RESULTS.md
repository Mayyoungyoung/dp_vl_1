# b127 保存池分析环境失败

实际源码b127ce8514a63917cc3f1b7278db8f65289ba985的保存池分析因私有`.venv-qwen`未安装matplotlib，在`plots()`阶段抛出`ModuleNotFoundError`，退出1。源码、十个完成阶段及原池校验已在此前执行；这是运行环境失败，没有新的生成、训练、选模或质量结果。

record_job PID760258/child760263，2026-10-03T04:13:10.610071+00:00→2026-10-03T04:13:23.700527+00:00，外层13.090456秒。recipe主体13.018525419秒与外层嵌套；CPU0、CUDA隐藏、GPU小时0。原日志/状态/recipe/registry及空analysis目录存在性已只读归档；原服务器目录保留不覆盖。

后续只允许在根线程单独调度时，用已存在且matplotlib可导入的`.venv` Python3.8执行同一b127科学源码，输出fresh `runs/observed_two_row_diffusion_analysis_v2/analysis`。不安装依赖，不修改指标，不删失败、不重选checkpoint。新runner是独立工程recipe，并非原32测试已覆盖的新源码。
