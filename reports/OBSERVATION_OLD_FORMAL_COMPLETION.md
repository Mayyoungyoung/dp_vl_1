# 旧六任务正式采集：机械完成核验

原 `6c4446921317fab43066d4928e7a5cdcbbf340d1` 作业于 2026-10-02 17:38:07 UTC 正常退出，exit0；开始于11:53:37UTC，墙钟5小时44分30秒。数据仍在服务器原目录 `data/observation_multitask_resume_probe_v1`，没有重新生成或覆盖。

17:50:11UTC只读核验：全部144个预登记父场景有闭合标记，所有worker lock消失。TRAIN96、DEV_MODEL12、DEV_SCORE12、CALIBRATION12、TEST_LOCKED12身份保持原登记。闭合数不是成功数，不由此推断锁定角色的数据质量。

本检查仅读取作业状态/时间/退出码、原partition注册表，以及closed.json和worker.lock的存在性。没有读取closed内容、轨迹、图像、标签、锁定结果或模型指标。[机械结果](observation_multitask_old_formal_completion_v1/mechanical_completion.json)和[实际审计脚本](observation_multitask_old_formal_completion_v1/audit_source.py)已保存。无需恢复或重复启动已完成作业。

这批数据与较早完成的84ee新批次是不同采集。只有原注册12个DEV_MODEL父经过单独来源、布局指纹、历史使用检查，进入固定模型迁移的准备；该资格不授权把其余角色合并训练或打开锁定测试。
