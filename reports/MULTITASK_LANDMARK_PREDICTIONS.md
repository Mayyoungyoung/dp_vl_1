# 六任务全部开发父场景的配对预测

源 `e1221123b0ddde4fc70ac3d731568c40e636d0c7` 实际从保存的 seed0 best 预测绘图，普通750步、辅助1500步，未增加模型调用或候选。每个方法包含全部12个DEV父，每父固定排序第一条语言；两方法同父3D坐标范围包含双方全部K4路径和所有参考。灰线是已采参考，彩线是预测，叉号为终点。该DEV已反复使用，图像不证明碰撞或机器人执行有效。

16个PNG/PDF的SHA逐项通过，8张PNG均逐页检查，无裁掉失败案例。两作业实际exit0，普通绘图9.288s、辅助8.783s（record_job墙钟，包括进程启动），CPU1。

|父场景页|普通集合回归|事件位置辅助|
|---|---|---|
|lift两父、cup第一父|[PNG](multitask_landmark_figures_v1/ordinary/tasks_1.png) / [PDF](multitask_landmark_figures_v1/ordinary/tasks_1.pdf)|[PNG](multitask_landmark_figures_v1/auxiliary/tasks_1.png) / [PDF](multitask_landmark_figures_v1/auxiliary/tasks_1.pdf)|
|cup第二父、button两父|[PNG](multitask_landmark_figures_v1/ordinary/tasks_2.png) / [PDF](multitask_landmark_figures_v1/ordinary/tasks_2.pdf)|[PNG](multitask_landmark_figures_v1/auxiliary/tasks_2.png) / [PDF](multitask_landmark_figures_v1/auxiliary/tasks_2.pdf)|
|reach两父、slide第一父|[PNG](multitask_landmark_figures_v1/ordinary/tasks_3.png) / [PDF](multitask_landmark_figures_v1/ordinary/tasks_3.pdf)|[PNG](multitask_landmark_figures_v1/auxiliary/tasks_3.png) / [PDF](multitask_landmark_figures_v1/auxiliary/tasks_3.pdf)|
|slide第二父、lid两父|[PNG](multitask_landmark_figures_v1/ordinary/tasks_4.png) / [PDF](multitask_landmark_figures_v1/ordinary/tasks_4.pdf)|[PNG](multitask_landmark_figures_v1/auxiliary/tasks_4.png) / [PDF](multitask_landmark_figures_v1/auxiliary/tasks_4.pdf)|

观察：两方法多条路线常集中在相似区域而一起偏离真实目标；lift/cup仍存在较大接触位置偏移，reach/slide末端明显偏离。辅助并未使所有场景成功。量化结论使用三种子全48指令指标，而非这12条固定语言的视觉印象；首次关闭位置分析也与路线ADE分开，详见 `OBSERVATION_MULTITASK_LANDMARK_THREE_SEED.md` 与 `MULTITASK_EVENT_LOCATION_DIAGNOSTIC.md`。

完整来源、选图规则、权重和预测SHA、实际命令、渲染校验见 `multitask_landmark_figures_v1/{ordinary,auxiliary}/index.json`、`local_verification.json` 和两份 status.json。复现应在相同不可变release运行 status 中命令，换新的输出目录，禁止覆盖这组证据。
