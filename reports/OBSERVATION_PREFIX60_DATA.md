# 六任务 prefix60：固定 TRAIN 扩展

按 `configs/multitask_prefix60_registration_v1.json` 的事前导出名单，固定各任务 TRAIN 父索引 0–7，共 48 父；DEV 仍为原索引 16/17，共 12 父。没有扩大或重新定义 DEV，没有按采集成功情况替换父场景。该注册发生在扩展模型使用之前，原始采集此前已在运行。

实际导出固定于 `2f7b3e9b06854f18e06b9d4bee406c08f4183d4f`。服务器 CPU1 先通过 10/10 新旧快照测试（1.30 秒，包含 Linux 符号链接拒绝），随后脚本确认全部 60 个请求父场景封口，机械门禁前后两次通过，才读取所选 TRAIN/DEV 当前观测与轨迹并原子导出。两项 `record_job` 均 exit 0。

实际数据：60 父、180 次提案、180 条成功正参考、0 setup failure、0 零参考父；TRAIN 192 条语言输入、DEV 48 条，共 240 条。每父的语言改写共享三条参考，不能将 240×3 称为独立轨迹。路线类型和生成语义/碰撞/执行标签仍未提供。

- 数据：`/home/wzy/dpvlm/route_set_v1/data/observation_multitask_prefix60_v1`
- snapshot manifest SHA256：`2423d870e9a07333edccb16cb2af179beb7c74c4a943d4393561bc2c29ae1d59`
- observations SHA256：`844478ff8f3e18b520772b86bf3f1914c46383c9b6eeaaddbfd80a5333bc57fc`
- supervision SHA256：`2b6897f8e91236df98eb5af2b15142502b3cab78690c3bb75938c9bac2aa04a9`

完整请求名单、源文件哈希、采集记录与实际 job 在 `reports/observation_multitask_prefix60_v1/`。SCORE、CALIBRATION、TEST_LOCKED 的原始图像、轨迹、目标和任务验收内容没有用于本次导出；跨角色重复检查只读取已保存机械元数据。

下一组配置 `configs/observed_multitask_prefix60_v1.json` 保持相同普通自由终点头、seed0、1500×32、K4 和每 250 步宏平均参考 ADE 选模，再做同数据检索对照。训练父场景从 12 增至 48，训练指令曝光仍为 48,000、完整候选状态仍为 192,000。编码缓存和实际总体耗时单独记录；当前仅准备了不可变启动脚本，尚未将计划结果写成已完成实验。
