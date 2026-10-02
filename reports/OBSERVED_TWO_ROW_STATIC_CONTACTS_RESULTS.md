# 六配置静态碰撞诊断：初始化接口失败

固定 release `1915de7dbe2c8cbc772250e092be87c83c7df786` 的唯一实跑已完成：36 项 Linux 测试通过（0.30 s），随后模拟器在初始化层级排序时退出。状态为 `setup_failed`，六个配置均未应用，六个未尝试；新 IK、路线、simulation-start 调用均为 0。这不是几何不可行、恢复门禁失败或碰撞归因结果。不能据此提出移动柱体的物理结论。

`record_job` PID 408489 / child 408490，2026-10-02 15:41:42.895197–15:41:55.786484 UTC，实际 12.891287 s；采集器内部 12.479965 s。外层 exit 0 表示失败证据被捕获并正常落盘，诊断本身没有完成。原 24-query 标签未改，`original_instant_reproduced=false`。

异常位置为 `configure_static_world` 的父节点遍历。固定 PyRep backend `simGetObjectParent` 将根节点合法返回值 `-1` 传入 `_check_return`，因此抛出 `RuntimeError`。同一固定源码的 `Object.get_parent` 用捕获 `RuntimeError` 后返回 `None` 处理根节点。本项目拟修复为：先核验已审计 inventory 中每个句柄/type，再读取同 backend 底层函数的原父句柄；`-1` 是根，非根父节点必须在 inventory 中，拒绝环和重复句柄，不吞其它验证异常。这只修接口，不更改场景、六个冻结向量或任何门禁。

失败发生前，实际加载的 `task_design.ttt` 和 `reach_target.ttm` 均已逐字节匹配固定 RLBench archive；Arm 源 SHA 与原查询一致。尚未执行 runtime shape mesh 归档或碰撞矩阵，不追溯声称旧查询已经保存 native scene 或 mesh hash。

- [完整失败及预算](observed_two_row_static_contacts_v1/metadata/summary.json)
- [堆栈](observed_two_row_static_contacts_v1/metadata/failure.json)
- [实际模型来源](observed_two_row_static_contacts_v1/metadata/model_assets.json)
- [进程、命令与状态](observed_two_row_static_contacts_v1/run/static_contacts.status.json)
- [逐文件 SHA 与远端位置](observed_two_row_static_contacts_v1/artifact_index.json)
- [可读分析](observed_two_row_static_contacts_v1/analysis.json)

修复后的本地合并检查为 40 passed / 0.65 s，新增根、缺失父、环、重复句柄、非法句柄与类型改变边界覆盖。没有自行重试；需要新固定 release、全新输出及 CPU1 授权后，才执行同一六配置比较。下一物理布局实验应等待实际 body/link 命中证据：柱体碰撞可支持只改相应柱位的独立几何版本；桌面或机器人 body 命中则不能靠挪柱作解释。无论哪种情况，都须重新动态采集检验，不把静态配置当有效路线。
