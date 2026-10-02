# 六配置静态比较：根接口修复后的原生崩溃

唯一重跑固定于 `e1221123b0ddde4fc70ac3d731568c40e636d0c7`，使用原 `observed_two_row_static_contacts_v1.json` 配置和同六个保存配置。没有改变姿态、几何、输入哈希或任何恢复门槛；输出为全新 `observed_two_row_static_contacts_root_api_fix1`，首次 1915 失败完整保留。

40 项服务器测试通过（0.31 s），随后诊断 child PID 418828 于 2026-10-02 16:04:32.827755–16:04:44.543212 UTC 运行 11.715457 s，发生 signal11，record_job exit1。六个配置尚未应用、全部未尝试，没有碰撞矩阵或新路线。原生崩溃绕过 Python finally，故没有采集器 summary/末尾 guard counts；[独立事后记录](observed_two_row_static_contacts_root_api_fix1/postmortem.json)明确不是伪造的运行期 summary。

初始化父节点问题已通过。已落盘新静态共同 world/RGB-D、原模型来源以及[旧共同状态与新静态状态的差异](observed_two_row_static_contacts_root_api_fix1/metadata/old_common_comparison.json)，尚无 mesh provenance 文件或配置 ledger。固定执行顺序在保存这些文件后立即调用 `mesh_provenance`，然后才进入配置循环。原生栈为 `simReleaseBuffer_internal / cfree`。

实际 pinned `pyrep/backend/sim.py` SHA 为 `25c6da95f975c9348a83af310df28af0bd7aed317bcd625bdcc4f7181e53765c`。[官方固定源码](https://github.com/stepjam/PyRep/blob/8f420be8064b1970aae18a9cfbc978dfb15747ef/pyrep/backend/sim.py)的 `simGetShapeMesh` 把 `outNormals` 指针覆盖成从 `outIndices` 读取的 Python list，按三倍 indices 长度越界读取，再将 `outNormals[0]` 这个整数当作 native 指针释放。这一具体缺陷与已保存阶段及崩溃栈一致；本轮没有 Python fault-frame/shape 级插桩，不声称已确定出错 shape。

最小修复移除 runtime mesh API 调用，不改共享 PyRep/CoppeliaSim。保留实际加载 `.ttt/.ttm` 对固定 archive 的文件哈希、每 shape 名称/handle/bbox/collidable flag 和原注册程序柱几何。所有输出明确 runtime vertex/index/normal hash 不可用；文件哈希只证明模型文件来源，不能证明旧动态查询瞬间或运行时 mesh 完全相同。新增 mock API 检查将 `get_mesh_data` 设为调用即失败，证明安全 provenance 路径不会触碰此接口；本地合并 41 passed / 0.81 s。

- [原生日志](observed_two_row_static_contacts_root_api_fix1/run/static_contacts.log)
- [作业命令、源码与退出状态](observed_two_row_static_contacts_root_api_fix1/run/static_contacts.status.json)
- [原始 manifest](observed_two_row_static_contacts_root_api_fix1/metadata/manifest.json)
- [逐文件事后索引](observed_two_row_static_contacts_root_api_fix1/artifact_index.json)

传输 archive SHA `130492f41270471243fa8a808f3fb8b95f918f1972d821ee29c7e3616504c939` 双端一致。不可变 wrapper SHA `2f85b06d09210a9e68f73a2d69252e6447fd07876bd51567d9126a971999c602`，bash -n 通过且使用 LF。NPZ 保留服务器和本地忽略归档，普通 Git 仅存元数据/图像/日志。

当前仍不能用本次故障提出移动柱体的因果结论。只在新的固定源码与独立输出中完成同六个静态配置后，才依据实际 body 命中提出一个新布局假设；后续布局有效性仍须真实动态采集验证。没有自行重启。
