# 固定源码1417cdc的实际服务器验证

2026-10-02 21:13:18.302–21:13:24.153 UTC，`wzy3090` 实际执行 **98 passed in 5.18s，无skip**，record_job退出0；外层墙钟5.851秒。PID567893，子PID567895。此次包括真实Torch前缀复现/恢复、独立12000普通收敛控制、原6000回归，以及传统空间路径惩罚A*和既有数据/评价接口测试。

- 不可变commit：`1417cdc802674a00e70d4ced3181c6a6ede2a99e`。父线程已核验的code-only部署tar SHA256：`32d9a5d6dec52393520ada81fd0badae319da23d85934177eed2b0d742543f10`。
- 原始 `targeted_tests.status.json`、`targeted_tests.log`、`registry.jsonl` 原样复制。测试命令、cwd、commit和进程记录以原状态文件为准。
- `.bootstrap/two_row_bound_controls_tests_v1.sh` 与服务器incoming wrapper逐字节相同，均存于 `source/`。wrapper先对两个新launcher执行 `bash -n`，成功才执行记录中的pytest；没有另造未记录的bash日志。
- 固定release的scripts、routeset、configs、tests共322文件的大小/SHA清单保存在 `source_index.json`。另外复制实际12000driver、配置、launcher、空间A* launcher和六个测试来源。测试副本仅改为 `.py.source.txt` 后缀，原字节不变，避免被默认pytest重复收集。
- 16项复制/生成来源映射均检查字节数和SHA256，见 `artifact_index.json`；该索引SHA256为 `8e73bd78b2a19c052e5b956afc90aafc6ea4a6b303f6bf37fdf7d2e510992ba7`。

此报告只证明指定源码测试通过，不预写12000真实复现门禁或最终训练结果，也不预写空间A*实测结果。归档只读已完成测试文件和源码；没有启动研究作业，没有读取任何locked原始数据。
