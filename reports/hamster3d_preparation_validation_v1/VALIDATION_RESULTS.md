# 3D HAMSTER 准备入口：实际验证与下载失败

固定运行源码 `1b0348ef48393a2d98113956575e99388c40d4a6`；6 文件实现来自 `94096fe`，后续提交只扩展了机械作业快照的前缀识别。此次仅准备公开模型资产，不是模型推理或路径质量结果。

最终状态：assets 于 **02:31:23.245909 UTC 自行退出 1**，实际 body 432.056268 s。代码 50/50 完整校验，模型 0/18；没有权重完成、没有环境安装或模型加载。下面早期 running 快照原字节继续保留，未改写成事后状态。

服务器实际验证：**11 passed，0 skipped，0.09 s，exit 0**。父/子 PID 707561/707562，UTC 2026-10-03 02:23:51.923777 至 02:23:52.305611。Linux 符号链接越界拒绝测试实际运行，补足先前 Windows 权限导致的跳过。

归档保留原 `tests.status.json`、`tests.log`、`registry.jsonl`、`pytest.xml`；5 个服务器代码/配置/测试原件，另一个协议文档从同一 Git commit 取原字节；实际验证 wrapper 两端逐字节相同。测试源在本归档使用 `.source.txt`，不会被 pytest 当成新测试再次收集。`SERVER_ORIGINALS_INDEX.json` 绑定远端完整路径、size 与 SHA256。

## 根线程已启动的下载

唯一 session 为 `runs/hamster3d_preparation_v1/sessions/20261003T022410Z_assets_707926`。启动 02:24:11.069160 UTC，父/子 PID 707926/707933，CPU0、GPU 隐藏。根线程启动；本分析者没有启动或重试下载、没有启动 environment、没有加载模型。

**02:30:17.521781 UTC 快照仍为 running**：50/50 官方代码文件校验通过（310,785 B），0/18 模型文件完成；模型首个 `.gitattributes` 仍在第 1 次库调用，账本尚无 failure。模型目录的 2 个文件共 1 B，仅为下载元数据；不能称权重已下载。此前 02:27:27 快照原字节另存，未覆盖。

进程当时 CPU0、child RSS 38,868 KiB、CPU约 0.1%；磁盘可用 1,307,469,684,736 B，系统可用 RAM 67,869,548,544 B。第一模型调用长时间未返回是实际进展限制，尚无可归因的网络异常类别，不能把“没有失败记录”解释成网络正常。未为追进度修改运行源码或增加下载器。

运行实际 SHA：

- 下载器：`489699a61f45763e8f1e69561f42163f6ca744a9e0f3a626b77b4ef53502ca81`。
- manifest：`53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032`。
- requirements：`8e19001051e3c9996530a6696f65936d5590444d52b0a7d0035f95efca719336`。
- 固定模型 revision：`ddc5987a56cdcb14e5e2297817612532e46e912b`。

以上使用服务器执行原字节的 SHA，不能拿先前本地 LF 文件的 SHA 替代。source/manifest 和 active status 内 identity 已逐项相等验证。

## 证据范围

`monitor_0230/SNAPSHOT.json` 是一次只读 metadata/resource 快照，不是最终 receipt。原件、快照与本报告各自进入 `ARTIFACT_INDEX.json`；后续完成或失败应另增归档版本，不能倒写本快照状态。下载成功之后仍需独立环境验收和独立 GPU 技术门槛，当前 0 模型构造、0 forward、0 项目数据输入读取。

## 失败原件与有界诊断

`failure_0231/` 新增保存最终 session 的全部 8 个小型原件，包括原日志、内/外 status、逐文件账本、manifest、身份、wrapper 和 registry；没有 receipt 或 assets.completed，两个锁均已释放，旧 PID 已退出。之前的总索引保留为 `index_history/ARTIFACT_INDEX_before_failure.json`。

原异常是 `LocalEntryNotFoundError`。服务器实际 MRO 为 `LocalEntryNotFoundError → EntryNotFoundError → HfHubHTTPError → HTTPError → RequestException → FileNotFoundError → OSError → ValueError`。旧脚本的宽泛 `except ValueError` 因而将网络/缓存缺失异常误记为 `integrity_failure_preserved`，只执行了 1 次外层库调用；**该标签不表示检测到 SHA 错误**。保留原错标账本，并在本报告明确其真实含义，不修改历史文件。

02:38 UTC 对同一个固定公开小文件做了恰好 2 次独立 HEAD 诊断，禁鉴权、禁自动重试、每子进程 16 s 上限：继承代理配置模式和禁环境直连模式分别在 10.0154 s、10.0086 s 返回 `ConnectTimeout`，均未获得 HTTP 响应。未输出签名跳转 URL、token 或环境内容。到公开 HF 端点的连接超时已被直接复现；DNS、出口路由或防火墙中哪一层造成不可达尚未证实。

固定 HF Hub 0.35.3 的 `_http.py` 内部最多 5 次重试；下载器给 HEAD 的 `etag_timeout=30`，且库可能被超时环境配置覆盖。已退出历史 PID 无法再回读环境。诊断 shell 的有效 ETag/download 默认均为 10 s、endpoint 为官方公共站点，不能据此补称旧 PID 的有效配置。代码文件累计下载/校验 48.4596 s，剩余约 383.60 s 发生在模型首个调用与收尾；结合此前 `do_sys_poll`/网络 socket、现有连接超时与库内部重试，足以解释长时间等待，不能归因为文件锁。

最小修复是只让显式资产 hash/size 校验抛专用 `IntegrityError`，网络异常保存安全 `exception_type/cause_type`。这只修复错误类别与外层重试入口，**不会修复服务器连接**；不因此盲目重启。公共资产可以在另一已授权、可连通的机器按同 manifest 获取，传回后仍逐 SHA 校验复用；是否执行由根线程决定。本诊断没有进行这类下载、传输或重试。
