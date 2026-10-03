# 3D HAMSTER 固定资产上传草稿 v1

状态：代码与纯文件测试，尚未上传。小文件阶段已于 02:56:57 UTC 完成 14/14 校验；权重阶段由根线程另行启动，不能提前当作完成。本入口要求本地全部 18 个最终文件通过原 manifest 的 size/SHA 后，才进行任何 SSH。

`scripts/upload_hamster3d_assets.py` 只调用已授权 `ssh/scp wzy3090`，不下载、不加载模型、不安装环境、不修改旧 assets source/manifest/identity，不自动触发服务器 assets --resume。根线程冻结 helper 后才可运行。

固定 manifest 使用服务器原件 SHA `53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032`；远端每次检查也重新读取 `1b0348ef48393a2d98113956575e99388c40d4a6` release 的同字节 manifest。原模型目录不变：`/home/wzy/dpvlm/route_set_v1/data/3d_hamster_ddc5987a56cdcb14e5e2297817612532e46e912b`。

流程为逐文件 `inspect → 必要时一次 SCP 到 .upload_staging/<filename>.part → SHA/size 验证 → 原子发布`。远端已有正确完整文件直接复用；错误完整文件立即保留并停止。已经完整正确的 staging 可以直接核验发布；不完整/错误 staging 保留并停止，不自动覆盖或重传。SCP 不提供本方案内的 partial Range 续传，失败后的 staging 由根线程核验后另行决定处理。

发布使用同一文件系统的 `os.link(staging,target)`，目标不存在时原子出现，目标竞争出现时内核拒绝覆盖，并重新核验目标；仅成功发布后移除同属本次 staging 的额外链接。任何错误完整目标都不会被 `replace` 覆盖。最后再读取并核验全部 18 个远端最终文件，保存最终 size/SHA 表。

并发由本地上传锁与远端 `runs/hamster3d_upload_v1/upload.lock` 控制；旧下载仍持 preparation.lock 时拒绝上传。远端 Python hash/校验/发布明确 taskset CPU0；SCP 为单一、无压缩的 OpenSSH 文件传输通道，其服务端 SSH/SFTP 进程由 SSH 守护程序调度，**不宣称整个网络栈或 SFTP subsystem 已强制 affinity0**。根线程应在资源排程允许时启动。

本地逐文件 SCP 最多一次、3600 s 总墙钟；超时/中断只终止本入口创建的当前 SCP 子进程，保留 remote .part。没有自动网络重试。远端校验调用最多 600 s/次。每次 issued、PID、command、exit、elapsed、safe JSON 响应和 SCP 日志保留；不使用 SSH verbose、不打印 token/环境。remote stdout 只返回受控 JSON，源文件通过 stdin 发送而不做 shell 字符串插值。

上传目录内完整文件+staging 上限 22 GiB，并保留文件系统 1 GiB 空余；模型 17.04 GiB，串行 staging 不制造第二份完整 17 GiB 副本。源码、模型调用、GPU 与新数据读取均为 0。

根线程应从新 commit 导出 immutable helper 到 `.bootstrap/upload_hamster3d_assets_<COMMIT>.py`，核 SHA 后单独运行。程序从自身位置定位本地仓库；远端为固定路径，不提供任意上传目的地选项。失败重新运行会核验现有完整文件，但不会盲目清理 staging 或遗留锁。

本地日志目录：`F:/dpvlm/.bootstrap/hamster3d_upload_v1`；远端仅新增 `runs/hamster3d_upload_v1` 审计记录和模型目录内 staging。任何后续环境/推理门槛仍由根线程单独启动。

离线验证：Windows 本地纯文件/模拟调用测试实际 12 passed / 0 skipped / 0.34 s，py_compile exit 0。覆盖完整文件复用、错误/不完整文件保留、竞争目标不覆盖、锁、固定 manifest、POSIX 路径生成和 SSH 超时安全记录。未实际调用 SSH/SCP；不能由这些测试宣称已经上传。
