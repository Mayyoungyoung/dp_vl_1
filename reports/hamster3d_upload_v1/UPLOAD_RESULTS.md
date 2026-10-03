# 3D HAMSTER 固定资产上传实际完成

冻结上传源码 `062e8767d09214df6bd5a21871928a1469f3f0a2`，run `20261003T033557Z_48300`，本地PID48300实际完成exit0。固定官方模型revision `ddc5987a56cdcb14e5e2297817612532e46e912b` 的18文件全部经一次SCP发出、staging校验及不覆盖发布，并在服务器执行最终18项size/SHA复核；18/18通过。总字节18,295,875,781，其中4份safetensors权重。完整manifest SHA `53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032`，实际运行脚本SHA `6673c2adc9745c2bef37886d7d919ec84a7e981e134869e70e458488aaa27479`。

本地58个会话原件含原status、18条SCP日志、39次remote动作receipt均保留；最后release实际成功，服务器release原status也只读下载/核对。18个最终服务器文件位置与hash在 `MODEL_REMOTE_INDEX.json`。本次归档没有拉取权重或重新hash约18GB模型，只验证封存收据/manifest/source一致性。

上传内部记录：`2026-10-03T03:35:57.763083+00:00` → `2026-10-03T04:06:40.491230+00:00`，body 1842.703000秒。root在全部本地预哈希之前另记外层起点 `2026-10-03T03:34:51.914572+00:00`；到上述内部结束时刻的已记录窗口为1908.576658秒（约31.810分钟），含预哈希/启动前段65.848511秒。**这不是完整进程退出耗时**：内部end写在release之前，末次release调用0.734000秒及最后写收据/退出开销未包含，实际退出UTC未封存，不补造。body、SCP、remotehash均为嵌套组件，不能重复相加；此前下载成本只引用原 `reports/hamster3d_local_download_v1/DOWNLOAD_RESULTS.md`，不再次计入上传。

SCP进程wall之和1647.327000秒，最终18文件服务器复核主体89.442458秒。远端Python校验/发布均taskset CPU0，不能把SSH/SFTP传输本身冒称全进程单核；GPU小时0、模型调用0。本轮未执行assets复核阶段、环境安装或TRAIN6探针，上传完成不能称机器人系统已复现或生成质量有效。

源文件与协议/测试副本保持实际字节，`.py`附`.txt`后缀；准备时的`actual_upload_started=false`与draft记录保留为历史收据，不能覆盖本节的实际完成结论。所有失败历史下载/服务器准备记录继续保留在原family，未被本轮成功替换。
