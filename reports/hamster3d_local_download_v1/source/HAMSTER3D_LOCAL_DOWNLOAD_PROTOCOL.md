# 3D HAMSTER 本地固定资产下载 v1

本入口只准备公开文件，不安装环境、不访问任务数据、不加载模型、不调用 GPU、不自动 SCP。服务器旧 assets 运行 `1b0348e` 已失败；原日志、错误标签与完整来源保留在 `hamster3d_preparation_validation_v1`。本方案不改旧服务器运行源、旧 manifest 或旧 preparation identity。

## 固定身份与目录

严格读取归档的服务器原 manifest：`reports/hamster3d_preparation_validation_v1/source/configs/hamster3d_assets_v1.json.source.txt`，SHA256 `53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032`。其 CRLF 字节与某些本地 LF checkout 不同；不把重新序列化的 JSON 当作相同字节原件。

仅 18 个模型文件，revision `ddc5987a56cdcb14e5e2297817612532e46e912b`，总量 18,295,875,781 B。服务器已经完成的 50 个代码文件不在此下载入口中。所有目标逐 size+SHA256 核验；LFS 与非 LFS 的期望值都来自同一固定 manifest。

- 文件目的地：`F:/dpvlm/.bootstrap/hamster3d_model_ddc5987a`。
- 日志/锁/状态：`F:/dpvlm/.bootstrap/hamster3d_local_preparation_v1`。
- owned 总量上限：22 GiB；每次调用前必须保留至少 1 GiB 文件系统空余，加上本次仍需接收的全部剩余字节。
- 不依赖 HF 客户端、不安装依赖；只有 Python 标准库、串行一个网络子进程。脚本不读取 HF token/.netrc，不向模型站点加入 Authorization，不打印环境或异常中的签名 URL。

根线程已经只读核验本地旧小文件缓存中的 13 个文件可复用；将由根线程单独保存原→目标 copy/hash receipt，再启动本入口。入口会逐 SHA 复用这 13 个完整文件，仅缺失的 `tokenizer.json` 发起传输，不另加缓存发现逻辑。

## 两个独立阶段

1. `small`：14 个非 safetensors 文件，包括 tokenizer。每文件 300 s 网络与 partial hash 校验窗口。
2. `weights`：4 个 safetensors shard。要求同 source/manifest 的 small 完成标记，重新核验 14 个小文件，再处理权重；每文件 7200 s 窗口。

不会从 small 自动进入 weights。每文件最多 3 次网络 worker 尝试；一个 transfer invocation 可能跟随 HTTPS 重定向，不能把 invocation 数解释为所有 HTTP 请求数。每次 socket timeout 30 s，同时父进程按剩余文件墙钟上限终止它自己创建的唯一网络 worker，防止多地址连接/重定向拖长总等待。已完整正确文件的本地哈希复用不发网络请求；磁盘哈希时间全部计入实际耗时。

失败时停止整个阶段，保留已经完成文件和 partial，不自动去下一文件掩盖失败。再次运行必须 `--resume`，生成新 run_id/session；source SHA、manifest SHA 和 destination 必须与旧 identity 完全相同。死进程遗留 lock 不自动删除，需根线程核实旧 PID/worker 后处置。

## 续传与完整性

每文件对应 `.part` 和固定身份 sidecar。未知身份、超长 partial、错误完整文件均保留并拒绝；正确完整文件从不覆盖。

已有 offset 时仅接受 HTTP 206，并严格要求 `Content-Range == bytes offset-(total-1)/total`；若有 Content-Length，则必须等于剩余字节。HTTP 200 绝不 append 到 partial，直接保存失败。请求 `Accept-Encoding: identity`，拒绝压缩传输，防止字节区间语义变化。短 body 留作后续 Range 续传；完整 partial 必须通过全部 SHA，才原子重命名为最终文件。SHA 错误不自动重下覆盖。

worker 只返回安全类别、状态码、offset/字节、耗时与 SHA。各文件先 durable 记录 issued，再启动 worker；worker PID、退出码与其 receipt 保留。只有全部文件完成且磁盘预算通过时才写 stage.completed。模型调用和 SCP 调用均固定为 0。

## 服务器误分类的独立最小修复

`prepare_hamster3d_external.py` 新版本只将显式路径/size/hash 校验改为自有 `IntegrityError`，并保存安全的 `exception_type/cause_type`。测试模拟 HF `LocalEntryNotFoundError(ValueError)`，证明其进入有界网络重试分支，不再伪报 SHA 错误。**这不是服务器连接修复**，本轮不将新源码替换到旧 assets run。原服务器 source 可在根线程逐 SHA 上传完成文件后执行同 source `assets --resume`，完整文件只校验复用。

## 冻结后根线程命令

根线程将 Git 中固定 commit 的 helper 原字节保存为 `.bootstrap/download_hamster3d_local_<COMMIT>.py`。脚本以自身文件定位仓库，并且 worker 继续调用同一 `__file__`；不依赖可继续编辑的源文件。

```powershell
& F:/ProgramData/anaconda3/python.exe F:/dpvlm/.bootstrap/download_hamster3d_local_<COMMIT>.py --stage small
# 读 small receipt 后另行运行，绝不自动衔接：
& F:/ProgramData/anaconda3/python.exe F:/dpvlm/.bootstrap/download_hamster3d_local_<COMMIT>.py --stage weights
# 失败恢复仍用同固定 helper 和 manifest：
& F:/ProgramData/anaconda3/python.exe F:/dpvlm/.bootstrap/download_hamster3d_local_<COMMIT>.py --stage weights --resume
```

本地纯测试覆盖 HTTP 200 禁止追加、206 offset/total/length/encoding 检查、短 body 续传、完整文件复用、坏 partial 不提升、最多 3 次尝试、owned/free 预算、墙钟终止自己的 worker、阶段门禁与日志去敏。此处没有将 fake HTTP 测试写成真实网络下载成功。
