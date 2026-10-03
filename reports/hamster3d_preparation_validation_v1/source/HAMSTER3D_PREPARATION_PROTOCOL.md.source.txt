# 3D HAMSTER：固定公开资产与独立环境准备 v1

状态：方案与准备代码，尚未下载服务器权重、安装环境、加载模型或产生路径。准备成功只表示资产和依赖就绪，不表示模型已经接入或完成系统复现。

## 固定来源与使用边界

- 官方代码：`DAVIAN-Robotics/3D_HAMSTER`，commit `97216a8493f46301bf569d398462b8bb21c458c5`。
- 官方模型：`DAVIAN-Robotics/3D_HAMSTER`，revision `ddc5987a56cdcb14e5e2297817612532e46e912b`；实际官方元数据为公开、无 gating。
- 模型卡及官方 README 明示 Apache-2.0；主 modeling 文件有 Apache-2.0 头，内置 LingBot-Depth 有完整 Apache-2.0 LICENSE。固定代码树没有 README 所链接的根 LICENSE：保留这一文档缺口，不补造文件。当前依据是上游明确的研究可用公开授权声明，保留上游声明和署名。
- 官方发布范围是高层路径推理；低层控制策略没有公开。本对照最多先报告观测任务层的外部预训练系统，不能称完整机器人系统复现，也不是同训练数据的方法机制对照。

一手来源：[固定代码 README](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/README.md)、[固定模型卡](https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER/blob/ddc5987a56cdcb14e5e2297817612532e46e912b/README.md)、[内置 LingBot LICENSE](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/hamster3d/lingbot_depth/LICENSE)。未运行官方模型，不写成已复现。

## 文件、目录与磁盘

完整清单是 `configs/hamster3d_assets_v1.json`：18 个模型文件、50 个推理代码/许可证文件。模型总量 18,295,875,781 B（17.039 GiB）；4 个权重 shard 共 18,279,874,332 B。LFS 文件记录官方 SHA256/size；全部非 LFS 文件已经用固定 revision 的实际小文件验证 Git blob SHA1，并记录实际 SHA256。此清单不含另一个基础 Qwen 或 LingBot 权重下载。

| 权重文件 | bytes | SHA256 |
|---|---:|---|
| model-00001-of-00004.safetensors | 4,998,056,552 | fa827a490ab99271c716b8c452699a7433e3d9590dc38a30c52c16cda42be6c9 |
| model-00002-of-00004.safetensors | 4,915,962,464 | 8e4ea77836fd0592dbb3aaa32eddad28bfa2b6dce9d477c876af39618c80d714 |
| model-00003-of-00004.safetensors | 4,915,962,496 | 1e7eb9154c22e356eb966bcf363630d30c0f40db47dadf784f23b7baae2e0d02 |
| model-00004-of-00004.safetensors | 3,449,892,820 | 304dc86467f00697aa68f3728880a64aed4128760c74f8c0f195f4360944b37d |

相对 `/home/wzy/dpvlm/route_set_v1` 的唯一新目录：

- `data/3d_hamster_ddc5987a56cdcb14e5e2297817612532e46e912b`：模型文件及其 local_dir 小型下载元数据/未完成文件。
- `external/3d_hamster_97216a8493f46301bf569d398462b8bb21c458c5`：固定推理代码，仅清单中的 50 个文件。
- `.venv-hamster3d`：全新私有 Python 环境。
- `runs/hamster3d_preparation_v1`：唯一准备身份、PID 锁、分次 session、逐文件账本、日志/状态/退出码、pip report/freeze、临时目录。

额外磁盘预算为 35 GiB：模型 17.039 GiB、私有环境估计 8 GiB、临时 shard/wheels 上界预留 6 GiB、代码日志约 0.1 GiB，其余余量。无第二份完整 HF snapshot。每个文件及环境阶段前检查已用量与剩余空间；单个 pip 操作仍是有界但非实时磁盘配额，安装后的实际用量再次审计。既有资源快照约 1.2 TiB 空闲，满足此预算；不据此扩大共享资源使用。

## 下载与恢复

下载器仅使用固定公开 repo/revision，显式 `token=False`，并设置禁用隐式 token、遥测与 Xet。逐文件 `hf_hub_download(..., local_dir=...)`，一个 worker、CPU affinity 0、GPU 隐藏。代码小文件仅从固定 GitHub raw URL 下载。不会打印环境、密码、token 或 HTTP 异常中的签名跳转 URL。

正确完整文件按 size+SHA256 复用，不覆盖。未完成 HF 下载由该版本库的 local_dir `.incomplete` 机制恢复；每个文件最多 3 次外层库调用，库内部有限网络重试可能使 HTTP 请求数大于 3，因此不把库调用数写成网络请求总数。完整文件 hash 不符立即保存并停止，不自动覆盖；代码坏 partial 同样保留。记录每个调用的 issued/success/failure、耗时和最终 hash。所有失败保留，固定版本不变。

准备身份包含下载器、manifest 和 requirements SHA；恢复必须一致。全局独占 PID 锁避免重复启动，进程崩溃遗留锁需先核实旧 PID 后由根线程处理，不自动抢锁。每次 launcher 建新 session，`assets --resume` 继续已有正确文件/partial；环境阶段单独启动。没有任何读取项目 TRAIN/DEV/锁定原始数据的代码。

## 独立环境

只读现有 Python base 为 `/home/wzy/dpvlm/route_set_v1/.observation-deps/python311/bin/python3.11`，实测 3.11.13。以其创建不含 system site packages 的新 `.venv-hamster3d`，不修改 `.venv`、`.venv-qwen`、`.venv-sim` 或 base。

`configs/hamster3d_environment_v1.txt` 固定：Torch 2.4.1+cu121、Torchvision 0.19.1+cu121、Transformers 4.57.1、Accelerate 1.10.1、HF Hub 0.35.3、Tokenizers 0.22.2、Safetensors 0.6.2、NumPy 1.26.4、Pillow 11.3.0、SciPy 1.15.3、OpenCV headless 4.10.0.84。其余公共依赖由 pip 解析并在实际 install report 中保存 wheel 来源/hash、完整 freeze 与 pip check；失败不得自动升级以上固定项。pip isolated 模式、公共 PyPI/cu121 index、无持久 wheel cache，临时文件计入准备目录。

仅通过固定代码目录 PYTHONPATH 导入官方模块，不安装 GUI/demo/merge 的全套依赖，不装 xformers/PEFT/Gradio，不改原代码。其内置 DINO 注意力已有 PyTorch SDPA fallback；将 xformers 禁用明确记作后端环境差异。环境验收仅 import/注册模型类、检查版本和 CUDA 未初始化；不构造模型、不下载额外基础模型、不 forward。

## 后续输入与 GPU 门槛（本准备任务不执行）

官方 [API](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/hamster3d/inference/api.py) 输入 RGB、对齐米制 depth、语言，无真实目标/障碍。保留官方最长边 640 resize、RGB 线性/depth 最近邻、几何分支 depth fp16 round-trip 与归一化、原指令+官方固定后缀。模型已内置 geometry encoder，不另用目标定位标签或新 depth 模型。

输出为 `[u,v,depth]`、u/v 归一化 0..1000，depth 米制。官方 demo 用 `d * inv(K) @ [u_px,v_px,1]`；该运算兼容记录的负 fx/fy。用原图宽高与实际 signed K，再用实际 camera-to-world 外参；若用 resize 图，则 K 同步缩放。不得取 abs(focal)、默认示例 K 或无坐标变换的图像翻转。由真实相机参数转世界坐标是统一表示适配，标签只在之后评价。

约 17.02 GiB 权重不能全放 8.4 GiB GPU 额度，但约 63 GiB 可用 CPU RAM 支持 CPU offload 候选。需要薄 loader 适配，官方 predictor 的 `device` 字符串同时用于输入搬运，不能直接塞 device_map 字典。视觉/完整 geometry encoder/merger 要留 GPU：官方直接调用 `geometry_encoder.encode`，其内部 `next(parameters()).device` 与 meta/offload hook 有风险。decoder 可逐块 offload；预估约 4.1 GiB 非 decoder 常驻权重、留其余空间给工作区/KV，需要真实峰值验证，不承诺一定可运行或延迟优势。

不要由 model config 的 use_cache=false 断言官方生成无 KV：Transformers 4.57.1 会载入独立 generation_config，其未写 use_cache 时默认 True。实际模型加载后记录最终值，不在准备阶段擅改。

未来技术门槛须另行固定 TRAIN 输入/调用/时间预算，在 GPU 空闲时执行。官方默认 greedy K1 与 1024 token cap；K4 stochastic 是额外适配、须显式记账。保留解析错误/空路径/全部 token 与候选、原始路径和事件分段 H24、连接与碰撞失败；不得由 GT 目标/几何修复。完整编码、CPU offload、解析、投影与评分全部计时。本准备无该类调用。

## 根线程启动命令（固定 release 后分开执行）

```bash
taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/<COMMIT>/scripts/launch_hamster3d_preparation_v1.sh assets
# 根线程读资产完整 receipt 后单独启动环境；此行不会由上一阶段自动运行。
taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/<COMMIT>/scripts/launch_hamster3d_preparation_v1.sh environment
# 网络中断后保持 source 与 manifest，使用新 session：
taskset -c 0 bash /home/wzy/dpvlm/route_set_v1/research_v2/releases/<COMMIT>/scripts/launch_hamster3d_preparation_v1.sh assets --resume
```

`record_job` 保存 launcher SHA、source commit、实际 command/PID/child PID/log/status/exit。只有上述准备动作可从此入口启动；模型输入门禁与推理入口完全分离。
