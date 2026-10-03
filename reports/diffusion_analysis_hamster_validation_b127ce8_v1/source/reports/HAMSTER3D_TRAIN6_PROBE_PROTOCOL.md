# 3D HAMSTER TRAIN6 技术探针草稿 v1

状态：只读官方代码与纯契约测试形成的草稿；未安装新环境、未上传资产、未加载/运行模型。本协议不能作为机器人系统已复现或方法优势的证据。先由根线程审查、冻结、完成资产和私有环境，再决定是否实际运行。原上传、下载和项目模型文件均不修改。

## 固定输入与调用预算

仅使用已授权旧 TRAIN 父 `two_row_reach_283200/283201`，每父 target0/1/2，顺序固定共6输入。绑定原 composite108 export SHA `04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc`；只解析 observations.jsonl 前六行。逐一核 image/同目录 observation.npz 的封存源 SHA与父根路径；不读取 supervision、routes、verification_only、DEV/LOCKED raw。校准矩阵和当前状态来自 observation NPZ；**模型只接收官方 RGB＋对齐深度＋语言**，K/c2w仅用于输出坐标适配，当前状态不额外拼进提示。

最多6次 official `model.generate`，每次 batch1/K1、greedy、max_new_tokens=1024、do_sample=False、temperature=None、top_p=None；与官方API保持一致。总生成token上限6144，训练/优化器/额外warmup/隐藏补采均0。每次原始autoregressive forward与输入token数另计，不能把6次generate称6次网络forward。失败、空/部分解析、非有限输出、达到token上限都留在原候选槽；不筛选、不补槽。

可运行性上限固定为加载阶段600秒、每请求300秒、整个主体2400秒；Linux alarm及token级停止共同约束，底层不可中断C/CUDA调用可能使真正退出稍晚，必须记录实际超时与耗时。OOM/时间超限/工程异常保存已发调用与剩余未发槽后停止，无自动第二方案。没有低分辨率、量化、关闭深度、改变dtype或改KV策略回退。empty/格式不规范但模型完成的候选保留，并可继续后续预声明输入，不重试该槽。通过streamer记录已经产生的token，hard timeout也保存部分token/text；streamer不新增forward。

## 固定官方来源与输入处理

资产manifest使用原服务器字节SHA `53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032`，逐SHA复核50代码＋18模型文件。代码commit `97216a8493f46301bf569d398462b8bb21c458c5`，模型revision `ddc5987a56cdcb14e5e2297817612532e46e912b`。只使用已存在本地目录和私有 `.venv-hamster3d`，要求 assets/environment 两个完成receipt均成立、所有core pins一致。Hugging Face offline/token=False，Python socket连接拒绝；入口没有下载或安装分支。

直接复用官方[preprocessing.py](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/hamster3d/inference/preprocessing.py)：RGB最长边640双线性，aligned depth最近邻；深度先float16再float32，最终随模型输入转bf16。模型[geometry路径](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/hamster3d/model/modeling.py)还会按Qwen视觉grid以bilinear重新采样geometry RGB/depth，保留该官方行为，不能误述成全链路最近邻。原instruction＋官方v5 suffix，不添加答案token/真实目标。

模型使用官方[API生成参数与解析器](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/hamster3d/inference/api.py)，只替换模型放置方式。保存实际loaded generation_config/use_cache和attention implementation；config内use_cache=false不代表最终generate状态，不能凭配置猜测或擅自改写。官方解析器可能fallback至v3或跳过无point条目，保存原text/token及解析后的原始列表，解析完成不是任务成功。

输出 `[u,v,depth]` 依官方[demo公式](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/scripts/trajectory_prediction_gradio.py)在原224图像尺度按 `d*inv(K)[uW/1000,vH/1000,1]` 转camera，再用实际camera-to-world转world。保留负fx/fy，不abs、不翻图、不裁u/v、不换depth、不吸附真目标。raw UVD/动作、camera/world数组都保存；本轮不制造H24质量指标、不做起点/终点修复、不评价TipValid/任务success。

## 8.4 GiB 放置与待实证风险

要求可见设备仅已授权GPU1 UUID，CPU0/单线程；PyTorch allocator上限取35%显存和8.4GiB的较小者。至少32GiB系统可用RAM才尝试加载。先真实 bf16 权重完整加载到CPU，拒绝未填入的meta参数；不会瞬间把17GiB权重全移GPU。完整视觉塔、完整LingBot encoder、geometry merger/fusion、embedding/norm/rotary和lm_head常驻GPU；实测常驻参数量必须留至少2GiB工作区才继续。

语言模型36个**实际被调用的decoder layer**逐层采用[Accelerate 1.10.1 cpu_offload](https://github.com/huggingface/accelerate/blob/v1.10.1/src/accelerate/big_modeling.py)，保留CPU权重映射、计算时载入GPU、之后释放。不给ModuleList挂无效计算hook，也不给geometry encoder做meta offload：[官方encode](https://github.com/DAVIAN-Robotics/3D_HAMSTER/blob/97216a8493f46301bf569d398462b8bb21c458c5/hamster3d/model/geometry_encoder.py)被直接调用，并从参数读device/dtype，绕过普通outer forward hook。启动后逐参数审计meta仅落在decoder，各其他参数必须cuda:0。

这只是候选工程方案。尚未证明custom from_pretrained低内存构造、所有非持久buffer、decoder KV/输入搬运及峰值能在配额内运行，也未证明与官方全GPU逐token完全等价；不能把一般offload文档当成实测证据。失败即保存，不自动扩大显存/换放置。CPU↔GPU每token搬运成本可能很高，6个K1技术调用不能当正式K4性能或固定时间预算结果。

## 保存与后续

fresh-only输出保存status/ledger、源与配置/资产/准备receipt/input SHA、placement、实际generation配置、输入提示token、原始与部分生成token、原text、官方解析动作/UVD、camera/world投影、各阶段实耗、峰值allocated/reserved/RSS、保守整任务GPU预留小时。原source不修改，不提供resume或自动重试。代码CLI尚不配launcher，避免在资产/环境未完成前误启动。

新文件：`scripts/probe_hamster3d_train6.py`、`configs/hamster3d_train6_probe_v1.json`、`tests/test_hamster3d_train6_probe.py`及本文。实际本地纯契约测试 **19 passed / 0 skipped / 0.71 s**，无Torch导入/forward/SSH/SCP。覆盖输入角色及额外答案键拒绝、预算固定、signed-K等值与不裁预测、空候选保留、路径越界、streamer部分token计数及offline socket恢复。真实GPU兼容性仍未测试；只有根线程冻结后授权的实际运行可产生技术结论。
