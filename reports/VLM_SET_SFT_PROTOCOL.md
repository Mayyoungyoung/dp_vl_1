# 直接VLM独立与整集合输出：基线接口与待执行协议

状态：仅实现并测试纯序列接口，尚未训练此SFT模型，不能将已有hidden-state路线头或其LoRA实验当作直接VLM集合SFT。首个实际验证使用固定真实Qwen3-VL-2B revision89644892与原processor/transformers4.57.1；不改共享依赖。这个分支用于补齐用户要求的强比较，不宣称SFT、LoRA或一次输出多条是新方法。

已读[Qwen官方训练框架](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)及其[数据处理源码](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/data/data_processor.py)：官方采用用户/助手消息并对助手答案做监督。本项目不照抄其token数字扫描，而是由实际processor产生完整序列与无答案生成前缀，逐token核验前缀完全一致并全置-100；不一致立即拒绝，禁止猜测prompt长度。训练的因果teacher forcing与推理严格分离，生成时仅可传无答案前缀。首次真实预检须证实这一边界、有限loss、真实adapter梯度和参数变化；纯序列测试不能代替它。

输出契约为世界坐标毫米整数的`K×H24×[x,y,z,open]` JSON。1mm坐标量化每坐标误差不超过0.5mm，不改变几何验收。每条包含完整当前起点，不事后强行修复；格式错误/短缺保留NaN槽，超出K的输出明确计更多候选且不从中挑K。重复有效路线仍占槽。原始字符串、token数、所有生成时间和解析状态必须保留，重试不得藏入同一预算。

生成输入仅RGB、语言、当前pose7/open1、相机内外参和实际观测depth。为直接VLM提供depth图像接口：RGB高/低字节编码毫米深度，第三通道标记有效性；无效/缺失不当作已证实自由空间。该非标准深度编码不因可逆就代表Qwen已学会读深度；vision processor重采样与视觉编码也不等同于原RGB-D点云编码器。因此独立与整集合对照必须共享这个相同输入/编码，初版不把二者与点云头称为完全相同表示或合理收敛的强对照。先检查实际TRAIN depth/token/路径量化和处理器行为，不按DEV结果调整编码。

预期首个训练比较使用同一variable-K训练模型，显式包含K1与K4请求；避免只训K4后用未见K1提示当弱独立基线。同checkpoint独立四次K1与一次K4分别产生4候选，无池化/筛选/隐藏修复；多次随机重复分别计算再平均。序列的顺序仅是序列化选择，已知参考不等于全部解，参考不足时允许同类正例重复，不生成不存在标签。未来若比较两个独立训练SFT目标，需另行对齐实际目标路线曝光、训练token和计算，不以训练步数相同偷换公平预算。

正式训练长度、K混合概率、采样温度、max_new_tokens与固定选模应在真实token/显存预检后一次性记录。恢复保存adapter、optimizer、scheduler、全部RNG、sampler、step、累计token/候选/耗时；不重复保存2B冻结基座。DEV仅负责开发，TEST_LOCKED/OOD不参与。当前已完成5项纯格式测试，实际训练、推理结果和成本均仍未测。
# Actual preflight failure and equivalent-loss repair

2026-10-02 13:14 UTC, immutable source `5fb74b15ee0b4b89f2900c63363e24434ec90ac5`: actual pinned Qwen loaded and the processor verified exact answer-free prefixes. The fixed TRAIN example required 776 prompt tokens; K1/K4 required 344/1361 supervised tokens (1120/2137 total). The full-vocabulary backward failed with CUDA OOM under the unchanged 35% GPU1 memory cap. This is a failed preflight, not completed SFT. Raw log, exit1 status and token audit are preserved in `vlm_route_sft_preflight_v1/`. CUDA logical device0 in that log maps to authorized physical GPU1.

The next implementation computes the identical shifted causal cross entropy only at supervised answer positions, in 64-token vocabulary chunks, recomputing each chunk during backward. The vocabulary head stays frozen; no answer tokens, trajectory points or candidates are truncated. Dense-loss/input-gradient equality has a dedicated small CPU test; actual-model success remains unverified until the separate v2 job finishes. The initial/final adapter files remain on the server and outside ordinary Git.
