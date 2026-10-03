# 3D HAMSTER服务器资产复核与独立环境完成

固定准备源码1b0348ef48393a2d98113956575e99388c40d4a6，官方code 97216a8493f46301bf569d398462b8bb21c458c5、model ddc5987a56cdcb14e5e2297817612532e46e912b；原manifest SHA53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032。

`assets --resume` session20261003T040919Z_assets_758043于04:09:19.897541–04:10:38.717650 UTC exit0。68文件（50源码、18模型）全部reused_verified，library下载调用0；它复核先前上传资产，没有重复下载。body78.638638秒、outer78.820109秒，CPU0、GPU/模型/数据输入均0。旧服务器HF失败不覆盖，详见原preparation_validation family。

`environment` session20261003T041150Z_environment_759295于04:11:50.122235–04:20:11.440906 UTC exit0。仅建立项目私有`.venv-hamster3d`，Python3.11.13、system_site_packages=false，CPU0/CUDA隐藏；body500.845579秒、outer501.318671秒。官方PyPI/PyTorch cu121安装与pip check、固定模块导入完成。核心锁定Torch2.4.1+cu121、torchvision0.19.1+cu121、transformers4.57.1、accelerate1.10.1、huggingface-hub0.35.3、tokenizers0.22.2、safetensors0.6.2、numpy1.26.4；实际全部传递依赖版本和安装来源保留在pip_freeze.txt与pip_install_report.json，不能只凭核心版本假称所有依赖预先锁死。没有修改共享环境。

阶段末本项目资产与环境owned24,145,299,790B，小于35GiB cap，filesystem free1,281,444,143,104B；没有从磁盘空余推断共享配额扩大。上述准备成本独立记录，不能冒充单请求推理延迟。两阶段均model_loaded=false、forward0、训练0；后续TRAIN6真实探针是另一个已登记运行，是否成功须读其完成账本。资产/依赖完成不等于机器人系统复现。

原19个服务器记录逐字节SHA核验并下载，模型与wheel文件均不进Git；源码/原始manifest可由reports/hamster3d_preparation_validation_v1的已封存副本恢复。命令均用原release下`scripts/launch_hamster3d_preparation_v1.sh`，先assets --resume，根代理核完成后才独立environment。所有状态与实际退出时间见sessions目录和完成marker。
