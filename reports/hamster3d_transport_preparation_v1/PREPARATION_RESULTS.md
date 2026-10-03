# HAMSTER 等值传输对照：私有测试依赖及真实39测试

固定源cbcd8129c967db6f0995a824754752cc7a6f0f5b的429文件实际部署SHA一致；原b127 probe字节SHA63a561d8c12fafe2fbe07eafbf3d72bf69db6d507203f14fa316ae10f60fc76d未变。第一请求87token超时原件仍保留。

启动测试前发现私有.venv-hamster3d没有pytest，因而先执行独立固定依赖补充。仅新增pytest8.4.2、iniconfig2.1.0、pluggy1.6.0、pygments2.19.2四个官方SHA固定纯wheel，--no-deps/--no-index/--require-hashes；现有packaging26.3满足依赖，不升级。before/after全45原包版本、4个Accelerate源码、原环境全部收据逐项保持，pipcheck均通过，共享环境未动。实际05:17:57.712369–05:18:03.346723UTC，PID789463/789468 exit0，主体5.571659秒、外层5.634354秒，0GPU/模型/测试；时间嵌套不相加。

root读完补充收据后独立真实39测试：新20项含3真实CPU Torch/Accelerate语义测试、旧19回归，全部通过/0skip。实际05:18:45.188185–05:18:48.548641UTC，PID789959/789964，runner3.319126秒、外层3.360456秒，CPU0/CUDA隐藏。pytest XML SHA4a8e7fff9d96a389df9ce560b8a312d62527ce5c4c315fd942ba1cc974bf393c。CPU测试不证明CUDA等值或官方模型速度。

此后已另行启动两个exact8实际调用；该运行中的科学结果不包含在这份已完成准备阶段归档。任何质量声明须来自独立probe证据，准备通过不算模型成功。所有配方/安装report/pytest源/实际状态及日志均绑定hash，0基础权重入Git。
