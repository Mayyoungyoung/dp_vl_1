# 固定3D HAMSTER资产本地下载实证

官方固定模型 `DAVIAN-Robotics/3D_HAMSTER@ddc5987a56cdcb14e5e2297817612532e46e912b` 的18文件已在本地完整校验。此记录不包含模型加载、推理、正式实验质量或服务器上传完成的主张。此前服务器网络失败仍保存在独立历史目录，没有被这次成功覆盖。

固定下载source `d0bdc9e582cf4d22bac979966e4557a93659d918`，冻结helper SHA `a8effa31b30c787026ef54994d98190ab8f42bb7c2b66ef7973042fc3523236c`。采用原服务器manifest逐size/SHA检查，identity、每次worker issued/exit与成功记录均原字节保留。13个已校验小文件共4,578,795B复用，另一次真实传输tokenizer.json 11,422,654B；14小文件合计16,001,449B。小文件阶段22.953秒。

权重阶段run `20261003T025803Z_weights_79384` 于02:58:03.023829 UTC启动，实际exit0，主体2178.094秒；4个shard分别590.344/569.969/588.203/429.407秒（嵌套，不能加到主体后再收费）。四次transfer worker各单attempt，无自动重试或剩余.part。权重18,279,874,332B，全部18资产18,295,875,781B；独立全18重读校验见checks/hamster_local_all18_verified.json。权重仅在本地忽略目录 `.bootstrap/hamster3d_model_ddc5987a`，不进入Git。每个官方SHA和bytes见原manifest及完成receipt。

GPU小时0，模型调用0，数据输入0；上述CPU与网络准备墙钟不能充当推理延迟。下一服务器上传是另一独立阶段，须重新全18本地校验、远端逐文件SHA和最终全18检查；不能由本地成功推断远端成功。后续仍需要原assets --resume的完整receipt、独立环境及实际有限TRAIN探针。
