# HAMSTER 两个 exact8 技术调用：输出逐值一致，传输门通过

实际完成原Accelerate hook与项目内pinned层传输的两个8-token partial。全部8次实际输出tensor均为[1,1,151936] bf16、finite、torch.equal，最大数值差0；两边8个token还与此前原87token失败输出的前8相同。新decode中位耗时是原34.05%，通过固定≤75%门槛。这里只验证一个TRAIN输入的8步条件输出与工程可行性，不是完整路径、机器人质量或一般输入数值等价证明。

固定cbcd8129c967db6f0995a824754752cc7a6f0f5b，429源码文件前后SHA一致；源driver实际archive SHA600d0ec9c68fcecfcbd016df2dce6e8ab54cddf9a407743909d36b41764e4f9c、config ee095817c93ab8afe4f1f9cdef076f23a3c713b9f17284a256a60508b1489cb2。真实39项测试/0skip与私有环境补充原件见[准备报告](../hamster3d_transport_preparation_v1/PREPARATION_RESULTS.md)。原官方模型/算子/bf16/预处理不改，旧b127失败全部15件原样复制到独立original_failure。

| 已测范围 | 原hook | 新层传输 |
|---|---:|---:|
| generate / forward / token | 1 / 8 / 8 | 1 / 8 / 8 |
| generation主体秒 | 29.799675 | 11.060573 |
| prefill秒，含logit拷贝 | 5.808302 | 2.816935 |
| 7次decode中位秒，含logit拷贝 | 3.385401 | 1.152700 |
| peak allocated bytes | 4,655,018,496 | 6,498,608,640 |
| peak reserved bytes | 4,800,380,928 | 6,618,611,712 |

原后新顺序固定，allocator/OS/cache温热与驻留层数、pinning、逐层hook、cache清理的差别共同存在，不能单独归因某一个优化。两臂都保存完整词表logits并计入CPU copy开销；本次实际每步仅输出最后位置，不是人为截断prefill。每次copy约0.00021–0.00057秒；这些仪器下延迟也不是无仪器部署吞吐或完整冷启动延迟。没有warmup、重试或把两次partial合成候选集合。

实际model_load距主体起点94.508725秒（包括资产核验等），transport conversion135.259365秒，32层pinned参数12,348,571,648B，4层常驻。内存规划包含活动整层、340B buffers及2GiB workspace；实际RSS峰22,116,167,680B <32GiB，whole CUDA reserved6,618,611,712B <8,864,694,272B。32个offload hook各pre/post8次、4驻留层hook计数0；全部官方tensor前后字节SHA保持。转换本身成本不能藏入缓存性能。

主体PID790704，05:19:48.826071–05:25:10.882132 UTC，322.056326秒/0.089460097GPU占用小时。外层PID790698/790703，05:19:48.654232–05:25:20.707600，332.053368秒/0.092237047GPU占用小时，包含主体及释放等费用；两者嵌套，绝不相加。旧300秒超时的外层411.174125秒另行保留，不能用新对照抹去旧成本。

report SHA64a3268eb541347a4a0ddccd251acfb6d72d794efcbee2bf8670b6e196b1835d，原inner状态SHA63a5fcafff06630c71368d6296b3f2d6682668d817070f656ab4ae81b5d8d875。REMOTE_ARTIFACT_INDEX记录41件：两份真实完整CPU logits PT留服务器并hash索引，NPY/NPZ保留ignored runs/synced_hamster3d_transport_exact8_v1，其余原日志/输入/所有层审计保留本目录。完整模型权重不入Git。

root已实际读完输出逐值/预算/资源/外层收据，决定下一独立fresh目录只尝试原TRAIN0一个完整K1请求，最多1024新token、请求900秒；全部加载/转换/生成计费，不自动补成6请求。该后续尚未运行，不能将本次技术门当完整路线成功、语义正确或方法贡献。原两个partial、首个87token失败都保留，不覆盖、不补JSON、不改测试标准。
