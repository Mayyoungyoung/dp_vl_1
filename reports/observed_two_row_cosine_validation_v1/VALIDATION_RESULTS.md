# 固定源码71e3485的真实服务器验证

2026-10-02 22:10:44.793556–22:10:54.265947 UTC，`wzy3090`实际完成 **91 passed in8.85s，无skip**；record_job PID592391/child592393，exit0，外层9.472391秒。JUnit记录91项、0失败、0错误、0跳过，三项新增Torch测试的case名称与成功状态均逐项核验。

- 不可变source：`71e34850481b79bdbc828d1c1944f0de00329960`。服务器code-only tar3686400字节，SHA256 `fdce82a6a19cc4c4188a054be006af472481c4df468ec3e82ef219f6cc9fa9ce`。
- 原始status、log、registry及pytest.xml逐字节复制；原命令、cwd与时间以这些原件为准。测试没有GPU或训练语料推理。
- 真Torch包含实际shared loop连续200与100+恢复100全STATE_FIELDS相等（含history）、旧constant adapter与原2步循环相等、实际LR/scheduler恢复及scoped函数引用恢复。纯测试涵盖固定12000/1536000预算、48次DEV选择、LR相位、完整sample-chain、跨协议和整体LR缩放拒绝。
- `.bootstrap/cosine_tests_v1.sh`与实际服务器incoming wrapper字节相同；两份原件均保存。源码、launcher、config以及五个实际测试文件另有副本，测试只追加`.source.txt`后缀而不改字节，避免默认pytest重复收集。
- `source_index.json`索引固定release下scripts/routeset/configs/tests的335个文件。`artifact_index.json`记录18项复制/索引来源，全部大小与SHA匹配，索引SHA256为 `61ff5f0507c2e194d698f43077dc319a97785eb6e159c319fe70179f5bf86a9f`。

此归档证明指定工程门禁通过，不预写cosine的训练结果或收益。归档阶段未读取仍在运行的主训练完成产物，未启动训练/推理/仿真，未读取reserved原始内容。root已另行启动fresh12000；实际退出后才分析保存池。
