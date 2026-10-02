# 两排正式 prefix28：传统观测规划强控制

这是已有传统控制的独立数据适配，不是新方法。入口 `scripts/evaluate_two_row_astar_v2.py` 只接收正式 closed-prefix export；不改 `observation_multiroute_astar_v2.py`、v1网格或 `observation_prototype_grounding.py`。服务器首轮定向测试已运行，但source guard因换行格式不一致而失败，尚未进行任何正式两排DEV生成，也没有新增模型训练。

## 源码字节门禁修复记录

`0d730b7ab16e22ad3dceefca7ca9b97293f9f91d` 服务器定向测试为5 passed、1 failed，失败发生在规划源码核验，未开始DEV生成。原因是初版只登记本地LF精确SHA，而实际服务器导出的三个旧planner文件均为CRLF。已只读取回当前0d730及最初成功A*v2实验的 `a32e91e49e11fa682ac0166572cb62303d21affe` 三个对应文件：两份服务器文件逐字节相同，且仅将CRLF替换为LF后与本地逐字节相同。历史实际report的script/shared-grid SHA等于当前服务器CRLF SHA；report本身双端SHA为 `7a018abc3f2178b5e216e3d12236ecc2711e6b735402c7bedd69ee27da6e3791`。

新guard为每个文件登记**两种已经核验的精确LF/CRLF字节SHA**，然后再次要求规范LF SHA等于原登记值。只接受这两种完整字节形式；混合换行、额外空白或任何代码变化仍拒绝。运行产物保存actual SHA、规范LF SHA及实际格式，不能把规范哈希冒充服务器原字节SHA。这不是放宽规划逻辑或跳过门禁，共享planner文件零修改。原失败日志/status/registry与只读比较证据保存在 `reports/observation_two_row_astar_v2/source_guard_failure_0d730b7/`；修复后的服务器重测由新的冻结release执行，本文不提前记为通过。

固定数据是正式 indices0..15 的16TRAIN和64..75的12DEV_MODEL，共84个请求图文条件。数据必须通过现有严格export/当前机械门禁。只对实际写出输入进行生成；36个请求DEV条件中缺输入的条件和其4个未尝试槽另列，不伪造观察。所有实际DEV条件均评价，无参考、未知类型、未支持语言和规划失败都不能静默删除。

TRAIN拟合使用同一已知正参考池。颜色原型仅根据每条精确语言指令所对应的TRAIN轨迹末点附近可见RGB-D像素拟合；workspace沿原 `fit_workspace` 从TRAIN路径边界加原6cm margin得到。两fit函数仅收到TRAIN行和`observation/routes`字段，不收到真实目标、模式编号、仿真障碍或DEV标签。缺正参考/不足可见像素按照原prototype规则记录，不用DEV补原型或调阈值。它是封闭精确指令控制，不具有Qwen或开放词汇能力。

每次DEV请求使用RGB、metric depth、相机内外参、当前末端pose/open和指令。current NPZ由输入图像同目录的 `observation.npz` 定位并核验export源hash，不从DEV监督取目标。原v2先做颜色连通区域定位和观测网格，再最多执行4次独立有界搜索；每槽原上限20000展开节点和2秒，时间检查原样每64节点一次，因而2秒不是严格墙钟截断。网格构建、27邻接attachment检查、原始/H24可见点/光线检查以及输出开销均计时，不声称整请求最多8秒。原2.5cm voxel、2cm可见表面clearance、4cm当前末端接触半径、3cm目标depth shell及所有其它CONFIG保持不变。

失败保留NaN槽，完整重复路线仍保留，检查后不筛选、不修复、不用额外搜索替换。每个请求K4指4次最多搜索和4个提交H24槽；原始完整折线与其H24是同一候选的表示，两个文件及所有内部检查均保存。颜色分割组件和局部attachment数量由原记录保留。异常程序/资源错误会保留已完成请求、当前traceback后中止，不把未执行搜索捏造成失败完成或自动重试。

每请求完整K4池和raw折线先写NPZ、计算SHA并写generation seal，然后才读取该条件的监督geometry/route_config，调用同 `observed_two_row_tip_eval_v1`。2cm线段/3cm目标/5mm起点/恒定reach事件和两排真实经过类型判据不变；unknown有效路径仍有效但不贡献虚构类别。模型不能使用真实几何修复输出。报告已知正参考类型覆盖分母、无参考数、全部候选和失败分母；不把参考条数当全部解数。

拟合时间单独计。逐请求报告实际IO及hash、反投影、定位/网格/搜索/原proxy检查、候选封存、评价label IO和两排检查的连续墙钟时间，并保留first/median/p95/all36实际值、CPU affinity和线程。工作区/prototype拟合不混为在线时间；同K4比较允许不同计算成本，不将它称为与神经头等时或同骨干优势。既有网格只表示有限可见点与深度自由空间代理，局部接触许可不等于未知空间安全，最终tip检查也不是全机械臂/IK/执行认证。

未来实测命令（仅root冻结并排CPU后执行）：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.record_job \
  --output /home/wzy/dpvlm/route_set_v1/runs/observation_two_row_astar_v2 \
  --run-id dev_model --resume-strategy fresh-output -- \
  /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.evaluate_two_row_astar_v2 \
  --data /home/wzy/dpvlm/route_set_v1/data/observation_two_row_prefix28_v1 \
  --output /home/wzy/dpvlm/route_set_v1/runs/observation_two_row_astar_v2/dev_model
```

运行cwd和CODE_COMMIT必须是审阅后的不可变release；以上只是准备命令，不代表作业已启动。
