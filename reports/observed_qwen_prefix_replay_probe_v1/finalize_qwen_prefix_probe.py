"""Validate archived text receipts only; no model/NPZ/PT reads."""
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import xml.etree.ElementTree as ET

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
P=ROOT/'reports/observed_qwen_prefix_replay_probe_v1'
def read(name):return json.loads((P/name).read_text(encoding='utf-8'))
def write(name,value):(P/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def wall(status):return (datetime.fromisoformat(status['end_utc'])-datetime.fromisoformat(status['start_utc'])).total_seconds()

def main():
    remote=read('REMOTE_ARTIFACT_INDEX.json');files={x['local_path']:x for x in remote['files']}
    for name,item in files.items():
        if item['downloaded']:
            assert sha(P/name)==item['sha256'] and (P/name).stat().st_size==item['bytes']
    original=read('probe/artifact_index.json')
    for name,item in original.items():
        got=files['probe/'+name]
        assert item['sha256']==got['sha256'] and item['bytes']==got['bytes']
    s=read('probe/status.json');final=read('probe/final_audit.json');steps=read('probe/step_reports.json')
    ledger=read('probe/call_ledger.json');counts=collections.Counter(x['kind'] for x in ledger)
    assert counts==dict(full=10,replay=10,head=4,optimizer_full=2,optimizer_replay=2)
    assert all(x['state']=='completed' for x in ledger)
    assert s['planned_budget']==s['actual_issued_budget'] and s['gate_passed']
    assert sum(x['candidate_path_states'] for x in ledger)==16
    assert all(x['exact'] and x['post_update_feature']['exact'] and not x['differences'] for x in steps)
    assert read('probe/frozen_before_sha256.json')==read('probe/frozen_after_sha256.json')
    assert all(final['adapter_changed'].values()) and all(final['head_changed'].values())
    for name in ('historical_feature_comparisons.json','initial_feature_comparisons.json'):
        a=read('probe/'+name);assert len(a)==6 and all(x['exact'] and x['max_abs']==0 for x in a)
    junit=ET.parse(P/'validation/pytest.xml')
    tests=list(junit.getroot().iter('testcase'))
    assert len(tests)==37 and not any(x.find(t) is not None for x in tests for t in ('failure','error','skipped'))
    outer=read('probe.status.json');validation=read('validation/tests.status.json');env=read('environment/pytest_install.status.json')
    for r in (outer,validation,env):assert r['status']=='completed' and r['exit_code']==0
    timing={}
    for kind in counts:
        times=[x['elapsed_seconds'] for x in ledger if x['kind']==kind]
        timing[kind]=dict(calls=len(times),total_seconds=sum(times),median_seconds=statistics.median(times),all_seconds=times)
    prefix=[x for x in remote['files'] if x['local_path'].startswith('probe/prefix_cache/')]
    qa=dict(all_downloaded_original_hashes_verified=True,original_probe_index_all_entries_verified=True,
        downloaded_files=sum(x['downloaded'] for x in remote['files']),remote_only_pt_files=sum(not x['downloaded'] for x in remote['files']),
        original_remote_files=len(files),actual_budget=s['actual_issued_budget'],all_issued_completed=True,
        historical_initial_exact_examples=6,official_replay_initial_exact_examples=6,
        paired_micro_steps_exact=2,post_update_feature_pairs_exact=2,adapter_changed_count=len(final['adapter_changed']),
        head_changed_count=len(final['head_changed']),frozen_parameter_tensors=len(read('probe/frozen_before_sha256.json')),
        frozen_parameter_hashes_equal=True,validation=dict(passed=37,skipped=0,failed=0),
        new_archive_model_forwards=0,new_archive_searches=0,pt_downloads=0)
    write('QA.json',qa)
    cost=dict(probe_body_seconds=s['elapsed_seconds'],probe_body_gpu_hours_reserved=s['gpu_hours_reserved'],
        probe_outer_seconds=wall(outer),probe_outer_gpu_hours_reserved=wall(outer)/3600,
        allocated_peak_bytes=s['peak_cuda_memory_allocated_bytes'],reserved_peak_bytes=s['peak_cuda_memory_reserved_bytes'],
        allocated_peak_GiB=s['peak_cuda_memory_allocated_bytes']/2**30,reserved_peak_GiB=s['peak_cuda_memory_reserved_bytes']/2**30,
        validation_outer_seconds=wall(validation),pytest_seconds=5.79,environment_outer_seconds=wall(env),
        ledger_by_kind=timing,prefix_payload_bytes=sum(x['bytes'] for x in prefix),prefix_payload_count=len(prefix),
        cost_scope='Body nested in outer; feature/head/optimizer calls nested in body. Backward/loss/state copy/hash/load/disk costs not separated; not a speedup benchmark.',
        cache_prepare_or_training_speedup_claim=False)
    write('COST_AND_BUDGET.json',cost)
    report=f'''# TRAIN6 Qwen 前缀重放：实际技术门禁通过

实际 source `eddfacaddab2d12c67f5a56fd775de173c05f9b2`；两父283200/283201的全部三目标指令，共6个固定TRAIN输入。2026-10-03 01:01:24 UTC完成，所有门禁通过。没有DEV、保留分区、正式续训或路线质量比较。本结果只支持原长度串行B1的冻结前缀重放，未证明padding/batched replay、实际学习收益或加速比。

## 原始结果

| 检查 | 实测 |
|---|---:|
| 当前官方完整特征 vs 共同头历史缓存 | 6/6逐值精确一致 |
| 写盘重读prefix后的零LoRA尾层特征 vs 完整前向 | 6/6逐值精确一致 |
| 两独立head/Adam分支的feature/loss/梯度/更新/optimizer/RNG | 2/2逻辑微步精确一致 |
| 每步更新后的full/replay特征 | 2/2精确一致，且均相对初始改变 |
| 实际改变的LoRA参数张量 | 8/8，114688参数 |
| 实际改变的普通头参数张量 | 60/60 |
| 冻结基础参数前后SHA | {qa['frozen_parameter_tensors']}/{qa['frozen_parameter_tensors']}一致 |
| 服务器测试 | 37 passed，0 skipped，5.79秒 |

第一步68个可训练张量中64个梯度非零；四个A张量因B零初始化而为零，符合预期。第二步68/68均有非零有限梯度。两步分别用两个父的target0正参考，loss为0.07525034994和0.20142270625；它们是不同样本的链路值，不能当学习曲线或效果改善。

原官方layer26输入已包含视觉/DeepStack及0..25层结果；仅运行原26/27层与final norm，按原mask作FP32 mean+last。每次更新重算尾层；未用过时的最终4096维缓存训练。历史缓存没有完整token IDs，因此仅声明原观察行key、图像、processor/source、token count与特征数组核验，本次实际prompt/token IDs均另行保存。

## 实际预算与成本

账本28项全部完成：10次完整Qwen、10次serial尾层重放、4次K4路径头，共16个路径状态；full/replay各2次optimizer更新，合计4次。没有额外调用、重试或候选选择。首次捕获6+更新时2+更新后2构成10次full，replay同构。见 [完整账本](probe/call_ledger.json) 和 [机器成本表](COST_AND_BUDGET.json)。

| 阶段 | PID/child | UTC开始–结束 | 墙钟 |
|---|---|---|---:|
| 独立环境安装pytest8.4.2 | {env['pid']}/{env['child_pid']} | {env['start_utc']}–{env['end_utc']} | {wall(env):.6f}s |
| 服务器CPU验证 | {validation['pid']}/{validation['child_pid']} | {validation['start_utc']}–{validation['end_utc']} | {wall(validation):.6f}s |
| 真实probe完整job | {outer['pid']}/{outer['child_pid']} | {outer['start_utc']}–{outer['end_utc']} | {wall(outer):.6f}s |

probe主体{s['elapsed_seconds']:.9f}s / {s['gpu_hours_reserved']:.12f}保守GPUh，嵌套于完整job {wall(outer):.6f}s / {wall(outer)/3600:.12f}GPUh，不能相加。GPU1、35%显存、CPU affinity `[1]`。峰值allocated {s['peak_cuda_memory_allocated_bytes']}B（{cost['allocated_peak_GiB']:.6f}GiB），reserved {s['peak_cuda_memory_reserved_bytes']}B（{cost['reserved_peak_GiB']:.6f}GiB）。这些是进程峰值/保守墙钟费用，不是利用率。

6份prefix序列化文件共{cost['prefix_payload_bytes']}字节。账本full-feature调用中位{timing['full']['median_seconds']*1000:.6f}ms、serial replay中位{timing['replay']['median_seconds']*1000:.6f}ms，但包含不同冷启动/梯度阶段，且未拆出backward、loss、状态复制、加载、hash、写盘。不能用二者比例声称正式训练加速或端到端时延；总probe除以两步也不是稳态微样本耗时。

## 来源、归档与限制

共同头固定composite last12000，SHA `ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3`。Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`，实际Torch2.4.1/Transformers4.57.1。运行源、官方依赖SHA、checkpoint/cache/export身份均见 [preflight](probe/preflight.json)。安装只发生在项目私有 `.venv-qwen`，日志保留pytest8.4.2及其依赖版本，没有升级共享环境或模型依赖。

本地下载42项JSON/文本/源码/日志，13项PT只保留远端路径、字节数和SHA；未下载权重或prefix张量。原55项全部索引及probe自身artifact_index逐条一致，[REMOTE_ARTIFACT_INDEX](REMOTE_ARTIFACT_INDEX.json)提供恢复来源，[QA](QA.json)记录核验。同步和本报告只读取已完成证据，0新增forward/search。

此probe为fresh-only、无重试恢复入口。两步状态PT是诊断证据，不能据此重新发相同预算。正式321输入prefix准备与3000×32配对必须另行冻结、实际执行及核验；本报告不宣称这些阶段已经完成。没有向MAIN添加质量行。
'''
    (P/'TECHNICAL_RESULTS.md').write_text(report,encoding='utf-8')
    for name in ('archive_qwen_prefix_probe.py','finalize_qwen_prefix_probe.py'):
        shutil.copyfile(ROOT/'.bootstrap'/name,P/name)
    index={f.relative_to(P).as_posix():dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='LOCAL_ARTIFACT_INDEX.json'}
    write('LOCAL_ARTIFACT_INDEX.json',index)
    print(json.dumps(dict(qa=qa,cost={k:v for k,v in cost.items() if k!='ledger_by_kind'},local_items=len(index)),ensure_ascii=False))

if __name__=='__main__':main()
