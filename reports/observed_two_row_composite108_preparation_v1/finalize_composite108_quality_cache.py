"""Finalize completed validation/quality/cache evidence, never inspect training."""
import hashlib,json,shutil
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
P=ROOT/'reports/observed_two_row_composite108_preparation_v1'
V=ROOT/'reports/observed_two_row_composite108_validation_v1'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def cost(p):
 v=load(p);assert v['status']=='completed' and v['exit_code']==0
 return dict(pid=v['pid'],child_pid=v['child_pid'],source_commit=v['code_commit'],start_utc=v['start_utc'],end_utc=v['end_utc'],
  process_seconds=(datetime.fromisoformat(v['end_utc'])-datetime.fromisoformat(v['start_utc'])).total_seconds(),exit_code=0)
q=load(P/'train_quality.json');m=load(P/'export_manifest.json');receipt=load(P/'merged_qwen_cache/composite_cache_receipt.json')
c=load(P/'cache/new_qwen_cache/status.json');npz=load(P/'CACHE_NPZ_HASH_INDEX.json')
tree=ET.parse(V/'pytest.xml').getroot();cases=list(tree.iter('testcase'))
assert len(cases)==70 and not list(tree.iter('failure')) and not list(tree.iter('error')) and not list(tree.iter('skipped'))
assert q['source_export_manifest_sha256']==sha(P/'export_manifest.json')==receipt['source_export_manifest_sha256']
assert q['positive_references']==len(q['references'])==1663 and len(q['conditions'])==q['observed_train_inputs']==285
assert q['capacity_gate_passed'] and not q['capacity_failures'] and q['model_h24_tip_valid_references']==1663
assert q['audit_source_sha256']==sha(P/'quality_source/scripts/run_observed_two_row_composite.py')
assert q['quality_core_sha256']==sha(P/'quality_source/scripts/audit_two_row_train_reference_quality.py')
assert c['samples']==96 and len(npz['files'])==321 and all(r['origin_bytes_sha256_equal'] for r in npz['files'])
assert receipt['old_reused_count']==225 and receipt['new_encoding_count']==96 and receipt['reused_dev_count']==36
assert all(x['npz_bytes_equal'] and x['feature_arrays_equal'] for x in receipt['old225_by_id'])
assert receipt==load(P/'cache/cache_stage_receipt.json')
newrows=[json.loads(r) for r in (P/'cache/new_train_observations.jsonl').read_text().splitlines()]
assert len(newrows)==96 and all(r['split']=='TRAIN' and int(r['parent_id'].split('_')[-1]) in range(400000,400032) and set(r)=={'id','parent_id','split','image','instruction'} for r in newrows)
for key,local in [('new_config_sha256','cache/new_qwen_cache/cache_config.json'),('new_index_sha256','cache/new_qwen_cache/samples.jsonl'),('new_status_sha256','cache/new_qwen_cache/status.json')]:
 assert receipt[key]==sha(P/local)
for name,value in receipt['artifact_sha256'].items():
 if name.endswith('.npz'):
  record=next(r for r in npz['files'] if r['filename']==name);assert record['sha256']==value
 else:assert sha(P/'merged_qwen_cache'/name)==value
def stats(key):
 values=np.array([r[key]['length_m'] for r in q['references']]);return dict(min_m=float(values.min()),mean_m=float(values.mean()),median_m=float(np.median(values)),p95_m=float(np.percentile(values,95)),max_m=float(values.max()),over2m=int((values>2).sum()),over3m=int((values>3).sum()))
quality=dict(protocol='composite108_actual_train_quality_summary_v1',positive_references=1663,actual_train_inputs=285,actual_train_parents=95,
 capacity_passed=1663,model_H24_tip_valid=1663,unknown_reference_types=sum(r['unknown_types'] for r in q['conditions']),
 no_positive_conditions=sum(r['positive_references']==0 for r in q['conditions']),
 event_transition_mismatches=sum(r['raw_event_transitions']!=r['h24_event_transitions'] for r in q['references']),
 maximum_nearest_endpoint_Linf_m=max(r['endpoint_support']['nearest_Linf_m'] for r in q['references']),
 maximum_nearest_endpoint_L2_m=max(r['endpoint_support']['nearest_L2_m'] for r in q['references']),
 raw_length=stats('raw'),model_H24_length=stats('model_H24'),filtering=False,dev_raw_arrays_opened=False,
 quality_json_sha256=sha(P/'train_quality.json'),quality_internal_seconds=q['elapsed_seconds'],job=cost(P/'quality.status.json'),
 scope='Capacity and checked positive-reference representation only; not model fit, generalization, or execution evidence.')
write(P/'QUALITY_SUMMARY.json',quality)
cachejob=cost(P/'cache.status.json');enc=load(P/'cache/encoding_process.json');assert enc['exit_code']==0
cache=dict(protocol='composite108_actual_cache_summary_v1',actual_new_qwen_encodings=96,reused_old_cache_files=225,
 reused_dev_cache_files=36,merged_cache_files=321,old_and_new_container_source_hashes_exact=True,
 new_input_contract=sorted(newrows[0]),new_roles=['TRAIN'],new_parent_indices=[0,31],
 official_revision=load(P/'merged_qwen_cache/cache_config.json')['revision'],
 feature_dimensions='2048 mean +2048 last =4096 route-condition features',extractor_sha256=receipt['extractor_sha256'],
 source_export_manifest_sha256=receipt['source_export_manifest_sha256'],new_load_seconds=c['load_seconds'],
 new_encoding_including_load_seconds=c['total_seconds'],new_encoding_subprocess_seconds=enc['elapsed_seconds'],
 peak_cuda_allocated_bytes=c['cuda_peak_allocated_bytes'],cache_job=cachejob,
 encoding_body_gpu_hours=c['total_seconds']/3600,outer_cache_job_gpu_hours=cachejob['process_seconds']/3600,
 cost_scope='Nested intervals: load is inside encoding body, body inside encoding process, process inside complete cache job. Historical225 cost not added. Not end-to-end planning latency.',
 new_forward_requests_in_archive=0,training_results_opened=False,NPZ_downloaded=False,
 merged_receipt_sha256=sha(P/'merged_qwen_cache/composite_cache_receipt.json'),cache_NPZ_hash_index_sha256=sha(P/'CACHE_NPZ_HASH_INDEX.json'))
write(P/'CACHE_SUMMARY.json',cache)
validation=dict(protocol='composite108_actual_linux_validation_v1',tests=70,failures=0,errors=0,skips=0,
 actual_torch_cases=sum('test_two_row_composite_training_torch' in x.get('classname','') for x in cases),
 pytest_seconds=16.90,job=cost(V/'tests.status.json'),pytest_xml_sha256=sha(V/'pytest.xml'),
 scope='Actual Linux CPU tests, including real Torch initialization/gradient-loop/resume checks; no research data inference.')
write(V/'VALIDATION_SUMMARY.json',validation)
for filename in ('sync_composite108_quality.py','sync_composite108_cache.py','finalize_composite108_quality_cache.py'):
 shutil.copyfile(ROOT/'.bootstrap'/filename,P/filename)
report=f'''# Composite108：实际测试、TRAIN质量与Qwen缓存

本阶段冻结源 `71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c`；在先前1fcc实际导出的相同manifest `{sha(P/'export_manifest.json')}` 上继续。服务器70项测试全部通过、0skip（含{validation['actual_torch_cases']}项真实Torch测试），随后独立TRAIN质量与cache均exit0。训练仍是另一个独立阶段，本归档没有读取其结果，不能从这些完成记录宣称方法有效。

## TRAIN正参考与容量

登记96 TRAIN父/288输入，实际95父/285输入；原缺失父及3条不可观察输入不替换。1663条已知正参考全部保留，其中{quality['unknown_reference_types']}条类型unknown；285条件均至少有一条正参考。1663/1663满足固定stride2观察点的逐坐标±.05m末端容量，1663/1663事件分段H24通过原两排tip检查，事件转换数量不一致{quality['event_transition_mismatches']}条。最大最近末端L∞距离{quality['maximum_nearest_endpoint_Linf_m']:.8f}m。

这只证明当前表示能容纳监督末端以及这些已知正参考的H24几何检查通过，不证明普通头能学会、能泛化或能安全执行。正参考长弧全部保留：raw均值{quality['raw_length']['mean_m']:.6f}m、P95 {quality['raw_length']['p95_m']:.6f}m、最长{quality['raw_length']['max_m']:.6f}m，{quality['raw_length']['over3m']}条超过3m。没有按长度/unknown筛选，也没有把未知类别当无效负例。

质量进程PID{quality['job']['pid']}/child{quality['job']['child_pid']}，{quality['job']['start_utc']}至{quality['job']['end_utc']}；外层{quality['job']['process_seconds']:.6f}秒，内部{q['elapsed_seconds']:.6f}秒，GPU小时0。只解码TRAIN原数组；原DEV字节可被完整性哈希读取，未打开其数组作此质量分析。

## 实际Qwen与精确缓存复用

真实官方Qwen3-VL-2B revision/processor `{cache['official_revision']}`，torch2.4.1+cu121/transformers4.57.1/bfloat16/max_pixels262144，冻结参数。只对first32 extension TRAIN的96条RGB＋语言输入实际编码，严格5键观察manifest，没有未来路径/目标/模式进入编码。

原225份NPZ（189 TRAIN＋36 reused DEV）逐容器SHA/数组来源核验后按字节复制；新96份也逐原编码容器SHA复制，合计321。原cache config/index/status的固定SHA、提取器实际源码SHA `{cache['extractor_sha256']}`、新manifest/配置/索引/status、合并receipt及321份NPZ路径/大小/实际SHA均封存。归档时又计算全部321份合并容器和各自old/new origin的SHA，全部相等；不重新编码、不下载这些NPZ。

cache PID{cachejob['pid']}/child{cachejob['child_pid']}，{cachejob['start_utc']}至{cachejob['end_utc']}；完整cache job {cachejob['process_seconds']:.6f}秒（保留GPU小时{cache['outer_cache_job_gpu_hours']:.9f}）。其中新编码子进程{enc['elapsed_seconds']:.6f}秒，编码body含加载{c['total_seconds']:.6f}秒（{cache['encoding_body_gpu_hours']:.9f}GPU小时），加载{c['load_seconds']:.6f}秒；峰值{c['cuda_peak_allocated_bytes']}字节。上述时间嵌套不能相加，原225份的历史编码成本不重复计入；这也不是单请求端到端规划时延。

## 归档与边界

`train_quality.json`保留全部1663参考统计；`QUALITY_SUMMARY.json`为派生摘要。`cache/`保留实际新96编码日志、索引和receipt，`merged_qwen_cache/`仅合并metadata，`CACHE_NPZ_HASH_INDEX.json`保留321个实际容器hash而不复制大文件。原export/readiness9份字节记录保持，后追加quality6份与cache15份原件均单独有remote索引。validation11份原件在独立 `reports/observed_two_row_composite108_validation_v1/`。

旧12DEV仍是reused开发集。新extension32DEV、旧SCORE/CALIBRATION/LOCKED原内容保持封存。未读取正在训练的模型、指标或输出；没有新增生成、搜索或训练。后续模型结果必须由实际完成收据独立归档。
'''
(P/'QUALITY_CACHE_RESULTS.md').write_text(report,encoding='utf-8')
(V/'VALIDATION_RESULTS.md').write_text(f"# Composite108 实际服务器测试\n\n70项通过、0失败、0skip，含{validation['actual_torch_cases']}项真实Torch测试。pytest16.90秒，实际job{validation['job']['process_seconds']:.6f}秒；PID{validation['job']['pid']}/child{validation['job']['child_pid']}。冻结源71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c；原XML/log/status和source全部保留。没有研究数据推理或模型收益声明。\n",encoding='utf-8')
for family in (P,V):
 for name in ('REMOTE_ARTIFACT_INDEX.json','QUALITY_REMOTE_ARTIFACT_INDEX.json','CACHE_REMOTE_ARTIFACT_INDEX.json'):
  if (family/name).exists():
   for filename,value in load(family/name)['files'].items():assert sha(family/filename)==value['sha256']
 files={str(p.relative_to(family)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(family.rglob('*')) if p.is_file() and p.name!='LOCAL_ARTIFACT_INDEX.json'}
 write(family/'LOCAL_ARTIFACT_INDEX.json',dict(files=files,new_forward_requests=0,training_results_opened=False))
print(json.dumps(dict(preparation_files=len(load(P/'LOCAL_ARTIFACT_INDEX.json')['files']),validation_files=len(load(V/'LOCAL_ARTIFACT_INDEX.json')['files']),
 quality_sha256=sha(P/'train_quality.json'),cache_receipt_sha256=sha(P/'merged_qwen_cache/composite_cache_receipt.json'),report_sha256=sha(P/'QUALITY_CACHE_RESULTS.md'),cache_outer_gpu_hours=cache['outer_cache_job_gpu_hours'])))
