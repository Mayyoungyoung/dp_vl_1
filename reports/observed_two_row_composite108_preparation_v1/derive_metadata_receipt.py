"""Derive an export-only receipt from already archived metadata; no corpus IO."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
F=ROOT/'reports/observed_two_row_composite108_preparation_v1'
OLD=ROOT/'reports/observed_two_row_prefix76_preparation_v1'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
m=load(F/'export_manifest.json');old=load(OLD/'export_manifest.json');oldq=load(OLD/'quality_summary.json')
assert sha(OLD/'export_manifest.json')==m['historical_export_manifest_sha256']=='309966e192a3c582cde503151e607b5ede679eaec596bd8ab2bc1fd06ffdbab1'
assert oldq['source_manifest_sha256']==m['historical_export_manifest_sha256']
assert m['requested_parents']==108 and m['requested_train_parents']==96 and m['requested_reused_dev_parents']==12
assert m['actual_inputs_by_role']==dict(TRAIN=285,DEV_MODEL=36)
assert m['actual_inputs']==321 and m['positive_references']==1868 and m['unobserved_requested_inputs']==3
assert len(m['sources']['old']['selected_indices'])==76 and len(m['sources']['extension']['selected_indices'])==32
prefix=[]
for name,proof in m['original_bytes_preserved'].items():
 assert proof['old_sha256']==old['output_files_sha256'][name]
 prefix.append(dict(file=name,**proof,historical_output_sha256_matches=True,
                    proof_origin='Actual frozen exporter verifies full historical bytes as exact prefix; this local check matches its receipt to original archived manifest. Corpus JSONL not copied or re-opened.'))
assert [p['old_rows'] for p in prefix]==[225,225,2025]
assert [p['appended_rows'] for p in prefix]==[96,96,864]
for name in ('mechanical_gate_at_export','mechanical_gate_after_raw_validation'):
 gate=m[name];assert gate['ready'] and not any(gate[k] for k in ('missing','selected_blocked','blocked_members','duplicate_groups'))
 assert not gate['extension_dev_raw_opened'] and not gate['old_reserved_raw_opened']
assert m['exporter_sha256']==sha(F/'source/scripts/export_two_row_composite_observations.py')
assert m['reused_reader_sha256']==old['exporter_sha256']
assert not m['training_authorized'] and not m['cache_generated']
dev_refs=oldq['roles']['DEV_MODEL']['positive_references'];train_refs=m['positive_references']-dev_refs
assert (train_refs,dev_refs)==(1663,205)
cost={}
for stage in ('readiness','export'):
 s=load(F/(stage+'.status.json'));assert s['status']=='completed' and s['exit_code']==0
 cost[stage]=dict(pid=s['pid'],child_pid=s['child_pid'],start_utc=s['start_utc'],end_utc=s['end_utc'],
  process_seconds=(datetime.fromisoformat(s['end_utc'])-datetime.fromisoformat(s['start_utc'])).total_seconds(),exit_code=0)
proof=dict(protocol='composite108_archived_prefix_proof_v1',actual_export_manifest_sha256=sha(F/'export_manifest.json'),
 historical_export_manifest_sha256=sha(OLD/'export_manifest.json'),prefix_files=prefix,
 metadata_only_local_recheck=True,raw_corpus_opened_by_this_analysis=False,corpus_rows_copied=False,
 new_forward_requests=0,quality_evidence_for_new_export=False,cache_reuse_performed=False,training_performed=False)
write(F/'HISTORICAL_PREFIX_PROOF.json',proof)
summary=dict(protocol='composite108_export_preparation_summary_v1',source_commit=load(F/'export.status.json')['code_commit'],
 requested_parents=108,selected_mechanically_closed_parents=108,requested_train_parents=96,requested_reused_dev_parents=12,
 observed_train_parents=95,observed_total_parents=107,requested_inputs=324,actual_inputs=321,
 actual_train_inputs=285,actual_reused_dev_inputs=36,new_extension_train_inputs=96,
 positive_references=1868,train_positive_references=1663,reused_dev_positive_references=205,
 reference_role_count_derivation='Unchanged historical DEV205 from old sealed metadata;1868-205=1663 TRAIN. No new supervision rows decoded.',
 new_extension_positive_references=1868-old['positive_references'],
 requested_route_proposals=2916,actual_attempt_records=2889,unattempted_proposals=27,
 missing_original_parent='two_row_reach_283220',unobserved_requested_inputs=3,failed_parent_replacements=0,
 registered_hash_scan_parents=404,closed_hash_metadata_parents=sum(len(s['mechanical_closure_files_sha256']) for s in m['sources'].values()),
 selected_duplicate_groups=[],mechanical_gate_at_export=m['mechanical_gate_at_export'],
 mechanical_gate_after_validation=m['mechanical_gate_after_raw_validation'],
 original_byte_prefix_proof='HISTORICAL_PREFIX_PROOF.json',source_files_hash_entries=len(m['source_files_sha256']),
 exported_files_sha256=m['output_files_sha256'],export_manifest_sha256=sha(F/'export_manifest.json'),
 sources=m['sources'],stage_cost=cost,gpu_hours=0,
 boundary='Export/closed mechanical gate only. No claim that the new TRAIN capacity, Qwen caching or training stages passed. New extension DEV and old reserved raw remain sealed.')
write(F/'PREPARATION_SUMMARY.json',summary)
shutil.copyfile(ROOT/'.bootstrap/sync_composite108_preparation.py',F/'archive_metadata_only.py')
shutil.copyfile(ROOT/'.bootstrap/finalize_composite108_preparation.py',F/'derive_metadata_receipt.py')
report=f'''# Composite108：实际导出准备记录

冻结源 `1fccf4898bfa17233f92e20476adeb8db12c6b60` 的readiness和export均真实exit0。所选108个父（旧76＋新32）全部机械闭合，导出321条实际观察、1868条已验证正参考。本记录仅证明闭合与导出；**不证明新组合数据的TRAIN容量/质量审计、Qwen缓存或训练已经通过**。

|范围|请求父|实际有观察父|实际输入|正参考|
|---|---:|---:|---:|---:|
|旧64 TRAIN＋新32 TRAIN|96|95|285|1663|
|原12 reused DEV|12|12|36|205|
|合计|108|107|321|1868|

原缺失父 `two_row_reach_283220` 仍保留，未用成功父替换。请求324输入中3条不可观察；2916个登记路线提案中2889有实际attempt记录、27未尝试。不能把108机械闭合说成108个成功场景。全部已验证正例按原规则保留，unknown类型和长弧不因本次导出删除；本记录不新增模型能力结论。

实际新增部分只有first32 extension TRAIN：96条输入、555条参考。没有导出/编码extension新32DEV。原12DEV是反复使用的开发集，绝不改称独立测试或OOD。角色参考数量来自原DEV205及其完整字节保留证明，与新总数1868相减得到TRAIN1663；本归档未另开或复制SUP内容。

## 历史字节与来源

exporter在实际导出中完整核验原文件是新文件的精确字节前缀，本地又将证明内SHA与原prefix76封存manifest逐项对照：

|文件|原行数|原字节数|追加行数|
|---|---:|---:|---:|
|observations.jsonl|225|74062|96|
|supervision.jsonl|225|481653|96|
|attempts.jsonl|2025|8559303|864|

各原字节SHA、原manifest SHA、输出SHA和source SHA完整保存在 `HISTORICAL_PREFIX_PROOF.json` 与原 `export_manifest.json`。此处证明的是观察/监督/attempt文件前缀，不是225个Qwen NPZ缓存已复用；后者必须由后续缓存阶段独立完成。

实际跨源门禁检查旧116＋新288共404份注册计划的exact/1mm几何哈希，以及当前148份已闭合actual1mm机械记录；所选108父无missing/blocked/duplicate，raw校验前后均通过。已封存角色仅提供机械哈希，不打开其raw。没有把布局哈希通过当作语义/路线质量认证。

新manifest SHA `{summary['export_manifest_sha256']}`；旧manifest `{m['historical_export_manifest_sha256']}`；原reader `{m['reused_reader_sha256']}`；本次exporter `{m['exporter_sha256']}`。manifest含3097项来源哈希和4个导出文件哈希，完整保留。

## 实际运行与归档边界

readiness PID{cost['readiness']['pid']}/child{cost['readiness']['child_pid']}，{cost['readiness']['start_utc']}至{cost['readiness']['end_utc']}，{cost['readiness']['process_seconds']:.6f}秒。export PID{cost['export']['pid']}/child{cost['export']['child_pid']}，{cost['export']['start_utc']}至{cost['export']['end_utc']}，{cost['export']['process_seconds']:.6f}秒。wrapper隐藏GPU、线程环境均1，GPU小时0；未作模型forward或搜索。

服务器数据目录 `/home/wzy/dpvlm/route_set_v1/data/observation_two_row_composite108_v1`。9份原metadata/log/status/source/wrapper逐字节归档，索引为 `REMOTE_ARTIFACT_INDEX.json`，archive SHA为 `{load(F/'SYNC_RECEIPT.json')['archive_sha256']}`。未下载图片、depth、轨迹NPZ或完整observations/supervision/attempts语料文件，未改变既有online归档或global文档。

原selection中的“prospective/no export”文字是注册时状态，保留原始字节；本次完成状态以实际job收据和manifest创建时间为准。后续质量/cache/train须各自真实收据，不能从这次成功导出推断。
'''
(F/'PREPARATION_RESULTS.md').write_text(report,encoding='utf-8')
index={str(p.relative_to(F)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size)
       for p in sorted(F.rglob('*')) if p.is_file() and p.name!='LOCAL_ARTIFACT_INDEX.json'}
write(F/'LOCAL_ARTIFACT_INDEX.json',dict(files=index,raw_corpus_opened=False,new_forward_requests=0))
print(json.dumps(dict(files=len(index),manifest_sha256=sha(F/'export_manifest.json'),report_sha256=sha(F/'PREPARATION_RESULTS.md'),stage_cost=cost)))
