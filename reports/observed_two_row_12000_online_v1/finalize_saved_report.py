"""Local saved-artifact report, QA and suggested table rows; no inference."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
F=ROOT/'reports/observed_two_row_12000_online_v1'
V=ROOT/'reports/observed_two_row_online_composite_validation_v1'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
analysis=read(F/'PAIRED_ONLINE_ANALYSIS.json');ids=analysis['arms']['constant']['request_order']
records={arm:{r['id']:r for r in map(json.loads,(F/arm/'requests.jsonl').read_text().splitlines())}
         for arm in ('constant','no_direct')}
qa=[]
for identifier in ids:
 with np.load(F/'constant/requests'/(identifier+'.npz'),allow_pickle=False) as a,\
      np.load(F/'no_direct/requests'/(identifier+'.npz'),allow_pickle=False) as b:
  exact={k:bool(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes())
         for k in ('mean_hidden','last_hidden','input_tokens','current')}
  assert all(exact.values()),identifier
 sums={}
 for arm in ('constant','no_direct'):
  r=records[arm][identifier]
  sums[arm]=sum(r[k] for k in ('generation_ms','prediction_seal_ms','label_and_check_ms'))-r['continuous_request_wall_ms']
  assert abs(sums[arm])<1e-6
 qa.append(dict(id=identifier,cross_arm_input_features_bytes_exact=exact,request_timing_partition_residual_ms=sums))
write(F/'SAVED_ARTIFACT_QA.json',dict(protocol='online12000_saved_artifact_qa_v1',requests_per_arm=36,
  sealed_prediction_npz_count=72,charged_slots=288,all_input_features_cross_arm_exact=True,
  all_request_partitions_sum=True,all_ids_retained=True,new_forward_requests=0,new_searches=0,per_request=qa))

fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
colors={'constant':'#31688e','no_direct':'#35b779'}
for arm in ('constant','no_direct'):
 times=[records[arm][i]['continuous_request_wall_ms'] for i in ids]
 ax[0].plot(range(36),times,'o-',ms=3,label=arm,color=colors[arm])
ax[0].set_yscale('log');ax[0].set_ylabel('Complete request latency (ms, log)')
ax[0].set_title('Actual serial Qwen + RGB-D + K4 + sealed output + checks | all 36 DEV conditions per arm')
ax[0].legend();ax[0].grid(alpha=.25)
delta=[records['no_direct'][i]['continuous_request_wall_ms']-records['constant'][i]['continuous_request_wall_ms'] for i in ids]
ax[1].bar(range(36),delta,color=['#35b779' if d<0 else '#b63679' for d in delta])
ax[1].axhline(0,color='black',lw=.7);ax[1].set_ylabel('no_direct - constant (ms)');ax[1].set_xlabel('Registered parent / target; no omitted requests')
ax[1].set_xticks(range(36));ax[1].set_xticklabels([i.replace('two_row_reach_','').replace('_target',' / ') for i in ids],rotation=90,fontsize=7)
fig.text(.5,.01,'First requests retained. Independent sequential processes, OS cache and initialization confounds; not a causal speedup experiment.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.025,1,1));fig.savefig(F/'ALL_REQUEST_LATENCY.png',dpi=150);plt.close(fig)
write(F/'FIGURE_MANIFEST.json',dict(file='ALL_REQUEST_LATENCY.png',sha256=sha(F/'ALL_REQUEST_LATENCY.png'),
 requests=ids,arms=['constant','no_direct'],all72_requests=True,source='requests.jsonl',new_forward_requests=0))

validation=ET.parse(V/'pytest.xml').getroot();cases=list(validation.iter('testcase'))
assert len(cases)==88 and not list(validation.iter('failure')) and not list(validation.iter('error')) and not list(validation.iter('skipped'))
vs=read(V/'tests.status.json');assert vs['exit_code']==0 and vs['status']=='completed'
write(V/'VALIDATION_SUMMARY.json',dict(tests=88,failures=0,errors=0,skips=0,actual_linux=True,
 elapsed_pytest_seconds=2.28,source_commit=vs['code_commit'],job=vs,pytest_xml_sha256=sha(V/'pytest.xml')))
shutil.copyfile(ROOT/'.bootstrap/archive_online12000.py',F/'archive_and_analyze_saved.py')
shutil.copyfile(ROOT/'.bootstrap/finalize_online12000.py',F/'finalize_saved_report.py')
shutil.copyfile(ROOT/'.bootstrap/no_direct_main269_check.json',F/'MAIN269_READONLY_CHECK.json')

main=read(ROOT/'reports/MAIN_RESULTS.json');suggestions=[]
for arm in ('constant','no_direct'):
 suffix='ordinary_Qwen_RGBD_no_direct_constant12000_best' if arm=='no_direct' else 'ordinary_Qwen_RGBD_constant12000_best'
 matches=[r for r in main if r['method']==suffix]
 if not matches:
  matches=[r for r in main if r.get('run_id')=='observed_two_row_prefix76_convergence_v1/peak_seed0/best']
 assert len(matches)==1,(arm,suffix)
 row=dict(matches[0]);s=read(F/arm/'summary.json');job=read(F/(arm+'.status.json'))
 outer=(datetime.fromisoformat(job['end_utc'])-datetime.fromisoformat(job['start_utc'])).total_seconds()
 row.update(method='ordinary_Qwen_RGBD_'+('constant12000' if arm=='constant' else 'no_direct_constant12000')+'_best_real_online',
   run_id='observed_two_row_12000_online_v1/'+arm,code_commit=job['code_commit'],
   source=f'reports/observed_two_row_12000_online_v1/{arm}/summary.json',
   training_exposures=0,complete_path_state_exposures=0,training_gpu_hours=0,
   common_pretraining_exposures=1536000,common_pretraining_gpu_hours=None,
   elapsed_s=s['total_pipeline_wall_seconds'],gpu_hours=s['gpu_hours_reserved'],
   job_outer_wall_seconds=outer,job_outer_gpu_reserved_hours=outer/3600,
   RGBD_Qwen_generation_ms=s['all_attempted_request_ms_median'],first_request_ms=s['first_attempted_request_ms'],
   request_ms_p95=s['all_attempted_request_ms_p95'],peak_cuda_allocated_bytes=s['peak_cuda_allocated_bytes'],
   candidate_ADE_m=None,endpoint_error_m=None,reference_evaluation_examples=None,
   event_state_accuracy=None,event_sequence_accuracy=s['EventSequenceCorrectAtK'],
   shared_encoding_id=None,shared_encoding_gpu_hours=None,
   checkpoint_selection_protocol='Fixed original best of completed12000; no new checkpoint selection',
   cost_scope='New real-online inference only: each of36 requests reads observation, encodes Qwen, processes RGB-D, generates K4, seals output then checks labels. No warmup/filter/retry. Original training not counted again; loading, metadata and first request preserved. Body and outer job costs nested. Independent sequential processes/OS cache prevent causal speedup claim.',
   budget_status='36 actual new Qwen requests and144 candidate slots in this arm;72/288 for both. All candidates retained.',
   reference_ADE_m=None)
 for key in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','KnownReferenceTypeCoverageAtK',
             'UnknownTypeTipValidCount','DuplicateClassifiedTipValidCount','TipClearAtK','semantic_goal_accuracy','AnySemanticGoalAtK'):
  row[key]=s[key]
 suggestions.append(row)
write(F/'MAIN_ROWS_SUGGESTED.json',dict(note='Suggestions only; MAIN unchanged. Original269 rows must remain exact. Inference-only cost; body and outer are alternatives, not additive.',rows=suggestions))

arms=analysis['arms'];a,b=arms['constant'],arms['no_direct']
jobs={arm:read(F/(arm+'.status.json')) for arm in ('constant','no_direct')}
outer={arm:(datetime.fromisoformat(v['end_utc'])-datetime.fromisoformat(v['start_utc'])).total_seconds() for arm,v in jobs.items()}
report=f'''# 12000步普通两臂：真实Qwen在线结果

两臂各36请求、K4，全部72次Qwen编码/288候选槽真实完成；没有重试、筛选、修复、warmup或新增训练。所有逐候选判定与各自原best保存池一致；全部72请求的mean/last隐藏特征及token为精确字节一致。去direct分支未显示常规请求速度优势，也不改变其已有质量下降结论。

源 `1fccf4898bfa17233f92e20476adeb8db12c6b60`。先实际Linux88测试全部通过、0skip（2.28秒），再两次metadata preflight成功，随后root分别启动真实在线评估。constant原best5500，no_direct原best3500；原训练均12000步/48次DEV选择，不是相同checkpoint步数。没有重新选模。所有角色仍是反复开发过的12父36个DEV条件，未读取新extension32DEV或locked。

|实际量|constant|no_direct|
|---|---:|---:|
|请求/候选槽|36/144|36/144|
|TipValid|59/144 (40.97%)|52/144 (36.11%)|
|AnyTipValid|30/36|27/36|
|已分类不同有效路线均值|.833333|.861111|
|已知正参考类型覆盖|.0587963|.0092593|
|语义目标正确|126/144|117/144|
|完整请求中位(ms)|{a['attempted_request_ms']['median']:.4f}|{b['attempted_request_ms']['median']:.4f}|
|完整请求P95(ms)|{a['attempted_request_ms']['p95']:.4f}|{b['attempted_request_ms']['p95']:.4f}|
|全部36请求均值(ms)|{a['attempted_request_ms']['mean']:.4f}|{b['attempted_request_ms']['mean']:.4f}|
|首请求(ms，未删除)|{a['first_request_ms']:.4f}|{b['first_request_ms']:.4f}|
|模型加载(s)|{a['model_loading_seconds']:.6f}|{b['model_loading_seconds']:.6f}|
|metadata启动(s)|{a['metadata_startup_seconds']:.6f}|{b['metadata_startup_seconds']:.6f}|
|实际body总耗时(s)|{a['total_pipeline_wall_seconds']:.6f}|{b['total_pipeline_wall_seconds']:.6f}|
|外层独立job耗时(s)|{outer['constant']:.6f}|{outer['no_direct']:.6f}|
|body保留GPU小时|{a['gpu_hours_reserved']:.9f}|{b['gpu_hours_reserved']:.9f}|
|显存峰值(GiB)|{a['peak_cuda_allocated_bytes']/2**30:.6f}|{b['peak_cuda_allocated_bytes']/2**30:.6f}|

完整请求含读取/hash、反投影、processor、真实Qwen、几何编码和路线头、GPU结果传输、预测NPZ封存、该条标签IO和原两排checker。连续wall没有把这些步骤遗漏；阶段明细在配对JSON，geometry encoder是geometry/head中的子项，不重复求和。无学习评分器，SelectedValid及完整机器人执行均未测/null。

独立进程先constant再no_direct，模型加载9.63→2.65秒、首请求927→572ms的差异受到文件页缓存/初始化影响，不能当架构带来的因果加速。完整请求中位只差{b['attempted_request_ms']['median']-a['attempted_request_ms']['median']:.4f}ms，P95反而更高。去首请求后的35请求仅描述性补充（主统计保留首请求）：均值{a['after_first_descriptive_ms']['mean']:.4f}/{b['after_first_descriptive_ms']['mean']:.4f}ms；不存在省略冷启动后的主结果替换。

## 完整性与成本

72个逐请求NPZ及seal全部核验，聚合pool与逐请求数组含NaN语义逐值一致，本次288槽均finite、无generation error；72个请求的generation+seal+check严格相加回连续wall。两臂同一输入的current/mean/last/token也逐字节相同。最大在线与原batch保存路线坐标差为{a['maximum_xyz_difference_from_original_m']:.10g}/{b['maximum_xyz_difference_from_original_m']:.10g}m，288槽六个决定字段完全保持；未重跑或修改原输出。

constant job PID{jobs['constant']['pid']}/child{jobs['constant']['child_pid']}，{jobs['constant']['start_utc']}至{jobs['constant']['end_utc']}；no_direct PID{jobs['no_direct']['pid']}/child{jobs['no_direct']['child_pid']}，{jobs['no_direct']['start_utc']}至{jobs['no_direct']['end_utc']}。body总GPU小时{a['gpu_hours_reserved']+b['gpu_hours_reserved']:.9f}，外层job合计{sum(outer.values())/3600:.9f}；二者嵌套不可相加。原训练、原Qwen缓存、validation测试和metadata preflight费用不混为新在线body费用。

本地只读归档193项原件，validation另7项；全部原始字节SHA在各 `REMOTE_ARTIFACT_INDEX.json`，NPZ被普通Git忽略但仍在本地和服务器，72seal与索引进入Git。checkpoint仅引用原hash，不重复下载/存基础模型。`PAIRED_ONLINE_ANALYSIS.json`按ID配对全部36条件与12父，`SAVED_ARTIFACT_QA.json`保存核验，`ALL_REQUEST_LATENCY.png`展示所有请求。归档/分析新增forward、搜索和训练均为0。

MAIN269两条no_direct训练行已与原summary逐字段复核，成本只best计、last为0。best已知参考覆盖明显下降，不能把Unique跨条件总数+1称全面覆盖改善；682845相对1231965减少549120参数、初始预测不同仍是容量混杂。`MAIN_ROWS_SUGGESTED.json`只提供两条新在线成本行供root审核，没有修改MAIN或历史269行。

结论保留：这是完整成本和一致性补证，不是新方法/独立泛化/执行成功；停止去分支继续扩展的既定决定不变。
'''
(F/'ONLINE_RESULTS.md').write_text(report,encoding='utf-8')
for family in (F,V):
 files={str(p.relative_to(family)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(family.rglob('*')) if p.is_file() and p.name!='LOCAL_ARTIFACT_INDEX.json'}
 write(family/'LOCAL_ARTIFACT_INDEX.json',dict(files=files,new_forward_requests=0))
print(json.dumps({'online_files':len(read(F/'LOCAL_ARTIFACT_INDEX.json')['files']),
 'validation_files':len(read(V/'LOCAL_ARTIFACT_INDEX.json')['files']),
 'report_sha256':sha(F/'ONLINE_RESULTS.md'),'all_72_requests_QA_passed':True}))
