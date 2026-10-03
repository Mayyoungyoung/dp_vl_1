"""Prepared offline analysis/contact sheets of a completed, verified archive.

Never reads the remote corpus and never invokes a model, planner or simulator.
Contact sheets are marked not viewed; a human/agent must actually inspect them.
"""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
DEST=ROOT/'reports/observed_two_row_extension128_quality_v1'
OLD=ROOT/'reports/observed_two_row_extension64_quality_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def index_of(row):return int(row['parent_id'].split('_')[-1])-400000


def distribution(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return dict(count=0,min=None,median=None,mean=None,p90=None,p95=None,max=None)
    if not np.isfinite(values).all():raise ValueError('nonfinite accepted-path diagnostic')
    return dict(count=len(values),min=float(values.min()),median=float(np.median(values)),mean=float(values.mean()),
        p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)),max=float(values.max()))


def group(analysis,slots,lo,hi):
    selected=[r for r in slots if lo<=index_of(r)<hi]
    conditions=[r for r in analysis['conditions'] if lo<=index_of(r)<hi]
    parents=[r for r in analysis['parents'] if lo<=r['index']<hi]
    if (len(selected)!=27*(hi-lo) or len(conditions)!=3*(hi-lo) or len(parents)!=hi-lo):
        raise ValueError('fixed all-slot denominator missing')
    accepted=[r for r in selected if r['status']=='accepted']
    failed=[r for r in selected if r['status']=='failed']
    lengths=[r['recomputed_fields']['length_m'] for r in accepted]
    endpoints=[r['recomputed_fields']['endpoint_error_m'] for r in accepted]
    instructions=collections.Counter()
    for i in range(lo,hi):
        path=DEST/'corpus/parents'/('two_row_reach_'+str(400000+i))/'observations.jsonl'
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['split']!='TRAIN' or row['parent_id']!='two_row_reach_'+str(400000+i):raise ValueError('nonselected role or parent')
            instructions[row['instruction']]+=1
    unknown=sum(r['actual_accepted_type'] is None for r in accepted)
    return dict(parents=hi-lo,conditions=len(conditions),slots=len(selected),
        statuses=dict(collections.Counter(r['status'] for r in selected)),accepted=len(accepted),failed=len(failed),
        other_outcomes=len(selected)-len(accepted)-len(failed),accepted_rate=len(accepted)/len(selected),
        known=len(accepted)-unknown,unknown=unknown,unknown_fraction=unknown/len(accepted) if accepted else None,
        R_gt4=sum(c['has_more_than_K4_known_types'] for c in conditions),
        R_histogram=dict(collections.Counter(c['distinct_known_types'] for c in conditions)),
        known_duplicate_references=sum(c['known_duplicate_references'] for c in conditions),
        zero_known_positive_conditions=[dict(id=c['id'],positive_references=c['valid_references'],
            unknown_references=c['unknown_valid_references']) for c in conditions if c['distinct_known_types']==0],
        min_refs=min(c['valid_references'] for c in conditions),max_refs=max(c['valid_references'] for c in conditions),
        length_m=dict(distribution(lengths),gt2=sum(x>2 for x in lengths),gt3=sum(x>3 for x in lengths),gt4=sum(x>4 for x in lengths)),
        endpoint_error_m=distribution(endpoints),
        longest=sorted([dict(id=r['input_id'],attempt=r['attempt'],length_m=r['recomputed_fields']['length_m'],trace_sha256=r['trace_sha256'])
            for r in accepted],key=lambda r:r['length_m'],reverse=True)[:10],
        failure_errors=dict(collections.Counter(r.get('error') for r in failed)),
        failure_predicates_nonexclusive=dict(robot_collision=sum(r.get('collision_pair') is not None for r in failed),
            raw_tip=sum(r.get('recomputed_fields',{}).get('tip_polyline_clear') is False for r in failed),
            h24_tip=sum(r.get('recomputed_fields',{}).get('tip_polyline_24_clear') is False for r in failed),
            type_changed=sum('recomputed_fields' in r and r['recomputed_fields'].get('actual_route_type')!=r['recomputed_fields'].get('h24_route_type') for r in failed),
            endpoint_gt3cm=sum(r.get('recomputed_fields',{}).get('endpoint_error_m',0)>.03 for r in failed)),
        strict_restore_passes=sum(bool(r.get('strict_restore_passed')) for r in selected),
        event_transition_counts=dict(collections.Counter(r.get('event_transitions') for r in selected)),
        explicit_get_path_calls=sum(r.get('planning_calls',0) for r in selected),
        planning_seconds=sum(r.get('planning_seconds',0) for r in selected),
        simulation_seconds=sum(r.get('simulation_seconds',0) for r in selected),
        worker_seconds=sum(p['closure']['worker_elapsed_seconds'] or 0 for p in parents),
        unknown_worker_seconds=sum(p['closure']['worker_elapsed_seconds'] is None for p in parents),
        missing_observation_ids=[x for p in parents for x in p['missing_observation_ids']],
        model_use_blocked=[p['parent_id'] for p in parents if p['model_use_blocked']],
        instruction_histogram=dict(sorted(instructions.items())))


def font(size):
    for path in (Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if path.is_file():return ImageFont.truetype(str(path),size)
    return ImageFont.load_default()


def contact_sheets():
    folder=DEST/'qa_views'
    if folder.exists():raise FileExistsError('preserve existing visual review sheets')
    folder.mkdir()
    records=[]
    for i in range(64,128):
        parent='two_row_reach_'+str(400000+i)
        sheet=Image.new('RGB',(2770,1450),'white');draw=ImageDraw.Draw(sheet)
        sources=[]
        for target in range(3):
            source=DEST/'analysis_run/train128'/parent/('target%d_all9.png'%target)
            with Image.open(source) as original:
                image=original.convert('RGB');original_size=list(image.size)
            # Half-resolution nine-slot plot: 455x216 pixels per route panel.
            image.thumbnail((1365,650),Image.Resampling.LANCZOS)
            x=20+(target%2)*1380;y=20+(target//2)*715
            draw.text((x,y),parent+' target'+str(target)+' | ALL 9 requested slots',fill='black',font=font(22))
            sheet.paste(image,(x,y+35))
            sources.append(dict(path=source.relative_to(DEST).as_posix(),sha256=sha(source),
                original_pixels=original_size,display_pixels=list(image.size),kind='all9_route_plot',target=target))
        source=DEST/'analysis_run/train128'/parent/'front.png'
        with Image.open(source) as image:front=image.convert('RGB');original_size=list(front.size)
        sheet.paste(front.resize((448,448),Image.Resampling.NEAREST),(1420,820))
        draw.text((1420,745),parent+' | recorded front (224px, 2x nearest)',fill='black',font=font(22))
        draw.text((1420,1290),'No filtering: accepted, failed, unknown all retained.',fill='black',font=font(20))
        draw.text((1420,1325),'Full-resolution original route plots remain archived.',fill='black',font=font(20))
        draw.text((1420,1360),'Visual inspection pending; no automatic QA pass.',fill='black',font=font(20))
        sources.append(dict(path=source.relative_to(DEST).as_posix(),sha256=sha(source),original_pixels=original_size,
            display_pixels=[448,448],kind='front',resampling='nearest'))
        output=folder/(parent+'_all3_targets_front.png');sheet.save(output)
        records.append(dict(path=output.relative_to(DEST).as_posix(),sha256=sha(output),pixels=list(sheet.size),
            parent=parent,sources=sources,viewed=False,observations=[]))
    front_pages=[]
    for page in range(4):
        sheet=Image.new('RGB',(1024,1064),'white');draw=ImageDraw.Draw(sheet);sources=[]
        for j in range(16):
            i=64+16*page+j;parent='two_row_reach_'+str(400000+i)
            source=DEST/'analysis_run/train128'/parent/'front.png'
            with Image.open(source) as image:front=image.convert('RGB')
            if front.size!=(224,224):raise ValueError('original front shape changed; do not silently resize')
            x=(j%4)*256;y=(j//4)*266
            draw.text((x+8,y+4),parent,fill='black',font=font(17));sheet.paste(front,(x+16,y+30))
            sources.append(dict(parent=parent,path=source.relative_to(DEST).as_posix(),sha256=sha(source)))
        output=folder/('new64_fronts_original_page%d.png'%(page+1));sheet.save(output)
        front_pages.append(dict(path=output.relative_to(DEST).as_posix(),sha256=sha(output),sources=sources,viewed=False,observations=[]))
    result=dict(protocol='extension128_complete_contact_sheet_inventory_v1',scope='new TRAIN indices64..127 only',
        unique_new_parent_count=64,unique_route_plots=192,unique_fronts=64,all_slots_per_condition=9,
        parent_sheets=records,front_pages=front_pages,visual_review_completed=False,
        prior64_images=256,prior64_byte_identity_receipt='SYNC_RECEIPT.json',
        prior64_actual_qa_reference='../observed_two_row_extension64_quality_v1/VISUAL_QA.json',
        note='Inventory proves source coverage, not visual inspection. Reviewer must actually open every assigned sheet and record findings.')
    write(DEST/'VISUAL_QA_PENDING.json',result)
    return result


def draft_report(analysis,groups,sync):
    status=read(DEST/'analysis_run/train128.status.json')
    def dt(value):return datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    job_seconds=(dt(status['end_utc'])-dt(status['start_utc'])).total_seconds()
    rows=[]
    for label,name in [('旧64','old64'),('新增64','added64'),('累计128','all128')]:
        g=groups[name]
        rows.append('| %s | %d / %d / %d | %d / %d / %d | %d / %d | %d |'%(label,g['parents'],g['conditions'],g['slots'],
            g['accepted'],g['failed'],g['other_outcomes'],g['known'],g['unknown'],g['R_gt4']))
    report='''# TRAIN128 质量归档结果草稿

全128注册TRAIN父的已完成结果已逐SHA同步；新增64父仍需完成实际视觉复核，此草稿不声明QA通过，也不自动授权下一采集或模型扩训。

| 范围 | 父 / 指令 / 请求槽 | 接受 / 失败 / 其他 | 已知 / unknown正参考 | 已知R>K4条件 |
|---|---:|---:|---:|---:|
TABLE

统计来自完整3456请求槽，没有按成功率删除失败、unknown或长路线。R只指采到的已分类正参考类型，不能解释为真实全部可行解数量。失败谓词是重叠诊断，不能相加为失败总数。此处没有新forward、search或仿真。

新增与累计的长度、端点误差、颜色频数、事件、严格恢复、失败谓词和逐父缺失见 `TRAIN128_SUPPLEMENT.json`。本轮只审采集和既有检查，不证明训练器H24重采样容量、完整连续全臂安全或任何方法优势；不改变当前已冻结composite108训练人口。

`qa_views/`包含新增64父全部192张全九槽图及64张front的64张父联系表，另有4页原分辨率front。`VISUAL_QA_PENDING.json`逐图记录源SHA和实际呈现大小；全部初始viewed=false，等待负责人分配实际复核。前64共256图已与原64归档逐字节相同，旧QA引用不冒称重新目视。

分析源码1a3eef1fb12d55e98d4d188a091ea40ea62c0a02；采集源码5c8f8e4f5cd478c793a0e0d9640005deaf700973。本次分析源码SHA与结果JSON SHA分开记录：
- analysis.json文件SHA：ANALYSIS_SHA
- analysis_source_sha256：SOURCE_SHA
- 主体分析耗时 BODY_S 秒；外层作业耗时 JOB_S 秒，二者嵌套、不能相加。GPU小时0。
- 归档 ORIGINAL_FILES 个原件、ORIGINAL_BYTES 字节，全部源SHA核验。实际wrapper、job状态/日志、注册/closure和全部128父status/log均保留。

大于10MiB的all_requested_slots.json在服务器和ignored local保留，索引含真实恢复路径/SHA；不进入普通Git，不从统计中移除。未读取extension新DEV或旧SCORE/CALIBRATION/LOCKED原始内容。最终报告必须补入实际视觉结论后再去掉“草稿”。
'''
    replacements={'TABLE':'\n'.join(rows),'ANALYSIS_SHA':sha(DEST/'analysis_run/train128/analysis.json'),
        'SOURCE_SHA':analysis['analysis_source_sha256'],'BODY_S':'%.6f'%analysis['analysis_elapsed_seconds'],
        'JOB_S':'%.6f'%job_seconds,'ORIGINAL_FILES':str(sync['files']),'ORIGINAL_BYTES':str(sync['bytes'])}
    for key,value in replacements.items():report=report.replace(key,value)
    (DEST/'TRAIN128_RESULTS_DRAFT.md').write_text(report,encoding='utf-8')


def local_index():
    original=read(DEST/'SERVER_ARCHIVE_INDEX.json')
    original_paths={row['path'] for row in original['files']}
    for row in original['files']:
        path=DEST/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('source artifact changed: '+row['path'])
    for name in ('archive_extension128_quality.py','extension128_quality_supplement.py'):
        source=ROOT/'.bootstrap'/name;target=DEST/'source'/name
        target.write_bytes(source.read_bytes())
    rows=[]
    for path in sorted(DEST.rglob('*')):
        if not path.is_file() or path.name=='LOCAL_ARTIFACT_INDEX.json':continue
        relative=path.relative_to(DEST).as_posix();size=path.stat().st_size
        storage='ordinary_git'
        if size>10*1024*1024:
            ignored=subprocess.run(['git','check-ignore','--no-index','-q',str(path)],cwd=ROOT).returncode==0
            storage='ignored_large_json' if ignored else 'large_pending_root_ignore_rule'
        rows.append(dict(path=relative,sha256=sha(path),bytes=size,original_server_artifact=relative in original_paths,git_storage=storage))
    result=dict(protocol='extension128_local_artifact_index_v1',files=rows,local_files=len(rows),
        original_files=len(original_paths),original_bytes=sum(r['bytes'] for r in original['files']),
        all_original_sha256_verified=True,visual_qa_pass_not_inferred=True,new_forward=0,new_search=0,new_simulation=0)
    write(DEST/'LOCAL_ARTIFACT_INDEX.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-confirmed-archive-complete',action='store_true',required=True)
    parser.add_argument('--index-only',action='store_true',help='After actual QA/report edits, reverify originals and refresh local hashes only')
    args=parser.parse_args()
    if args.index_only:
        result=local_index();print(json.dumps(dict(local_files=result['local_files'],original_files=result['original_files'])));return
    if (DEST/'TRAIN128_SUPPLEMENT.json').exists():raise FileExistsError('do not overwrite an existing analysis')
    index=read(DEST/'SERVER_ARCHIVE_INDEX.json');sync=read(DEST/'SYNC_RECEIPT.json')
    if not sync['all_server_sha256_verified']:raise ValueError('archive must be verified first')
    for row in index['files']:
        if sha(DEST/row['path'])!=row['sha256']:raise ValueError('original archive changed: '+row['path'])
    analysis=read(DEST/'analysis_run/train128/analysis.json');slots=read(DEST/'analysis_run/train128/all_requested_slots.json')
    if len(slots)!=3456 or analysis['selected_train_indices']!=list(range(128)):raise ValueError('whole TRAIN128 only')
    groups={name:group(analysis,slots,lo,hi) for name,lo,hi in [('old64',0,64),('added64',64,128),('all128',0,128)]}
    old=read(OLD/'TRAIN64_SUPPLEMENT.json')['groups']['all64']
    for key in ('parents','conditions','slots','accepted','failed','known','unknown','R_gt4','strict_restore_passes','explicit_get_path_calls'):
        if groups['old64'][key]!=old[key]:raise ValueError('old64 counts changed: '+key)
    for key in ('accepted','failed','known','unknown','R_gt4','strict_restore_passes','explicit_get_path_calls'):
        if groups['all128'][key]!=groups['old64'][key]+groups['added64'][key]:raise ValueError('group partition mismatch: '+key)
    if groups['all128']['accepted']!=analysis['accepted_reference_count'] or groups['all128']['unknown']!=analysis['accepted_unknown_count']:
        raise ValueError('supplement differs from original full analysis')
    write(DEST/'TRAIN128_SUPPLEMENT.json',dict(groups=groups,analysis_sha256=sha(DEST/'analysis_run/train128/analysis.json'),
        all_slots_sha256=sha(DEST/'analysis_run/train128/all_requested_slots.json'),old64_counts_match=True,
        new_forward=0,new_search=0,new_simulation=0,new_dev_raw_opened=False,visual_qa_pending=True))
    qa=contact_sheets();draft_report(analysis,groups,sync);local_index()
    print(json.dumps(dict(groups=groups,sheets=len(qa['parent_sheets']),front_pages=len(qa['front_pages']),visual_qa_completed=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
