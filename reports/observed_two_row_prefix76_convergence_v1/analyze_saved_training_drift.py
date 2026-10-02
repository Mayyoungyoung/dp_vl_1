"""Inspect sealed predictions and logs only; no model, forward, simulator or raw labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
        p90=float(np.quantile(a,.9)),minimum=float(a.min()),maximum=float(a.max())) if len(a) else None


def main():
    inputs={}
    for name in ('peak_seed0/history.json','stage1500.log','finish.log','peak_seed0/train/per_scene.json',
            'fixed_last_train/per_scene.json','peak_seed0/train/predictions.npz','fixed_last_train/predictions.npz'):
        inputs[name]=sha(ROOT/name)
    history=read(ROOT/'peak_seed0/history.json')
    assert [x['step'] for x in history]==list(range(250,12001,250))
    losses=[]
    for name in ('stage1500.log','finish.log'):
        for line in (ROOT/name).read_text().splitlines():
            if line.startswith('{'):
                row=json.loads(line)
                if 'path_loss' in row:losses.append(dict(row,source_log=name))
    assert len(losses)==120 and [r['step'] for r in losses]==list(range(100,12001,100))
    best=read(ROOT/'peak_seed0/train/per_scene.json');last=read(ROOT/'fixed_last_train/per_scene.json')
    a=np.load(ROOT/'peak_seed0/train/predictions.npz');b=np.load(ROOT/'fixed_last_train/predictions.npz')
    assert np.array_equal(a['scene_ids'],b['scene_ids']) and len(best)==len(last)==189
    assert [r['scene_id'] for r in best]==[r['scene_id'] for r in last]==a['scene_ids'].tolist()
    rows=[];pass_hist={};error_stats={};by_goal={}
    for stage,data in (('best5500',best),('last12000',last)):
        pass_hist[stage]=dict(sorted(Counter(sum(c['semantic_goal_correct'] for c in r['tip_candidates']) for r in data).items()))
        failed=[c['endpoint_error_m'] for r in data for c in r['tip_candidates'] if not c['semantic_goal_correct']]
        error_stats[stage]=dict(all_endpoint_errors_m=stats([c['endpoint_error_m'] for r in data for c in r['tip_candidates']]),
            failed_endpoint_errors_m=stats(failed),failed_le_4cm=sum(v<=.04 for v in failed),
            failed_between_4_and_6cm=sum(.04<v<=.06 for v in failed),failed_gt_6cm=sum(v>.06 for v in failed))
        by_goal[stage]={}
        for goal in range(3):
            candidates=[c for r in data if r['scene_id'].endswith('target'+str(goal)) for c in r['tip_candidates']]
            by_goal[stage][str(goal)]=dict(candidate_slots=len(candidates),semantic_correct=sum(c['semantic_goal_correct'] for c in candidates),
                tip_valid=sum(c['TipValid'] for c in candidates),tip_clear=sum(c['tip_segments_clear'] for c in candidates))
    for i,(x,y) in enumerate(zip(best,last)):
        ca,cb=x['tip_candidates'],y['tip_candidates'];sa=sum(c['semantic_goal_correct'] for c in ca);sb=sum(c['semantic_goal_correct'] for c in cb)
        ea=a['paths'][i,:,-1];eb=b['paths'][i,:,-1];aa=a['learned_surface_anchor'][i];ab=b['learned_surface_anchor'][i]
        rows.append(dict(scene_id=x['scene_id'],parent_id=x['parent_id'],goal_index=int(x['scene_id'][-1]),
            best_semantic_correct=sa,last_semantic_correct=sb,group='lost' if sb<sa else 'gained' if sb>sa else 'same',
            best_tip_valid=sum(c['TipValid'] for c in ca),last_tip_valid=sum(c['TipValid'] for c in cb),
            best_endpoint_errors_m=[c['endpoint_error_m'] for c in ca],last_endpoint_errors_m=[c['endpoint_error_m'] for c in cb],
            best_anchor_xyz=aa.tolist(),last_anchor_xyz=ab.tolist(),anchor_shift_m=float(np.linalg.norm(ab-aa)),
            anchor_delta_xyz=(ab-aa).tolist(),endpoint_shift_m=np.linalg.norm(eb-ea,axis=-1).tolist(),
            best_endpoint_offset_from_anchor_m=np.linalg.norm(ea-aa,axis=-1).tolist(),
            last_endpoint_offset_from_anchor_m=np.linalg.norm(eb-ab,axis=-1).tolist()))
    group_stats={group:dict(conditions=sum(r['group']==group for r in rows),
        anchor_shift_m=stats([r['anchor_shift_m'] for r in rows if r['group']==group])) for group in ('lost','gained','same')}
    compact_history=[dict(step=r['step'],mean_recent_total_training_loss=r['loss'],
        dev_semantic=r['dev_model']['semantic_goal_accuracy'],dev_tip_valid=r['dev_model']['TipValidAtK'],
        dev_unique=r['dev_model']['UniqueClassifiedTipValidAtK'],dev_endpoint_error_m=r['dev_model']['candidate_endpoint_error_m']) for r in history]
    result=dict(protocol='sealed_train_pool_drift_v1',best_step=5500,fixed_last_step=12000,inputs_sha256=inputs,
        history=compact_history,all_logged_recent_losses=losses,semantic_correct_count_histogram=pass_hist,
        by_goal=by_goal,error_stats=error_stats,condition_groups=group_stats,
        all_four_lost=[r['scene_id'] for r in rows if r['best_semantic_correct']==4 and r['last_semantic_correct']==0],
        conditions=rows,new_forward_requests=0,new_optimizer_updates=0,raw_labels_opened=False,
        scope='Saved shared-anchor and endpoint movement is an association, not a causal LR/attention-gradient diagnosis. '
            'Only best5500 and last12000 have TRAIN predictions; intervening history metrics are DEV, not TRAIN. '
            'Recent100-batch training losses are not full-TRAIN loss or the final single-batch loss. '
            'Semantic correctness here includes the original3cm endpoint threshold; a miss does not alone prove wrong object identity.')
    (ROOT/'TRAINING_DRIFT_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axes[0].plot([r['step'] for r in losses],[r['path_loss'] for r in losses],label='Path loss (last100 minibatches)')
    axes[0].set_yscale('log');axes[0].set_ylabel('Path loss');axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].plot([r['step'] for r in compact_history],[r['dev_semantic'] for r in compact_history],label='DEV endpoint semantic')
    axes[1].plot([r['step'] for r in compact_history],[r['dev_tip_valid'] for r in compact_history],label='DEV TipValid@4')
    for ax in axes:ax.axvline(5500,color='gray',linestyle='--',alpha=.7)
    axes[1].set_ylabel('Rate');axes[1].set_xlabel('Training step');axes[1].set_ylim(0,1);axes[1].legend();axes[1].grid(alpha=.25)
    fig.suptitle('All saved history: ordinary prefix76, fixed12000 steps; dashed=original best5500')
    fig.tight_layout();fig.savefig(ROOT/'training_history.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for stage,data in (('best5500',best),('last12000',last)):
        errors=np.array([c['endpoint_error_m'] for r in data for c in r['tip_candidates']])*100
        axes[0].hist(errors,bins=np.linspace(0,max(error_stats[s]['all_endpoint_errors_m']['maximum'] for s in error_stats)*100,25),
            histtype='step',linewidth=1.5,label=stage)
    axes[0].axvline(3,color='k',linestyle='--');axes[0].set_xlabel('Endpoint error (cm), all756 slots');axes[0].set_ylabel('Count');axes[0].legend()
    for group,color in [('same','gray'),('lost','tab:red'),('gained','tab:blue')]:
        rr=[r for r in rows if r['group']==group]
        axes[1].scatter([r['anchor_shift_m']*100 for r in rr],
            [(np.mean(r['last_endpoint_errors_m'])-np.mean(r['best_endpoint_errors_m']))*100 for r in rr],
            s=16,alpha=.65,c=color,label=group)
    axes[1].set_xlabel('Saved shared-anchor shift (cm)');axes[1].set_ylabel('Mean endpoint error change (cm)');axes[1].legend()
    fig.suptitle('All189 TRAIN conditions: descriptive best-to-last comparison, no new predictions')
    fig.tight_layout();fig.savefig(ROOT/'training_endpoint_drift.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(groups=group_stats,error_stats=error_stats,loss_points=len(losses),history_points=len(history))))


if __name__=='__main__':main()
