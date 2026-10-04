"""Three-seed paired parent analysis of saved fixed-q candidate pools."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def metrics(row):
    c=row['candidates'];k=row['K']
    return dict(CandidateValidRate=k['8']['ValidCount']/8,
        CollisionFailureRate=np.mean([not x['tip_segments_clear'] for x in c]),
        GoalFailureRate=np.mean([not x['semantic_goal_correct'] for x in c]),
        EventFailureRate=np.mean([not x['event_state_sequence_correct'] for x in c]),
        StartFailureRate=np.mean([not x['starts_at_current_state'] for x in c]),
        EndpointErrorM=np.mean([x['endpoint_error_m'] for x in c]),
        PathLengthM=np.mean([x['length_m'] for x in c]),
        AnyValid8=float(k['8']['ValidCount']>0),ValidCount8=k['8']['ValidCount'],
        GeometricModeCount8=k['8']['GeometricModeCount'],
        DuplicateValidCount8=k['8']['ValidCount']-k['8']['GeometricModeCount'],
        TwoDistinctValid4=float(k['4']['TwoDistinctValid']),
        TwoDistinctValid2=float(k['2']['TwoDistinctValid']),
        ReferenceModeCoverage8=k['8']['ReferenceModeCoverage'],
        ReferenceModeCoverage4=k['4']['ReferenceModeCoverage'],
        SelectedValid1=k['1']['ValidCount'],UnknownValidCount=row['unknown_valid'])


def analyze(root):
    results={};all_rows={}
    for split in ('old_dev','dev32'):
        rr={(a,s):read(root/'evaluation'/split/('%s_seed%d'%(a,s))/'rows.json') for a in ('A','B') for s in range(3)}
        all_rows[split]=rr
        ids=[r['id'] for r in rr['A',0]];parents=sorted({r['parent'] for r in rr['A',0]})
        for rows in rr.values():assert [r['id'] for r in rows]==ids
        keys=list(metrics(rr['A',0][0]));arrays={k:np.array([[list(metrics(r).values()) for r in rows]])[0] for k,rows in rr.items()}
        out={'requests':len(ids),'parents':len(parents),'seeds':{},'three_seed_means':{},'paired_parent_bootstrap':{}}
        for a in ('A','B'):
            out['seeds'][a]=[{k:float(v) for k,v in zip(keys,arrays[a,s].mean(0))} for s in range(3)]
            out['three_seed_means'][a]={k:float(np.mean([out['seeds'][a][s][k] for s in range(3)])) for k in keys}
        delta=np.mean([arrays['B',s]-arrays['A',s] for s in range(3)],axis=0)
        pd=np.array([delta[[i for i,r in enumerate(rr['A',0]) if r['parent']==p]].mean(0) for p in parents])
        rng=np.random.default_rng(20261004);draw=rng.integers(len(parents),size=(10000,len(parents)))
        ci=np.percentile(pd[draw].mean(1),[2.5,97.5],axis=0)
        for j,k in enumerate(keys):out['paired_parent_bootstrap'][k]=dict(delta=float(pd[:,j].mean()),CI95=ci[:,j].tolist())
        # Query IDs have no stable semantics: compare sets of valid geometric words.
        transitions=[]
        for s in range(3):
            for a,b in zip(rr['A',s],rr['B',s]):
                aw={w for w in a['words'] if w is not None};bw={w for w in b['words'] if w is not None}
                transitions.append(dict(seed=s,id=a['id'],new_valid_modes=len(bw-aw),lost_valid_modes=len(aw-bw),
                    delta_valid=b['K']['8']['ValidCount']-a['K']['8']['ValidCount'],delta_modes=len(bw)-len(aw),
                    A_valid=a['K']['8']['ValidCount'],B_valid=b['K']['8']['ValidCount'],A_modes=len(aw),B_modes=len(bw),
                    A_collision=sum(not c['tip_segments_clear'] for c in a['candidates']),B_collision=sum(not c['tip_segments_clear'] for c in b['candidates']),
                    A_goal_fail=sum(not c['semantic_goal_correct'] for c in a['candidates']),B_goal_fail=sum(not c['semantic_goal_correct'] for c in b['candidates'])))
        out['mode_transitions']=dict(new_mode_occurrences=sum(t['new_valid_modes'] for t in transitions),lost_mode_occurrences=sum(t['lost_valid_modes'] for t in transitions),
            valid_gain_requests=sum(t['delta_valid']>0 for t in transitions),valid_gain_without_mode_count_gain=sum(t['delta_valid']>0 and t['delta_modes']<=0 for t in transitions),
            mode_gain_requests=sum(t['delta_modes']>0 for t in transitions),mode_loss_requests=sum(t['delta_modes']<0 for t in transitions))
        out['cases']=dict(largest_mode_gain=max(transitions,key=lambda x:x['delta_modes']),largest_mode_loss=min(transitions,key=lambda x:x['delta_modes']),
            largest_goal_regression=max(transitions,key=lambda x:x['B_goal_fail']-x['A_goal_fail']),largest_collision_regression=max(transitions,key=lambda x:x['B_collision']-x['A_collision']))
        write(root/(split+'_transitions.json'),transitions);results[split]=out
    write(root/'RESULTS.json',results)
    table=['| Split/seed | Valid A→B | Collision A→B | Goal fail A→B | Any@8 A→B | Modes@8 A→B | Two@4 A→B | Ref@8 A→B | Ref@4 A→B | qTop1 A→B |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields=['CandidateValidRate','CollisionFailureRate','GoalFailureRate','AnyValid8','GeometricModeCount8','TwoDistinctValid4','ReferenceModeCoverage8','ReferenceModeCoverage4','SelectedValid1']
    for split,out in results.items():
        for seed in [0,1,2,'mean']:
            ab=out['three_seed_means'] if seed=='mean' else {a:out['seeds'][a][seed] for a in ('A','B')}
            cells=['%s/%s'%(split,seed)]+[('%.3f→%.3f'%(ab['A'][k],ab['B'][k]) if k=='GeometricModeCount8' else '%.2f→%.2f%%'%(100*ab['A'][k],100*ab['B'][k])) for k in fields]
            table.append('| '+' | '.join(cells)+' |')
    (root/'CORE_TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf8')
    plots(root,results,all_rows)


def plots(root,results,all_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.plot_geometric_modes import examples
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
    for row,split in enumerate(('old_dev','dev32')):
        out=results[split]
        for col,key in enumerate(['CandidateValidRate','CollisionFailureRate','GeometricModeCount8','ReferenceModeCoverage4']):
            ax=axes[row,col]
            for s in range(3):ax.plot([0,1],[out['seeds'][a][s][key] for a in ('A','B')],marker='o',label='seed%d'%s)
            ax.set_xticks([0,1],['A Ordinary','B + clearance']);ax.set_title(split+' / '+key);ax.grid(alpha=.2)
            if col==0:ax.legend()
    fig.savefig(root/'PAIRED_RESULTS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,split in zip(axes,('old_dev','dev32')):
        for i,a in enumerate(('A','B')):
            d=results[split]['three_seed_means'][a];unknown=d['UnknownValidCount'];known=d['ValidCount8']-unknown
            ax.bar(i,known,color='royalblue',label='known valid' if i==0 else None)
            ax.bar(i,unknown,bottom=known,color='darkorange',label='unknown valid' if i==0 else None)
            ax.bar(i,8-known-unknown,bottom=known+unknown,color='lightcoral',label='invalid' if i==0 else None)
        ax.set_xticks([0,1],['A Ordinary','B + clearance']);ax.set_ylim(0,8);ax.set_title(split+' / all slots, three-seed mean');ax.legend()
    fig.savefig(root/'ALL_SLOT_OUTCOMES.png',dpi=160);plt.close(fig)
    # Existing route visualization reused for each seed; no favorable seed selection.
    import shutil
    for split in ('old_dev','dev32'):
        for s in range(3):
            folder=root/'figures'/split/('seed%d'%s);folder.mkdir(parents=True,exist_ok=True)
            for a in ('A','B'):shutil.copyfile(root/'evaluation'/split/('%s_seed%d'%(a,s))/'pool.npz',folder/(a+'_pool.npz'))
            shutil.copyfile(root/'evaluation'/split/'B_seed0/REFERENCE_GEOMETRY.json',folder/'REFERENCE_GEOMETRY.json')
            examples(folder,{a:all_rows[split][a,s] for a in ('A','B')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root)
