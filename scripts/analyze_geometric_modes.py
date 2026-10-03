"""Read-only fixed portal analysis, including all valid unknown predictions."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT, RUN, OLD, sha, read, write, lines
from routeset.geometric_modes import portal_word, encode_word, summarize_modes

SOURCES = {}
def checked(path):
    path = Path(path); SOURCES[str(path)] = sha(path); return path

def load_references(data, ids):
    from routeset.observed_route_head import resample_event_segments
    from routeset.observed_probability import reference_cluster_weights
    from scripts.evaluate_observed_two_row import scene_metrics
    labels = {r['id']:r for r in lines(checked(data/'supervision.jsonl')) if r['id'] in set(ids)}
    result = {}
    for identifier in ids:
        label = labels[identifier]
        assert label['split'] == 'DEV_MODEL'
        cfg = read(checked(label['route_config']))
        with np.load(checked(label['observation'])) as a:
            current = {k:a[k] for k in ('gripper_pose','gripper_open')}
        with np.load(checked(label['verification_only'])) as a:
            truth = {k:a[k] for k in ('obstacle_centers','obstacle_halfsizes')}
        paths, events = [], []
        for file in label['routes']:
            with np.load(checked(file)) as a:
                p,e = resample_event_segments(a['gripper_pose'],a['gripper_open'],24)
            paths.append(p); events.append(e)
        candidates = []
        if paths:
            _, candidates = scene_metrics(np.array(paths),np.array(events),current,truth,label['semantic_targets'],label['route_types'],cfg)
        words = [encode_word(portal_word(p,cfg)) if c['TipValid'] else None for p,c in zip(paths,candidates)]
        weights = reference_cluster_weights(np.array(paths)[None],np.ones((1,len(paths)),dtype=bool))[0] if paths else []
        result[identifier] = dict(config=cfg, paths=np.array(paths), words=words,
            balanced_reference_weights=list(map(float,weights)),
            reference_valid=[c['TipValid'] for c in candidates], current=current, truth=truth, label=label)
    return result

def evaluate_pool(paths, valid, ids, parents, candidates, refs, q=None, pi=None):
    from routeset.observed_probability import select_route_indices
    import torch
    torch.set_num_threads(1)
    selections = {'8':np.tile(np.arange(8),(len(ids),1))}
    if q is not None:
        for k in (1,2,4):
            selections[str(k)] = select_route_indices(torch.tensor(paths),torch.tensor(q),k).numpy()
    rows = []
    for n, identifier in enumerate(ids):
        ref = refs[str(identifier)]; cfg = ref['config']
        words = [encode_word(portal_word(p,cfg)) if v else None for p,v in zip(paths[n],valid[n])]
        refwords = {w for w in ref['words'] if w is not None}
        known = {words[i] for i,c in enumerate(candidates[n]) if valid[n,i] and c['declared_passage_type'] is not None}
        unknown = {words[i] for i,c in enumerate(candidates[n]) if valid[n,i] and c['declared_passage_type'] is None}
        legacy = {tuple(c['declared_passage_type']) for i,c in enumerate(candidates[n]) if valid[n,i] and c['declared_passage_type'] is not None}
        row = dict(id=str(identifier),parent=str(parents[n]),valid=valid[n].astype(int).tolist(),words=words,
            reference_words=sorted(refwords),reference_valid=ref['reference_valid'],
            unique_classified=len(legacy),unknown_valid=int(sum(valid[n,i] and c['declared_passage_type'] is None for i,c in enumerate(candidates[n]))),
            unknown_only_modes=len(unknown-known),unknown_new_vs_references=len(unknown-refwords),
            K={k:dict(summarize_modes(valid[n],words,refwords,indices[n]),selected=indices[n].tolist()) for k,indices in selections.items()},
            candidates=candidates[n],q=None if q is None else q[n].tolist())
        # pi mass on valid geometric groups, invalid mass kept explicitly.
        if pi is not None:
            row['pi_mass'] = {w:float(sum(pi[n,i] for i in range(8) if words[i]==w)) for w in set(words) if w is not None}
            row['pi_invalid_mass'] = float(pi[n][~valid[n]].sum())
            target_mass={w:sum(weight for word,weight in zip(ref['words'],ref['balanced_reference_weights']) if word==w) for w in refwords}
            row['pi_reference_target_mass']=target_mass
            row['pi_valid_mode_mass_TV']=.5*(row['pi_invalid_mass']+sum(abs(row['pi_mass'].get(w,0)-target_mass.get(w,0)) for w in set(row['pi_mass'])|set(target_mass))) if refwords else None
        rows.append(row)
    summary = aggregate(rows)
    eligible = [r for r in rows if len(r['reference_words'])>=2]
    collapsed = [r for r in eligible if r['K']['8']['GeometricModeCount']==1]
    summary['collapse_gate'] = dict(eligible=len(eligible),exactly_one=len(collapsed),
        zero_valid=sum(r['K']['8']['GeometricModeCount']==0 for r in eligible),
        passed=len(collapsed)>=6 and len(collapsed)>=len(eligible)/2)
    return summary, rows

def aggregate(rows):
    summary = dict(requests=len(rows),parents=len({r['parent'] for r in rows}),
        CandidateValidRate=float(np.mean([r['K']['8']['ValidCount']/8 for r in rows])),
        AnyValidAt8=float(np.mean([r['K']['8']['ValidCount']>0 for r in rows])),
        UniqueClassified=float(np.mean([r['unique_classified'] for r in rows])),
        UnknownValidCount=sum(r['unknown_valid'] for r in rows),
        UnknownOnlyModeCount=sum(r['unknown_only_modes'] for r in rows),
        UnknownModeNotInReferences=sum(r['unknown_new_vs_references'] for r in rows),
        ModeCountHistogram={str(k):sum(r['K']['8']['GeometricModeCount']==k for r in rows) for k in range(9)},K={})
    for k in rows[0]['K']:
        summary['K'][k]={key:float(np.mean([r['K'][k][key] for r in rows if r['K'][k][key] is not None]))
            if any(r['K'][k][key] is not None for r in rows) else None
            for key in ('ValidCount','GeometricModeCount','TwoDistinctValid','ReferenceModesHit','ReferenceModeCoverage')}
        summary['K'][k]['CoverageEligible'] = sum(r['K'][k]['ReferenceModeCoverage'] is not None for r in rows)
        summary['K'][k]['MultimodePoolLost'] = sum(r['K']['8']['TwoDistinctValid'] and not r['K'][k]['TwoDistinctValid'] for r in rows)
    return summary

def bootstrap(a,b):
    keys = [('8','ValidCount'),('8','GeometricModeCount'),('4','TwoDistinctValid'),('4','ReferenceModeCoverage'),('1','ValidCount')]
    ar={r['id']:r for r in a}; br={r['id']:r for r in b}; assert set(ar)==set(br)
    parents=sorted({r['parent'] for r in a}); rng=np.random.default_rng(20261004)
    draws=rng.integers(len(parents),size=(10000,len(parents))); out={}
    for k,metric in keys:
        if k not in a[0]['K']:continue
        deltas=[]
        for parent in parents:
            values=[br[i]['K'][k][metric]-ar[i]['K'][k][metric] for i in ar if ar[i]['parent']==parent and ar[i]['K'][k][metric] is not None]
            deltas.append(np.mean(values) if values else np.nan)
        samples=np.nanmean(np.array(deltas)[draws],axis=1)
        out[metric+'@'+k]=dict(delta=float(np.nanmean(deltas)),CI95=np.nanpercentile(samples,[2.5,97.5]).tolist())
    return out

def plots(output, datasets, refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # All conditions, both arms, including all failures; no favorable subset.
    names=list(datasets)
    for start in range(0,len(datasets[names[0]][1]),6):
        fig=plt.figure(figsize=(16,18))
        for col,name in enumerate(names):
            paths,rows=datasets[name]
            for local,row in enumerate(rows[start:start+6]):
                n=start+local; ax=fig.add_subplot(6,len(names),local*len(names)+col+1,projection='3d')
                ref=refs[row['id']]
                for c,h in zip(ref['truth']['obstacle_centers'],ref['truth']['obstacle_halfsizes']):
                    vertices=np.array([c+h*np.array([x,y,z]) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                    faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
                    ax.add_collection3d(Poly3DCollection([vertices[f] for f in faces],alpha=.15,facecolor='gray'))
                for p in ref['paths']:ax.plot(*p.T,color='gray',alpha=.3,lw=.7)
                for i,p in enumerate(paths[n]):
                    color='crimson' if not row['valid'][i] else ('darkorange' if row['candidates'][i]['declared_passage_type'] is None else 'royalblue')
                    ax.plot(*p.T,color=color,alpha=.75,lw=1.3 if row['valid'][i] else .8,ls='-' if row['valid'][i] else '--')
                    ax.text(*p[len(p)//2],str(i),fontsize=6,color=color)
                # Shared per-request bounds cover both arms, all references and boxes.
                allpoints=[datasets[key][0][n].reshape(-1,3) for key in names]
                if len(ref['paths']):allpoints.append(ref['paths'].reshape(-1,3))
                allpoints.extend([ref['truth']['obstacle_centers']-ref['truth']['obstacle_halfsizes'],ref['truth']['obstacle_centers']+ref['truth']['obstacle_halfsizes']])
                points=np.concatenate(allpoints);lo=points.min(0)-.03;hi=points.max(0)+.03
                ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo)
                ax.set_title('%s %s\nvalid=%d modes=%d unknown=%d'%(name,row['id'].replace('two_row_reach_',''),row['K']['8']['ValidCount'],row['K']['8']['GeometricModeCount'],row['unknown_valid']),fontsize=9)
                ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z');ax.view_init(25,-65)
        fig.suptitle('All M8 candidates: blue=known valid, orange=unknown valid, red dashed=invalid; gray=references/boxes')
        fig.tight_layout();fig.savefig(output/('ALL_ROUTES_%02d.png'%start),dpi=130);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--new-dev',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    results={}; datasets={};refs=None
    for arm in ('ordinary','balanced_probability'):
        family=RUN/('M8_%s_seed0_expanded_scores'%arm)
        poolpath=family/'pools/DEV_MODEL/pool.npz' if a.new_dev is None else a.new_dev/arm/'pool.npz'
        with np.load(checked(poolpath)) as z:pool={k:z[k] for k in z.files}
        ids=pool['ids'].astype(str)
        if refs is None:refs=load_references(OLD if a.new_dev is None else ROOT/'data/geometric_modes_v1/DEV_MODEL',ids)
        scenes=read(checked(poolpath.parent/'per_scene.json'))
        candidates=[r['candidates'] for r in scenes]
        if a.new_dev is None:
            with np.load(checked(family/'calibration_seed0/predictions.npz')) as z:logits=z['dev_logits']
            temperature=read(checked(family/'calibration_seed0/summary.json'))['temperature']
            q=1/(1+np.exp(-np.clip(logits/temperature,-60,60)))
        else:q=pool['q']
        summary,rows=evaluate_pool(pool['paths'],pool['labels'].astype(bool),ids,pool['parents'],candidates,refs,q,pool['pi'])
        results[arm]=summary;write(a.output/(arm+'_rows.json'),rows)
        datasets[arm]=(pool['paths'],rows)
    results['paired_parent_bootstrap']=bootstrap(datasets['ordinary'][1],datasets['balanced_probability'][1])
    if a.new_dev is None:
        for seed in (1,2):
            for arm in ('ordinary','balanced_probability'):
                folder=RUN/('M8_%s_seed%d_expanded'%(arm,seed))/'dev/step03000'
                with np.load(checked(folder/'predictions.npz')) as z:pool={k:z[k] for k in z.files}
                scenes=read(checked(folder/'per_scene.json'))
                # Generation evaluation embeds candidates in each per-scene row.
                candidates=[r['tip_candidates'] for r in scenes]
                valid=np.array([[c['TipValid'] for c in row] for row in candidates])
                summary,rows=evaluate_pool(pool['paths'],valid,pool['scene_ids'],pool['parent_ids'],candidates,refs)
                name='%s_seed%d'%(arm,seed);results[name]=summary;write(a.output/(name+'_rows.json'),rows)
    write(a.output/'RESULTS.json',results);write(a.output/'SOURCE_INDEX.json',SOURCES)
    plots(a.output,datasets,refs)
    print(__import__('json').dumps(results),flush=True)

if __name__=='__main__':main()
