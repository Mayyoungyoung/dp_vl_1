"""Fixed parent witnesses, invalidated-coordinate repair, paired family summaries."""
import argparse
from collections import Counter
import json
import numpy as np
from scripts.mode_geometry_experiment import RUN,DATA,old
from scripts.run_observed_probability import read,write,sha,lines
from scripts.research_v3_audit import mode,plain
from scripts.evaluate_paired_modes import references,check_candidates


def main(names,output,figures=False):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    witness_path=old.RUN/'strong_counterfactual_v1/rows.json'
    assert sha(witness_path)=='515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084'
    witness=read(witness_path)['safety_mean']
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    refs={k:references(r) for k,r in labels.items()}
    with np.load(old.SUPPORT/'support.npz') as z:
        known={str(k):set(z['modes'][i,z['mask'][i]]) for i,k in enumerate(z['ids']) if z['splits'][i]=='DEV_MODEL'}
    folders={'parent':old.RUN/'safety_mean/evaluation_matched_q_v1'}
    for n in names:
        if n.startswith('historical_'):folders[n]=ROOT_HIST/n[len('historical_'):]/'evaluation_matched_q_v1'
        else:
            name,sampling=n.split(':') if ':' in n else (n,'balanced')
            folders[n]=RUN/name/('eval_'+sampling)
    models={};pools={};hashes={}
    for n,f in folders.items():
        models[n]={r['id']:r for r in read(f/'rows.json')}
        with np.load(f/'pool.npz') as z:pools[n]={k:z[k] for k in z.files}
        pools[n]['index']={str(k):i for i,k in enumerate(pools[n]['ids'])}
        hashes[str(f/'pool.npz')]=sha(f/'pool.npz')
    opportunities=[]
    for w in witness:
        a,b=w['source_id'],w['destination_id'];ai=pools['parent']['index'][a]
        original=models['parent'][a]
        for word in sorted(set(original['words'])-{None}):
            slots=[j for j,(v,c,m) in enumerate(zip(w['source_valid'],w['cross_valid'],original['words'])) if v and not c and m==word]
            if slots and word in known[b]:
                same=[j for j,(v,m) in enumerate(zip(w['source_valid'],original['words'])) if v and m==word]
                opportunities.append(dict(source_id=a,destination_id=b,family=w['family'],mode=word,old_slots=slots,
                    all_same_mode_source_routes_invalid=all(not w['cross_valid'][j] for j in same)))
    results={};allrecords={}
    for n,model in models.items():
        lost=recovered=retained=0;counts=Counter();records=[];family={}
        for w in witness:
            b=w['destination_id'];present={x for x,v in zip(model[b]['words'],model[b]['valid']) if v and x is not None}
            parent_present={x for x,v in zip(models['parent'][b]['words'],models['parent'][b]['valid']) if v and x is not None}
            surv=set(w['surviving_modes']);hit=surv&present
            retained+=len(hit);lost+=len((surv&parent_present)-present);recovered+=len((surv-parent_present)&present)
            family.setdefault(w['family'],dict(denom=0,retained=0,adaptation_denom=0,adapted=0))
            family[w['family']]['denom']+=len(surv);family[w['family']]['retained']+=len(hit)
        for o in opportunities:
            b=o['destination_id'];r=model[b];pool=pools[n];j=pool['index'][b]
            p=pool['paths'][j];valid=np.asarray(r['valid']);words=r['words']
            rawwords=[mode(x,refs[b]['config']) for x in p] # diagnostic label even on invalid paths
            parent=pools['parent'];a=parent['index'][o['source_id']]
            oldp=parent['paths'][a,o['old_slots']]
            copied=bool((np.linalg.norm(p[:,None]-oldp[None],axis=-1).mean(-1)<.001).any())
            if any(v and w==o['mode'] for v,w in zip(valid,words)):category='valid_same_mode_repair'
            elif copied:category='copied_old_coordinates'
            elif any(w==o['mode'] for w in rawwords):category='same_mode_but_invalid'
            elif valid.any():category='other_valid_modes_only'
            else:category='no_valid_output'
            counts[category]+=1;records.append(dict(o,category=category))
            if o['all_same_mode_source_routes_invalid']:
                counts['strict_opportunities']+=1;counts['strict_repairs']+=int(category=='valid_same_mode_repair')
            family[o['family']]['adaptation_denom']+=1;family[o['family']]['adapted']+=int(category=='valid_same_mode_repair')
        rr=list(model.values());f=folders[n];m=read(f/'metrics.json')
        results[n]=dict(raw=m['raw'],selected=m['selected'],condition_hit=m.get('condition_hit'),
            retained=retained,retention=retained/2169,lost_of_1872=lost,recovered_of_297=recovered,
            adaptation_opportunities=len(opportunities),adaptation_counts={k:v for k,v in counts.items() if not k.startswith('strict_')},
            strict_adaptation=dict(opportunities=counts['strict_opportunities'],repairs=counts['strict_repairs']),
            adaptation_fraction=counts['valid_same_mode_repair']/len(opportunities),families=family,
            invalid_reasons={key:sum(not c[key] for r in rr for c in r['candidates']) for key in ('semantic_goal_correct','tip_segments_clear','event_state_sequence_correct')})
        assert retained==1872-lost+recovered
        allrecords[n]=records
    assert results['parent']['retained']==1872
    families=sorted(results['parent']['families']);draw=np.random.default_rng(6100910).integers(32,size=(5000,32))
    differences={}
    for a in names:
        for b in ['parent']+names:
            if a==b:continue
            af=results[a]['families'];bf=results[b]['families']
            delta=np.array([af[f]['retained']-bf[f]['retained'] for f in families]);den=np.array([af[f]['denom'] for f in families])
            differences[a+' minus '+b]=dict(retention_delta=float(delta.sum()/den.sum()),retention_CI95=np.quantile(delta[draw].sum(1)/den[draw].sum(1),[.025,.975]).tolist())
    write(out/'RESULTS.json',plain(dict(models=results,differences=differences,witness_sha256=sha(witness_path),pool_hashes=hashes,
        adaptation_definition='Fixed parent-valid old paths fail destination; destination known valid same-mode reference. Deduplicated source/destination/word.',
        uncertainty='32 reused DEV layout families; conditional on observed training seeds; not independent test evidence')))
    write(out/'adaptation_rows.json',allrecords)
    print(json.dumps({n:{k:v for k,v in r.items() if k!='families'} for n,r in results.items()}),flush=True)
    if figures:plot(out,names,results,allrecords,pools,models,refs)


def plot(out,names,results,records,pools,models,refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Deterministic first newly repaired, first lost, first remaining failure.
    last=names[-1];chosen=[]
    comparisons=list(zip(records['parent'],records[last]))
    for predicate in (lambda a,b:a['category']!='valid_same_mode_repair' and b['category']=='valid_same_mode_repair',
                      lambda a,b:a['category']=='valid_same_mode_repair' and b['category']!='valid_same_mode_repair',
                      lambda a,b:b['category']!='valid_same_mode_repair'):
        candidates=[b for a,b in comparisons if predicate(a,b) and b not in chosen]
        if candidates:chosen.append(candidates[0])
    for fi,o in enumerate(chosen):
        cols=['parent']+names[-3:];fig=plt.figure(figsize=(16,10))
        for row,ident in enumerate((o['source_id'],o['destination_id'])):
            ref=refs[ident]
            for col,n in enumerate(cols):
                ax=fig.add_subplot(2,len(cols),row*len(cols)+col+1,projection='3d')
                centers=ref['truth']['obstacle_centers'];halves=ref['truth']['obstacle_halfsizes']
                ax.bar3d(*(centers-halves).T,*(2*halves).T,alpha=.16,color='gray',shade=False)
                pool=pools[n];idx=pool['index'][ident];r=models[n][ident]
                for j,p in enumerate(pool['paths'][idx]):
                    hit=r['valid'][j] and r['words'][j]==o['mode']
                    ax.plot(*p.T,color='green' if hit else '#2369a0' if r['valid'][j] else '#c75050',alpha=.95 if hit else .45,lw=2.5 if hit else 1)
                if row:
                    pp=pools['parent'];oldp=pp['paths'][pp['index'][o['source_id']],o['old_slots'][0]]
                    ax.plot(*oldp.T,'k--',lw=1.5)
                ax.set_title(n+' | '+('source' if row==0 else 'edited')+'\nvalid='+str(sum(r['valid']))+'/8')
                ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)');ax.view_init(35,-65)
        fig.suptitle(o['source_id']+' → '+o['destination_id']+' | '+o['mode']+'\n'+o['category']+'; green: valid same mode; blue: other valid; red: invalid; dashed: invalidated old path',fontsize=10)
        fig.subplots_adjust(left=.01,right=.98,bottom=.05,top=.86,wspace=.08,hspace=.25)
        fig.savefig(out/('case_%d.png'%fi),dpi=150);plt.close(fig)
    write(out/'figure_cases.json',chosen)


ROOT_HIST=old.ROOT/'runs/verified_set_v1/all_population_v2'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--figures',action='store_true')
    a=p.parse_args();main(a.names,a.output,a.figures)
