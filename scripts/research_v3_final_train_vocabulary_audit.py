"""Full TRAIN all-mode control forward then sealed three-model vocabulary audit."""
import numpy as np
from scripts.research_v3_posttrain_collision import main as audit
from scripts.research_v3_frequency import RUN,SUPPORT
from scripts.paired_modes_data import DATA
from scripts.run_observed_probability import read,write,sha,lines
from scripts.research_v3_audit import mode


def main():
    audit('all_mode_train_audit_v1','verified_edit_all_modes')
    roots={'safety_mean':RUN/'posttrain_collision_v1','verified_edit_augmented':RUN/'verified_edit_train_failure_audit_v1','verified_edit_all_modes':RUN/'all_mode_train_audit_v1'}
    predictions={};checks={};hashes={};counts={}
    for n,root in roots.items():
        r=read(root/'RESULTS.json');counts[n]=r['counts']
        assert sha(root/'predictions.npz')==r['prediction_sha256'] and sha(RUN/n/'last.pt')==r['checkpoint_sha256']
        with np.load(root/'predictions.npz') as z:predictions[n]={str(i):z['paths'][j] for j,i in enumerate(z['ids'])}
        checks[n]={r['id']:[c['TipValid'] for c in r['candidates']] for r in read(root/'rows.json')}
        for p in (root/'RESULTS.json',root/'rows.json',root/'predictions.npz'):hashes[str(p)]=sha(p)
    supports={}
    for key,root in [('original',SUPPORT),('expanded',RUN/'verified_edit_all_modes_support_v1')]:
        assert sha(root/'support.npz')==read(root/'receipt.json')['support_sha256'];hashes[str(root/'support.npz')]=sha(root/'support.npz')
        with np.load(root/'support.npz') as z:supports[key]={str(i):set(z['modes'][j,z['mask'][j]]) for j,i in enumerate(z['ids']) if z['splits'][j]=='TRAIN'}
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    ids=list(supports['original']);assert len(ids)==1152 and set(ids)==set(supports['expanded'])==set(labels)
    for n in roots:assert set(ids)==set(predictions[n])==set(checks[n])
    records=[]
    for i in ids:
        cfg=read(labels[i]['route_config']);hashes[labels[i]['route_config']]=sha(labels[i]['route_config'])
        old=supports['original'][i];new=supports['expanded'][i];extra=new-old;assert old<=new
        rec=dict(id=i,family=i.rsplit('_',2)[0],original_classes=len(old),expanded_classes=len(new),new_classes=len(extra),models={})
        for n in roots:
            words={mode(p,cfg) for p,ok in zip(predictions[n][i],checks[n][i]) if ok};words.discard(None)
            rec['models'][n]=dict(valid_distinct=len(words),original_hits=len(words&old),expanded_hits=len(words&new),new_class_hits=len(words&extra),
                original_recall=len(words&old)/len(old),expanded_recall=len(words&new)/len(new),
                original_capacity_deficit=min(8,len(old))-len(words&old),expanded_capacity_deficit=min(8,len(new))-len(words&new))
        records.append(rec)
    summary={n:{k:float(np.mean([r['models'][n][k] for r in records])) for k in records[0]['models'][n]} for n in roots}
    totals={n:{k:int(sum(r['models'][n][k] for r in records)) for k in ('original_hits','expanded_hits','new_class_hits')} for n in roots}
    result=dict(counts=counts,mean=summary,total_hits=totals,requests=1152,families=128,new_reference_classes=sum(r['new_classes'] for r in records),
        input_sha256=hashes,scope='TRAIN fitting attribution only, fixed final models, reused sealed baseline predictions. Positive vocabularies incomplete; capacity-normalized deficits do not establish a new mechanism or independent-test result.')
    out=RUN/'all_mode_train_audit_v1';write(out/'vocabulary_rows.json',records);write(out/'VOCABULARY.json',result)
    print(dict(mean=summary,total_hits=totals),flush=True)


if __name__=='__main__':main()
