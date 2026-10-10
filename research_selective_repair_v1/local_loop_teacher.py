"""Counterbalanced enriched TRAIN oracle support acquisition, unchanged native executor."""
import argparse,copy,os,time
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,SOURCE,read,write,sha
from research_selective_repair_v1.local_loop_support import proposals
from research_selective_repair_v1.execute import callback
from research_selective_repair_v1.execution_semantics import audit

def run(name,support):
    from scripts import collect_observed_layout_variation as physical
    from scripts.collect_observed_layout_hash_recovery import validate_initial_geometry
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    support_folder=RUN/support;decision=read(support_folder/'SUMMARY.json');selected=decision['selected_TRAIN_requests']
    assert selected and all(r['fit_family'] and r['eligible']>0 for r in selected)
    dataset=RUN/'body_native_branch_data_v1/samples.npz';assert sha(dataset)==decision['dataset_sha256']
    with np.load(dataset) as z:d={k:z[k] for k in ('ids','slots','options','paths','completed','roles')}
    assert set(d['roles'])=={'TRAIN'}
    registration=read(ROOT/'data/selective_repair_interventions_v1/registration.json')
    parents={p['parent_id']:p for p in registration['parent_plan'] if p['role']=='TRAIN'}
    base=read(RUN/'constraints_seed0_v1/config.json')['post_base'];sealed=[]
    for row in selected:
        ix=np.flatnonzero((d['ids']==row['id'])&(d['options']==0));ix=ix[np.argsort(d['slots'][ix])];assert len(ix)==8
        candidates,ok,reasons=proposals(d['paths'][ix],d['completed'][ix[0]],base,row['starts'])
        assert ok.tolist()==row['eligible_mask'];sealed.append(dict(row=row,candidates=candidates,eligible=ok))
    np.savez_compressed(out/'SEALED_TRAIN_ORACLE_PROPOSALS.npz',ids=[r['row']['id'] for r in sealed],paths=[r['candidates'] for r in sealed])
    source_hashes={str(p.relative_to(SOURCE)):sha(p) for directory in ('scripts','research_selective_repair_v1') for p in (SOURCE/directory).rglob('*') if p.suffix in ('.py','.sh')}
    write(out/'PROTOCOL.json',dict(source_commit=os.environ.get('CODE_COMMIT'),support_sha256=sha(support_folder/'SUMMARY.json'),
        proposal_sha256=sha(out/'SEALED_TRAIN_ORACLE_PROPOSALS.npz'),selected=selected,attempts_per_arm=len(selected)*8,
        oracle_use='TRAIN observed first failure is edit-support label/acquisition ONLY. No deployment or learned localization claim.',
        scope='Enriched TRAIN target acquisition, not overall distribution/fresh/returned4 acceptance',native_RNG_controlled=False,locked_access=False))
    folders={};records={label:[] for label in ('null','loop')};schedule=[('null','loop'),('loop','null'),('loop','null'),('null','loop')]
    for label in records:
        folder=RUN/(name+'_'+label);folder.mkdir(parents=True,exist_ok=False);folders[label]=folder
        write(folder/'MANIFEST.json',dict(source_commit=os.environ.get('CODE_COMMIT'),source_sha256=source_hashes,
            TRAIN_feedback_only=True,oracle_support_TRAIN_only=True,locked_access=False,recipe=label,teacher_routes_per_parent=8,
            proposal_sha256=sha(out/'SEALED_TRAIN_ORACLE_PROPOSALS.npz')))
    physical.collect_routes=callback;physical.validate_initial_geometry=validate_initial_geometry;physical.PROTOCOL='local_closed_loop_TRAIN_oracle_support_v1'
    for j,pre in enumerate(sealed):
        ident=pre['row']['id'];parent=ident.rsplit('_target',1)[0];target=int(ident.rsplit('_target',1)[1])
        for label in schedule[j%4]:
            folder=folders[label];plan=copy.deepcopy(parents[parent]);option=1 if label=='null' else 2
            plan.update(execution_id=ident,execution_target=target,execution_paths=pre['candidates'][:,option].tolist(),execution_selected=list(range(8)))
            file=folder/('plan%d.json'%j);write(file,plan);dest=folder/('parent%d'%j)
            result=physical.physical_worker(registration,dict(source_sha256=source_hashes),plan,file,dest)
            assert result['status']=='collection_finished',result
            records[label].append(read(dest/'EXECUTION.json'));print(dict(id=ident,recipe=label,success=records[label][-1]['success'],modified_slots=int(pre['eligible'].sum())),flush=True)
    summaries=[]
    for label,folder in folders.items():
        write(folder/'SUMMARY.json',dict(attempted=len(selected)*8,success=sum(r['success'] for r in records[label]),records=records[label],TRAIN_feedback_only=True,locked_access=False))
        audit(folder.name+'_semantics_v1',folder.name);s=read(RUN/(folder.name+'_semantics_v1')/'SUMMARY.json')
        summaries.append(dict(recipe=label,E8=s['mean_executable_distinct'],successful_clear=s['successful_clear_routes'],attempted=len(selected)*8,requests=s['requests_results']))
    write(out/'SUMMARY.json',dict(records=summaries,seconds=time.monotonic()-tic,scope='Enriched TRAIN oracle-support native acquisition. Neither deployment localization nor fresh/returned4 evidence. Native RNG not controlled.',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--support',required=True);run(**vars(p.parse_args()))
