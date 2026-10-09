"""Physical local interventions; family roles fixed before rendering or inference."""
import argparse,copy,os,subprocess,time
from pathlib import Path
from scripts import paired_modes_data as paired
from research_selective_repair_v1.core import ROOT,SOURCE,RUN,read,write,sha,lines,plain

def locations(tag):return ROOT/('data/selective_repair_'+tag),RUN/('collection_'+tag)

def prepare(tag='interventions_v1',policy=None):
    policy=policy or SOURCE/'configs/selective_repair_data_v1.json';spec=read(policy);data,work=locations(tag)
    original=paired.read
    def scoped(p):
        value=original(p)
        if Path(p)==paired.POLICY:value.update(seed=spec['seed'],train_families=spec['train_families'],dev_families=spec['dev_families'],variants=['open','closed','shifted'])
        return value
    paired.read=scoped
    try:value=paired.registration()
    finally:paired.read=original
    old_hashes={p['registered_geometry_1mm_sha256'] for p in read(paired.DATA/'registration.json')['parent_plan']}
    old_hashes|={p['registered_geometry_1mm_sha256'] for p in read(ROOT/'data/feasible_space_generalization_v1/registration.json')['parent_plan']}
    if tag!='interventions_v1' and (ROOT/'data/selective_repair_interventions_v1/registration.json').exists():old_hashes|={p['registered_geometry_1mm_sha256'] for p in read(ROOT/'data/selective_repair_interventions_v1/registration.json')['parent_plan']}
    plans=[]
    for f in range(spec['train_families']+spec['dev_families']):
        base=value['parent_plan'][3*f];role='TRAIN' if f<spec['train_families'] else 'DEV_MODEL'
        for variant in spec['variants']:
            p=copy.deepcopy(base);c=p['config'];r=p['changed_row'];other=1-r
            if variant=='near':c['post_y'][r][0]+=.025
            elif variant=='narrow':
                mid=sum(c['post_y'][r])/2;c['post_y'][r]=[mid-.055,mid+.055]
            elif variant=='tall':c['post_heights'][r]+=.07
            elif variant=='far':c['post_y'][other][1]+=.025
            elif variant=='closed':
                mid=sum(c['post_y'][r])/2;c['post_y'][r]=[mid-.028,mid+.028]
            elif variant=='combo':
                c['post_y'][r][0]+=.023;c['post_heights'][other]+=.055;c['row_x'][r]+=.007
            ident='sr_family_%06d'%(spec['family_origin']+f)
            p.update(index=len(plans),family_id=ident,parent_id=ident+'_'+variant,variant=variant,role=role,split=role)
            p['seed']=spec['family_origin']+len(plans);c.update(seed=p['seed'],split=role)
            cs,hs=paired.geom.geometry(c);p['registered_geometry_1mm_sha256']=paired.quantized_hash_v2(cs,hs,c['goal_xyz'])
            assert p['registered_geometry_1mm_sha256'] not in old_hashes
            p['guide_plans']=[paired.geom.guide_plan(c,t) for t in range(3)];p['geometry_precheck']=paired.geom.geometry_precheck(c)
            p['low_gap_certificates']=paired.geom.gap_certificates(c);assert p['geometry_precheck']['passed'];plans.append(p)
    assert len({p['registered_geometry_1mm_sha256'] for p in plans})==len(plans)
    value.update(protocol=spec['protocol'],policy=spec,parent_plan=plans,source_commit=os.environ.get('CODE_COMMIT'),previous_geometry_overlap=0,confirmation=spec['confirmation'])
    data.mkdir(parents=True,exist_ok=False);write(data/'registration.json',value)
    write(data/'source_sha256.json',{str(f.relative_to(SOURCE)):sha(f) for d in ('routeset','scripts','configs','research_selective_repair_v1') for f in (SOURCE/d).rglob('*') if f.is_file() and f.suffix in ('.py','.json','.sh')})
    print(dict(tag=tag,parents=len(plans),requests=3*len(plans),TRAIN=3*len(plans)*spec['train_families']/(spec['train_families']+spec['dev_families'])),flush=True)

def worker(tag,index):
    data,work=locations(tag);paired.DATA=data;paired.RUN=work;return paired.worker(index)

def collect(tag='interventions_v1'):
    data,work=locations(tag);work.mkdir(parents=True,exist_ok=True)
    for p in read(data/'registration.json')['parent_plan']:
        i=p['index'];receipt=work/('%03d.json'%i)
        if receipt.exists():
            prior=read(receipt)
            if prior['status']=='completed':continue
            raise RuntimeError('Unresolved rendering attempt; inspect it before explicit recovery')
        cmd=[str(ROOT/'.venv-sim/bin/python'),'-m','research_selective_repair_v1.data','worker','--tag',tag,'--index',str(i)]
        tic=time.monotonic();r=dict(index=i,parent_id=p['parent_id'],command=cmd,status='running',pid=None)
        with (work/('%03d.log'%i)).open('w') as log:
            child=subprocess.Popen(cmd,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT);r['pid']=child.pid;write(receipt,r);code=child.wait()
        r.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.monotonic()-tic);write(receipt,r);print(r,flush=True)
        if code:raise RuntimeError('Render failed, no replacement or inferred success')

def export(tag='interventions_v1'):
    data,work=locations(tag);paired.DATA=data;paired.export()
    manifest=read(data/'export/manifest.json');manifest.update(protocol=read(data/'registration.json')['protocol'],family_separation=True,confirmation=read(data/'registration.json')['confirmation']);write(data/'export/manifest.json',manifest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','collect','export']);p.add_argument('--tag',default='interventions_v1');p.add_argument('--index',type=int);p.add_argument('--policy',type=Path);a=p.parse_args()
    if a.stage=='prepare':prepare(a.tag,a.policy)
    elif a.stage=='worker':worker(a.tag,a.index)
    else:globals()[a.stage](a.tag)
