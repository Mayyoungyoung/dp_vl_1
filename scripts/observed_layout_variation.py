"""Pure TRAIN12 registration/geometry. No simulator, model, or raw-data IO."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scripts import register_two_row_formal as prior

PROTOCOL='observed_layout_variation_train12_v1'
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/observed_layout_variation_train12_v1.json'
HISTORY_FILES=('configs/observed_two_row_formal116_exclusions_v1.json',
               'configs/observed_two_row_formal116_registered_v1.json',
               'configs/observed_two_row_extension288_registered_v1.json')
GOALS=[[[.46,-.16,.84],[.46,0,.84],[.46,.16,.84]],
       [[.48,-.19,.84],[.45,.025,.855],[.49,.18,.825]]]
TABLE=[([.24],[[0]],[.10],0),([.30],[[.04]],[.18],1),
       ([.27],[[-.09,.09]],[.12],0),([.31],[[-.07,.11]],[.18],1),
       ([.17,.35],[[-.10,.10],[-.075,.075]],[.14,.14],0),
       ([.17,.35],[[-.10,.10],[-.0275,.0275]],[.14,.14],0),
       ([.20,.36],[[-.12,.08],[-.075,.075]],[.12,.16],1),
       ([.20,.36],[[-.12,.08],[-.0275,.0275]],[.12,.16],1),
       ([.18,.34],[[-.18,0,.18],[-.18,0,.18]],[.10,.14],0),
       ([.22,.36],[[-.20,-.02,.16],[-.17,.01,.19]],[.14,.18],1),
       ([.16,.32],[[-.16,.015,.19],[-.19,-.015,.16]],[.18,.12],0),
       ([.20,.35],[[-.20,0,.20],[-.16,0,.16]],[.12,.16],1)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def chash(value):return hashlib.sha256(canonical(value)).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def geometry(config):
    centers=[];halves=[]
    for x,ys,h in zip(config['row_x'],config['post_y'],config['post_heights']):
        for y in ys:
            centers.append([x,y,config['post_base_z']+h/2]);halves.append([.0175,.0175,h/2])
    return np.asarray(centers,dtype=float),np.asarray(halves,dtype=float)


def physical_hash(centers,halves,goals,quantum=None):
    centers,halves,goals=(np.asarray(x,dtype=float) for x in (centers,halves,goals))
    if centers.ndim!=2 or centers.shape[1:]!=(3,) or halves.shape!=centers.shape or goals.shape!=(3,3):
        raise ValueError('Complete variable-N centers/halves and three goals required')
    if not np.isfinite(np.concatenate((centers.ravel(),halves.ravel(),goals.ravel()))).all():
        raise ValueError('Nonfinite geometry')
    order=np.lexsort((centers[:,2],centers[:,1],centers[:,0]))
    a=np.concatenate((centers[order].ravel(),halves[order].ravel(),goals.ravel()))
    # The exact legacy four-box encoding preserves comparability with old hashes.
    if len(centers)==4:
        a=np.concatenate((a,[0.,0.,0.]))
        encoded=np.rint(a/quantum).astype('<i8') if quantum else a.astype('<f8')
        return hashlib.sha256(encoded.tobytes()).hexdigest()
    encoded=np.rint(a/quantum).astype('<i8') if quantum else a.astype('<f8')
    return hashlib.sha256(canonical(dict(schema='variable_axis_aligned_cuboids_v1',n=len(centers)))+encoded.tobytes()).hexdigest()


def gap_certificates(config,centers=None,halves=None):
    if centers is None:centers,halves=geometry(config)
    centers=np.asarray(centers);halves=np.asarray(halves);offset=0;rows=[];m=config['tip_clearance_m']
    for row,ys in enumerate(config['post_y']):
        cs=centers[offset:offset+len(ys)];hs=halves[offset:offset+len(ys)];offset+=len(ys)
        if len(cs)!=len(ys) or not np.allclose(cs[:,0],config['row_x'][row],atol=1e-6,rtol=0):
            raise ValueError('Actual boxes do not match the registered row')
        for j in range(1,len(ys)):
            left=float(cs[j-1,1]+hs[j-1,1]+m);right=float(cs[j,1]-hs[j,1]-m)
            bottom=float(max((cs-hs)[:,2])+m);top=float(min((cs+hs)[:,2])-m)
            rows.append(dict(row=row,label='gap%d'%j,low_height_band_m=[bottom,top],
                y_interval_m=[left,right],signed_width_m=right-left,
                closed_for_this_low_relation=bool(right<=left and top>=bottom),
                all_route_classes_exhausted=False))
    return rows


def crossing_signature(xyz,config):
    xyz=np.asarray(xyz,dtype=float);xs=config['row_x'];m=config['tip_clearance_m']
    if xyz.ndim!=2 or xyz.shape[1:]!=(3,) or len(xyz)<2 or not np.isfinite(xyz).all():return None
    if not xyz[0,0]<xs[0] or not xyz[-1,0]>xs[-1]:return None
    labels=[[] for _ in xs];events=[]
    for a,b in zip(xyz[:-1],xyz[1:]):
        hits=[]
        for row,x in enumerate(xs):
            if not min(a[0],b[0])<=x<=max(a[0],b[0]):continue
            if abs(b[0]-a[0])<=1e-12:points=[a,b];fraction=0.
            else:
                fraction=(x-a[0])/(b[0]-a[0]);points=[a+fraction*(b-a)]
            for p in points:
                top=config['post_base_z']+config['post_heights'][row]
                if p[2]>top+m:label='over'
                elif config['post_base_z']+m<=p[2]<=top-m:
                    ys=config['post_y'][row];y=p[1]
                    if y<ys[0]-.0175-m:label='gap0'
                    elif y>ys[-1]+.0175+m:label='gap%d'%len(ys)
                    else:
                        found=[j for j in range(1,len(ys)) if ys[j-1]+.0175+m<y<ys[j]-.0175-m]
                        if len(found)!=1:return None
                        label='gap%d'%found[0]
                else:return None
                labels[row].append(label)
            hits.append((fraction,row))
        for _,row in sorted(hits):
            if not events or events[-1]!=row:events.append(row)
    if events!=list(range(len(xs))) or any(not row or len(set(row))!=1 for row in labels):return None
    return tuple(row[0] for row in labels)


def intents(config):
    n=len(config['post_y'][0])
    if len(config['row_x'])==1:
        # Nine slots, including repeated intent variants, are not nine route types.
        labels=['gap%d'%i for i in range(n+1)]+['over']
        return [(labels[i%len(labels)],) for i in range(9)]
    indices=([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,2),(3,3)] if n==2 else
             [(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3),(4,4)])
    return [tuple('over' if j==n+1 else 'gap%d'%j for j in pair) for pair in indices]


def row_point(config,row,label,variant):
    ys=config['post_y'][row];top=config['post_base_z']+config['post_heights'][row];m=config['tip_clearance_m']
    if label=='over':return float(np.mean(ys))+.01*variant,top+.025
    j=int(label[3:])
    if j==0:y=ys[0]-.0175-m-.075
    elif j==len(ys):y=ys[-1]+.0175+m+.075
    else:
        lo=ys[j-1]+.0175+m;hi=ys[j]-.0175-m
        if hi<=lo:return None
        y=(lo+hi)/2
    return y+.005*variant,min(.865,top-.025)


def guide_plan(config,target):
    centers,halves=geometry(config);result=[]
    for slot,intent in enumerate(intents(config)):
        variant=(-1,0,1)[slot//3] if len(config['row_x'])==1 else 0
        yz=[row_point(config,r,label,variant) for r,label in enumerate(intent)]
        points=[];reason=None
        if any(p is None for p in yz):reason='registered_low_gap_closed'
        else:
            xs=config['row_x']
            for row,(y,z) in enumerate(yz):
                x=xs[row];points.extend([[x-.05,y,z],[x,y,z],[x+.05,y,z]])
                if row==0 and len(xs)==2:
                    mid=(xs[0]+xs[1])/2
                    points.extend([[mid,y,z],[mid,yz[1][0],yz[1][1]]])
            points.append(config['goal_xyz'][target])
            raw=np.asarray([config['entry_xyz']]+points)
            if not prior.batch.collector.legacy.tip_polyline_clear(raw,centers,halves,config['tip_clearance_m']):
                reason='registered_ideal_tip_guide_collision'
            elif crossing_signature(raw,config)!=intent:reason='registered_ideal_signature_ambiguous'
        if len(points)>9:raise ValueError('Guide budget exceeded')
        result.append(dict(slot=slot,intent_supervision_only=list(intent),guide_variant=variant,
            waypoints_supervision_only=points,planning_call_limit=len(points),
            collection_allowed=reason is None,precheck_reason=reason,
            scope='Ideal collection guide only; no robot reachability or actual route class claim'))
    return result


def geometry_precheck(config):
    cs,hs=geometry(config);reasons=[]
    if not np.isfinite(np.concatenate((cs.ravel(),hs.ravel()))).all() or np.any(hs<=0):reasons.append('invalid_geometry')
    for i in range(len(cs)):
        for j in range(i):
            if np.all(np.abs(cs[i]-cs[j])<hs[i]+hs[j]):reasons.append('physical_box_overlap')
    for p in [config['entry_xyz']]+config['goal_xyz']:
        if np.any(np.all(np.abs(cs-np.asarray(p))<=hs+config['tip_clearance_m'],axis=1)):
            reasons.append('entry_or_goal_inside_inflated_box')
    return dict(passed=not reasons,reasons=sorted(set(reasons)),
                scope='Layout consistency only; individual closed guide slots do not invalidate a parent')


def historical_metadata(project=ROOT):
    """Open only pre-existing registration/exclusion JSON; never any raw role."""
    rows=[];files={}
    for relative in HISTORY_FILES:
        path=Path(project)/relative;value=read(path);files[relative]=sha(path)
        if 'entries' in value:rows.extend(copy.deepcopy(value['entries']))
        else:
            for p in value['parent_plan']:
                rows.append(dict(parent_id=p['parent_id'],role=p['role'],version=value['protocol'],
                    geometry_exact_sha256=p['registered_geometry_exact_sha256'],
                    geometry_1mm_sha256=p['registered_geometry_1mm_sha256'],source=relative,source_sha256=sha(path)))
    if len(rows)!=9+116+288:raise ValueError('All prior public9/registered116/registered288 mechanical identities required')
    return rows,files


def validate_spec(spec):
    fixed=dict(protocol=PROTOCOL,seed_start=401000,parents=12,targets_per_parent=3,slots_per_target=9,
        requested_slots=324,color_seed=401999,role='TRAIN',stages={'pilot4':[0,4],'pilot12':[4,12]},
        setup_path_budget=0,initialization_ik_budget=0,fallback_attempts=0,geometry_resampling_attempts=0,
        maximum_explicit_get_path_calls=2916,maximum_steps_per_segment=1000,soft_wall_seconds=2700,
        tip_clearance_m=.02,endpoint_tolerance_m=.03,canonical_entry_tolerance_m=.01,
        minimum_visible_pixels=10,post_base_z=.755,post_width_m=.035,image_size=224,
        row_guide_x_offset=.05,reference_set_complete=False,all_solution_count=None)
    if any(spec.get(k)!=v for k,v in fixed.items()):raise ValueError('Frozen TRAIN12 policy differs')
    if spec['layout_table']!=[dict(row_x=x,post_y=y,post_heights=h,goal_group=g) for x,y,h,g in TABLE]:
        raise ValueError('One-shot12 geometry matrix differs')
    original=read(ROOT/'configs/observed_two_row_extension288_v1.json')
    if (spec['canonical_init']!=original['canonical_init'] or spec['color_table_sha256']!=prior.canonical_hash(prior.color_table())
            or spec['color_source_sha256']!=original['color_source_sha256']):
        raise ValueError('Canonical initial state or official color table changed')


def build_registration(spec,excluded,source_files):
    validate_spec(spec)
    if len(excluded)!=413:raise ValueError('Full9+116+288 mechanical exclusion table required')
    for entry in excluded:
        if not all(entry.get(k) for k in ('parent_id','role','version','source')):raise ValueError('Incomplete mechanical source identity')
        for key in ('geometry_exact_sha256','geometry_1mm_sha256','source_sha256'):
            v=entry.get(key)
            if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
                raise ValueError('Incomplete mechanical hashes')
    rng=np.random.RandomState(spec['color_seed']);plans=[]
    for i,row in enumerate(spec['layout_table']):
        family=4 if i==5 else 6 if i==7 else i
        colors=(copy.deepcopy(plans[family]['target_colors']) if i in (5,7) else
                [copy.deepcopy(prior.color_table()[int(j)]) for j in rng.choice(20,3,replace=False)])
        config=dict(protocol=PROTOCOL,split='TRAIN',seed=spec['seed_start']+i,
            row_x=row['row_x'],post_y=row['post_y'],post_heights=row['post_heights'],post_base_z=.755,
            entry_xyz=[0.,0.,.865],goal_xyz=copy.deepcopy(GOALS[row['goal_group']]),
            tip_clearance_m=.02,endpoint_tolerance_m=.03,maximum_steps_per_segment=1000,
            requested_setup_actions=0,requested_route_proposals=27,preparation_xyz=[])
        centers,halves=geometry(config);check=geometry_precheck(config)
        plans.append(dict(index=i,seed=401000+i,parent_id='layout_variation_%d'%(401000+i),
            family_id='layout_variation_family_%d'%family,role='TRAIN',split='TRAIN',config=config,
            obstacle_count=len(centers),target_colors=colors,geometry_precheck=check,
            registered_geometry_exact_sha256=physical_hash(centers,halves,config['goal_xyz']),
            registered_geometry_1mm_sha256=physical_hash(centers,halves,config['goal_xyz'],.001),
            low_gap_certificates=gap_certificates(config),guide_plans=[guide_plan(config,t) for t in range(3)]))
    # Existing gate closes all members, keeps all27 requested slots, and never replaces a parent.
    gate=prior.apply_duplicate_gate(plans,excluded)
    gate['geometry_scope']='Variable-N boxes and ordered goals; legacy-compatible encoding for N4; no color/seed/q drift.'
    for p in plans:
        p['predeclared_closed_guide_slots']=sum(not s['collection_allowed'] for row in p['guide_plans'] for s in row)
        p['requested_slots']=27
    return dict(protocol=PROTOCOL,specification=copy.deepcopy(spec),parent_plan=plans,canonical_init=spec['canonical_init'],
        excluded_hashes=copy.deepcopy(excluded),history_source_sha256=source_files,duplicate_gate=gate,
        execution_order=list(range(12)),role_counts={'TRAIN':12},requested_parents=12,requested_routes=324,
        reference_set_complete=False,all_solution_count=None,server_id_availability_verified=False,
        simulation_started=False,training_authorized=False)


def pair_evidence(opened,closed):
    """A closed low relation plus an accepted open witness, never no-sample proof."""
    if (opened['index'],closed['index']) not in ((4,5),(6,7)):
        raise ValueError('Only the two preregistered opening pairs')
    cert=[r for r in closed['actual_low_gap_certificates'] if r['row']==1 and r['label']=='gap1']
    witness=[r for r in opened['accepted_route_witnesses'] if r['actual_route_type'] is not None and r['actual_route_type'][1]=='gap1']
    established=bool(opened['initialization_passed'] and closed['initialization_passed'] and len(cert)==1 and
                     cert[0]['closed_for_this_low_relation'] and witness)
    return dict(open_index=opened['index'],closed_index=closed['index'],established=established,
        open_witnesses=witness,closed_certificate=cert,all_solution_count=None,
        conclusion='This same low middle relation is witnessed open and excluded closed' if established else
                   'Not established; no absence-of-samples inference')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();spec=read(args.config);excluded,sources=historical_metadata()
    value=build_registration(spec,excluded,sources)
    value['registration_source_sha256']={str(Path(__file__).relative_to(ROOT)):sha(__file__),str(args.config.relative_to(ROOT)):sha(args.config)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(parents=12,requested_slots=324,registration_sha256=sha(args.output),server_id_availability_verified=False)))


if __name__=='__main__':main()
