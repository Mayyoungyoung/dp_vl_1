"""One-shot fresh rendered families; frozen-model diagnostics, never training."""
import argparse, copy, json, os, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image
from scripts import paired_modes_data as paired
from scripts.run_observed_probability import ROOT, SOURCE, read, write, sha, lines

DATA = ROOT / 'data/feasible_space_generalization_v1'
RUN = ROOT / 'runs/feasible_space_v1/generalization_collection_v1'
POLICY = SOURCE / 'configs/feasible_space_generalization_v1.json'


def prepare():
    policy = read(POLICY)
    original = paired.read
    def scoped_read(path):
        value = original(path)
        if Path(path) == paired.POLICY:
            value.update(seed=policy['seed'], train_families=0, dev_families=policy['families'],
                         variants=['open','closed','shifted'])
        return value
    paired.read = scoped_read
    try: value = paired.registration()
    finally: paired.read = original
    original_plans=value['parent_plan'];expanded=[]
    for family in range(policy['families']):
        templates={p['variant']:p for p in original_plans[3*family:3*family+3]}
        for variant in policy['physical_variants']:
            p=copy.deepcopy(templates.get(variant,templates['open']))
            p.update(index=len(expanded),variant=variant);expanded.append(p)
    value['parent_plan']=expanded
    old_hashes = {p['registered_geometry_1mm_sha256'] for p in read(paired.DATA / 'registration.json')['parent_plan']}
    for p in value['parent_plan']:
        family = p['index'] // len(policy['physical_variants'])
        p['family_id'] = 'fresh_fs_family_%06d' % (641000 + family)
        p['parent_id'] = p['family_id'] + '_' + p['variant']
        p['seed'] = 641000 + p['index']; p['config']['seed'] = p['seed']
        c = p['config']; r = p['changed_row']
        if p['variant'] == 'shifted':
            c['post_y'][r] = [y + .017 for y in c['post_y'][r]]
        elif p['variant'] == 'narrow':
            midpoint = np.mean(c['post_y'][r]); c['post_y'][r] = [float(midpoint-.06), float(midpoint+.06)]
        elif p['variant'] == 'tall':
            c['post_heights'] = [h+.08 for h in c['post_heights']]
        cs, hs = paired.geom.geometry(c)
        p['registered_geometry_1mm_sha256'] = paired.quantized_hash_v2(cs, hs, c['goal_xyz'])
        p['guide_plans'] = [paired.geom.guide_plan(c, t) for t in range(3)]
        p['geometry_precheck'] = paired.geom.geometry_precheck(c)
        p['low_gap_certificates'] = paired.geom.gap_certificates(c)
        assert p['geometry_precheck']['passed']
        assert p['registered_geometry_1mm_sha256'] not in old_hashes
    new = [p['registered_geometry_1mm_sha256'] for p in value['parent_plan']]
    assert len(new) == len(set(new)) == 80
    value.update(protocol=policy['protocol'], policy=policy, source_commit=os.environ.get('CODE_COMMIT'),
                 evaluation_only=True, old_family_hash_overlap=0)
    DATA.mkdir(parents=True, exist_ok=False)
    write(DATA / 'registration.json', value)
    write(DATA / 'source_sha256.json', {str(p.relative_to(SOURCE)): sha(p)
          for directory in ('scripts', 'routeset', 'configs', 'research_feasible_space_v1')
          for p in (SOURCE / directory).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.sh')})
    print(json.dumps(dict(registered_parents=80, requests=336, old_family_hash_overlap=0)), flush=True)


def worker(index):
    paired.DATA = DATA; paired.RUN = RUN
    return paired.worker(index)


def collect():
    RUN.mkdir(parents=True, exist_ok=False)
    receipts = []
    for p in read(DATA / 'registration.json')['parent_plan']:
        i = p['index']; cmd = [str(ROOT / '.venv-sim/bin/python'), '-m',
                               'research_feasible_space_v1.generalization_data', 'worker', '--index', str(i)]
        r = dict(index=i, parent_id=p['parent_id'], command=cmd, status='running', start_unix=time.time())
        with (RUN / ('%03d.log' % i)).open('w') as log:
            child = subprocess.Popen(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
            r['pid'] = child.pid; write(RUN / ('%03d.json' % i), r); code = child.wait()
        r.update(exit_code=code, elapsed_seconds=time.time()-r['start_unix'], status='completed' if code==0 else 'failed')
        write(RUN / ('%03d.json' % i), r); receipts.append(r)
        print(json.dumps(r), flush=True)
        if code: raise RuntimeError('Fresh rendering failed; preserve parent and inspect before recovery')
    write(RUN / 'RESULTS.json', dict(receipts=receipts, render_workers=1, all_closed=True))


def export():
    paired.DATA = DATA
    paired.export()
    out = DATA / 'export'; policy = read(POLICY)
    obs, labels, meta = [lines(out / (name+'.jsonl')) for name in ('observations', 'supervision', 'metadata')]
    assert len(obs) == 240, 'Missing physical scenes must remain visible as a collection failure'
    originals = list(obs); lb = {r['id']:r for r in labels}; md = {r['id']:r for r in meta}
    parents = sorted({r['parent_id'] for r in originals if r['parent_id'].endswith('_open')})
    for i, parent in enumerate(parents):
        items = [r for r in originals if r['parent_id'] == parent]
        for variant in policy['observed_variants']:
            new_parent = parent[:-4] + variant
            directory = DATA / 'corruptions' / new_parent; directory.mkdir(parents=True, exist_ok=False)
            rgb = np.array(Image.open(items[0]['image']).convert('RGB'))
            with np.load(lb[items[0]['id']]['observation']) as a: observed = {k:a[k] for k in a.files}
            depth = observed['depth'].copy(); rng = np.random.default_rng(policy['seed']+i)
            if variant == 'noise':
                rgb = np.clip(rgb.astype(float)+rng.normal(0,3,rgb.shape),0,255).astype(np.uint8)
                depth = (depth+rng.normal(0,.002,depth.shape)).astype(depth.dtype)
            else:
                h,w = depth.shape; x0,x1 = int(.38*w),int(.62*w); y0,y1 = int(.325*h),int(.675*h)
                rgb[y0:y1,x0:x1] = 0; depth[y0:y1,x0:x1] = 0
            observed['depth'] = depth
            Image.fromarray(rgb).save(directory/'front.png'); np.savez_compressed(directory/'observation.npz', **observed)
            write(directory/'MANIFEST.json', dict(source_parent=parent, source_image_sha256=sha(items[0]['image']),
                  source_observation_sha256=sha(lb[items[0]['id']]['observation']), variant=variant, policy_sha256=sha(POLICY),
                  geometry_unchanged=True, synthetic_observation_corruption=True, seed=policy['seed']+i))
            for r in items:
                new_id = new_parent + r['id'][len(parent):]
                o = dict(r, id=new_id, parent_id=new_parent, image=str(directory/'front.png'))
                l = dict(lb[r['id']], id=new_id, parent_id=new_parent, observation=str(directory/'observation.npz'))
                m = dict(md[r['id']], id=new_id, variant=variant)
                obs.append(o); labels.append(l); meta.append(m)
    assert len(obs) == policy['expected_requests'] == 336
    for name, rows in [('observations',obs),('supervision',labels),('metadata',meta)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = read(out/'manifest.json')
    manifest.update(protocol=policy['protocol'], observations=336, families=16, evaluation_only=True,
                    physical_parents=80, corrupted_observation_parents=32,
                    source_sha256={n:sha(out/n) for n in ('observations.jsonl','supervision.jsonl','metadata.jsonl')})
    write(out/'manifest.json',manifest); print(json.dumps(manifest),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','collect','export']);p.add_argument('--index',type=int)
    a=p.parse_args();worker(a.index) if a.stage=='worker' else globals()[a.stage]()
