"""Paired parent-scene development analysis, including failures and artifacts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.multigate import load_dataset, path_validity, route_modes, wall_boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.runs), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, records, predictions, artifact_index = [], {}, {}, []
    data = load_dataset(a.data, 'DEV_MODEL')
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'summary.json').exists(): continue
        summary = json.loads((folder/'summary.json').read_text())
        config = json.loads((folder/'config.json').read_text())
        records[folder.name] = json.loads((folder/'dev_model'/'per_scene.json').read_text())
        archive = np.load(folder/'dev_model'/'predictions.npz')
        assert np.array_equal(archive['scene_ids'], data['scene_ids'])
        predictions[folder.name] = archive['paths']
        row = {'run_id':folder.name,'tier':'controlled_geometry','split':'DEV_MODEL',
               'objective':config['objective'],'seed':config['seed'],'K':config['candidates'],
               'steps':config['steps'],'best_step':summary['best_step'],
               'ValidAtK':summary['metrics']['valid_rate'],'AnyValidAtK':summary['metrics']['success'],
               'UniqueValidAtK':summary['metrics']['unique_valid'],
               'ReferenceCoverageAtK':summary['metrics']['reference_coverage'],
               'SelectedValidAtK':None,'head_batch1_ms':summary['metrics']['head_batch1_ms_median'],
               'elapsed_s':summary['elapsed_s'],'gpu_hours':summary['gpu_hours_reserved'],
               'gradient_target_exposures':summary['trajectory_exposures'],'peak_cuda_memory_mb':summary['peak_cuda_memory_mb'],
               'code_commit':config['code_commit'],'dataset_sha256':config['dataset_sha256']}
        rows.append(row)
        for name in ['best.pt','last.pt','summary.json','history.json','config.json','train.log','dev_model/predictions.npz','dev_model/per_scene.json']:
            path = folder/name
            artifact_index.append({'run_id':folder.name,'path':str(path.resolve()),'sha256':sha256(path),'bytes':path.stat().st_size})
    write_json(out/'results.json', rows)
    if rows:
        with open(out/'results.csv','w',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(out/'artifact_index.json', artifact_index)
    comparisons = []
    rng = np.random.default_rng(2719)
    for left in rows:
        for right in rows:
            if left['objective']!='subset' or right['objective']!='positive' or left['K']!=right['K'] or left['seed']!=right['seed']: continue
            l, r = records[left['run_id']], records[right['run_id']]
            assert [x['parent_id'] for x in l] == [x['parent_id'] for x in r]
            parents = sorted(set(x['parent_id'] for x in l))
            by_parent = {parent:[i for i,x in enumerate(l) if x['parent_id']==parent] for parent in parents}
            pair = {'baseline':left['run_id'],'comparison':right['run_id'],'split':'DEV_MODEL','parent_count':len(parents),'metrics':{}}
            for metric in ['unique_valid','valid_rate','success','reference_coverage']:
                delta = np.array([r[i][metric]-l[i][metric] for i in range(len(l))])
                means = np.array([delta[by_parent[parent]].mean() for parent in parents])
                bootstrap = means[rng.integers(len(means),size=(5000,len(means)))].mean(1)
                pair['metrics'][metric] = {'difference':float(means.mean()),'parent_bootstrap_ci95':np.quantile(bootstrap,[.025,.975]).tolist()}
            comparisons.append(pair)
    write_json(out/'paired_comparisons.json', comparisons)
    breakdown = {}
    counts = data['path_mask'].sum(1)
    for name, pred in predictions.items():
        records_by_type = []
        for count in sorted(set(counts)):
            indices = np.flatnonzero(counts==count)
            record = {'reference_types':int(count),'parents':len(indices)}
            for metric in ['valid_rate','unique_valid','success','reference_coverage','collision_rate']:
                record[metric] = float(np.mean([records[name][i][metric] for i in indices]))
            records_by_type.append(record)
        failures = []
        for idx,(scene,paths) in enumerate(zip(data['scenes'],pred)):
            checks = path_validity(paths,scene); modes = route_modes(paths,scene)
            failures.append({'scene_id':str(data['scene_ids'][idx]),'reference_types':int(counts[idx]),
                             'valid':checks['valid'].tolist(),'collision':checks['collision'].tolist(),
                             'modes':modes.tolist(),'lengths':checks['lengths'].tolist(),
                             'in_bounds':checks['in_bounds'].tolist(),'endpoint_error':checks['endpoint_error'].tolist()})
        breakdown[name] = {'by_reference_types':records_by_type, 'per_scene':failures}
    write_json(out/'failure_analysis.json', breakdown)
    if predictions:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        names = list(predictions)
        # Fixed first four DEV_MODEL scenes: no favorable selection.
        fig, axes = plt.subplots(4,len(names),figsize=(5*len(names),14),squeeze=False)
        for col,name in enumerate(names):
            for idx in range(4):
                ax = axes[idx,col]; scene = data['scenes'][idx]
                lo,hi = wall_boxes(scene)
                for lower,upper in zip(lo,hi):
                    ax.add_patch(patches.Rectangle(lower[:2],upper[0]-lower[0],upper[1]-lower[1],color='.6'))
                for j,path in enumerate(predictions[name][idx]):
                    good = bool(path_validity(path,scene)['valid'])
                    ax.plot(path[:,0],path[:,1],'-' if good else '--',label='{} {}'.format(j,'valid' if good else 'invalid'))
                ax.scatter(scene[[0,3]],scene[[1,4]],c=['black','red'],zorder=5)
                ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',title='{} / DEV {} / {} types'.format(name,idx,counts[idx]))
                ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out/'first_four_dev.png',dpi=140); plt.close(fig)
    print(json.dumps({'rows':rows,'paired':comparisons}),flush=True)


if __name__=='__main__': main()
