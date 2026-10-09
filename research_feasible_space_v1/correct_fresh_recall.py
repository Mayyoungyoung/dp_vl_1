"""Correct a legacy/operational mode-label mismatch from sealed predictions only."""
import argparse,json
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.generalization_data import DATA
from scripts.run_observed_probability import read,write,sha,lines
from scripts.evaluate_paired_modes import references
from scripts.research_v3_audit import mode,average


def main(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'};known={};reference_hashes={}
    for k,r in labels.items():
        ref=references(r);known[k]={mode(p,ref['config']) for p,valid in zip(ref['paths'],ref['reference_valid']) if valid};known[k].discard(None)
        for f in r['routes']:reference_hashes[f]=sha(f)
    assert len(known)==336 and all(known.values())
    corrections={}
    for folder in sorted(RUN.glob('generalization_*/eval_adaptive')):
        rows=read(folder/'rows.json');metric=read(folder/'metrics.json');before=metric['raw']['recall']
        for row in rows:
            for key,slots in [('raw',range(8)),('selected',row['selected_indices'])]:
                hit={row['words'][j] for j in slots if row['valid'][j] and row['words'][j] is not None}
                row[key]['recall']=len(hit & known[row['id']])/len(known[row['id']])
        target=out/folder.parent.name;target.mkdir()
        metric['raw']['recall']=average([r['raw'] for r in rows])['recall'];metric['selected']['recall']=average([r['selected'] for r in rows])['recall']
        metric.update(recall_definition='Sparse checked reference paths classified by research_v3 first-forward operational words; legacy portal strings superseded',
                      original_metrics_sha256=sha(folder/'metrics.json'),original_rows_sha256=sha(folder/'rows.json'))
        write(target/'rows.json',rows);write(target/'metrics.json',metric)
        corrections[folder.parent.name]=dict(original_recall=before,corrected_recall=metric['raw']['recall'],all_other_predictions_and_metrics_unchanged=True)
    assert len(corrections)==15
    report=dict(models=corrections,reference_sha256=reference_hashes,reference_mode_counts={k:len(v) for k,v in known.items()},
                failure='Fresh recall compared legacy full-portal reference strings with operational first-forward prediction labels',
                correction='Reclassify valid reference paths with the exact prediction mode definition; do not re-decode, update or overwrite saved results',
                added_decodes=0,added_model_updates=0,locked_access=False)
    write(out/'MANIFEST.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('reference_sha256','reference_mode_counts')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);main(**vars(p.parse_args()))
