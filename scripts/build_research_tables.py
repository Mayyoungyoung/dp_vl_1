"""Build mixed-scope evidence tables without filling unmeasured metrics."""
import csv
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]; reports=root/'reports'; rows=[]
    historical=json.loads((reports/'multiseed_summary.json').read_text())
    for method,splits in historical['summary'].items():
        for split,metrics in splits.items():
            if split not in ['test','ood']: continue
            for index,seed in enumerate(historical['training_seeds']):
                rows.append(dict(tier='historical_controlled_CLIP_oracle',task='fixed_three_modes',protocol='historical_public',
                    method=method,seed=seed,split=split.upper(),K=3,
                    ValidAtK=metrics['valid_rate']['values'][index],AnyValidAtK=metrics['success']['values'][index],
                    UniqueValidAtK=metrics['unique_valid']['values'][index],ReferenceCoverageAtK=metrics['coverage']['values'][index],
                    source='reports/multiseed_summary.json'))
    source=reports/'v2_controlled'/'results.json'
    if source.exists():
        for row in json.loads(source.read_text()):
            rows.append(dict(tier=row['tier'],task='variable_opening_pairs',protocol='development_selection',
                method=row['objective']+'_set_regression',seed=row['seed'],split=row['split'],K=row['K'],
                ValidAtK=row['ValidAtK'],AnyValidAtK=row['AnyValidAtK'],UniqueValidAtK=row['UniqueValidAtK'],
                ReferenceCoverageAtK=row['ReferenceCoverageAtK'],head_batch1_ms=row['head_batch1_ms'],
                elapsed_s=row['elapsed_s'],gpu_hours=row['gpu_hours'],code_commit=row['code_commit'],source=str(source.relative_to(root))))
    source=reports/'v2_planner'/'results.json'
    if source.exists():
        for row in json.loads(source.read_text()):
            rows.append(dict(tier=row['tier'],task='variable_opening_pairs',protocol='analytical_special_case',method=row['method'],
                seed=None,split=row['split'],K=row['K'],ValidAtK=row['valid_rate'],AnyValidAtK=row['success'],
                UniqueValidAtK=row['unique_valid'],ReferenceCoverageAtK=row['reference_coverage'],SelectedValidAtK=row['selected_valid'],
                source=str(source.relative_to(root))))
    for source in sorted((reports/'v2_completion').glob('*/summary.json')):
        result=json.loads(source.read_text())
        for context,metrics in result['metrics'].items():
            if not isinstance(metrics,dict): continue
            rows.append(dict(tier='controlled_completion',task=context,protocol=metrics['budget_description'],
                method=source.parent.name,seed=0,split='DEV_MODEL',K=metrics['total_candidates'],
                ValidAtK=metrics['union_valid_rate'],AnyValidAtK=metrics['union_success'],UniqueValidAtK=metrics['union_unique_valid'],
                ReferenceCoverageAtK=metrics['union_reference_coverage'],additional_unique_valid=metrics['additional_unique_valid'],
                new_valid_rate=metrics['new_valid_rate'],new_candidates=metrics['new_candidates'],
                supplied_candidates=metrics['supplied_candidates'],end_to_end_controlled_ms=metrics['batch1_gate_head_check_ms_median'],
                source=str(source.relative_to(root))))
    keys=['tier','task','protocol','method','seed','split','K','ValidAtK','AnyValidAtK','UniqueValidAtK','ReferenceCoverageAtK',
          'SelectedValidAtK','additional_unique_valid','new_valid_rate','new_candidates','supplied_candidates','head_batch1_ms',
          'end_to_end_controlled_ms','elapsed_s','gpu_hours','code_commit','source']
    rows=[{key:row.get(key) for key in keys} for row in rows]
    (reports/'MAIN_RESULTS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (reports/'MAIN_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print('Wrote %d measured result rows; null means unmeasured.'%len(rows))


if __name__=='__main__':main()
