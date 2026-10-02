"""CPU-only descriptive analysis of the completed fixed nine-forward TRAIN audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json

SUMMARY_SHA = '8b1a1fed1ad7d23b82178f7bba8143aabe4598a278a48400925237f4b3c195e4'


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.percentile(x,95)), min=float(x.min()), max=float(x.max())) if len(x) else dict(count=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original train8 output directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh independent descriptive output required')
    if sha256(args.input/'summary.json') != SUMMARY_SHA:
        raise ValueError('Exact frozen actual audit required')
    summary = json.loads((args.input/'summary.json').read_text())
    if summary['status'] != 'completed' or summary['actual_model_forwards'] != 9 or len(summary['records']) != 8:
        raise ValueError('Complete fixed eight TRAIN references plus causal control required')
    for name, receipt in summary['artifacts'].items():
        path = (args.input/name).resolve()
        if args.input.resolve() not in path.parents or sha256(path) != receipt['sha256']:
            raise ValueError('Original audit artifact changed')
    interface = json.loads((args.input/'interface.json').read_text())
    categories = summary['aggregate_token_categories']
    scalars = [s for r in summary['records'] for s in r['scalars'] if s['category'] == 'coordinate']
    parsed = [s for s in scalars if s['predicted_integer'] is not None]
    errors = [s['signed_integer_error'] for s in parsed]
    nll_total = sum(v['nll_sum'] for v in categories.values())
    rows = []
    for r in summary['records']:
        coords = [s for s in r['scalars'] if s['category'] == 'coordinate']
        usable = [s for s in coords if s['predicted_integer'] is not None]
        rows.append(dict(scene_id=r['scene_id'], conditional_nll=r['per_token_nll_mean'],
            coordinate_scalars=72, parsed_coordinate_scalars=len(usable),
            parsed_coordinate_absolute_error_mm=describe([abs(s['signed_integer_error']) for s in usable]),
            conditional_endpoint_error_to_reference_m=r['conditional_endpoint_error_to_reference_m'],
            saved_free_endpoint_error_to_same_reference_m=r['saved_free_endpoint_error_to_same_reference_m']))
    result = dict(protocol='vlm_teacher_audit_descriptive_analysis_v1', source_summary_sha256=SUMMARY_SHA,
        source_commit=summary['code_commit'], analyzer_sha256=sha256(__file__),
        additional_model_forwards=0, additional_raw_observation_or_reference_reads=0,
        categories=categories, aggregate_token_nll=summary['aggregate_token_nll'],
        coordinate_share_of_total_nll=categories['coordinate']['nll_sum']/nll_total,
        original_loss_max_absolute_difference=max(r['loss_absolute_difference'] for r in summary['records']),
        maximum_coordinate_roundtrip_error_m=max(max(i['roundtrip_max_absolute_error_m_by_axis']) for i in interface),
        coordinate_scalars=len(scalars), parsed_coordinate_scalars=len(parsed), unparseable_coordinate_scalars=len(scalars)-len(parsed),
        parsed_coordinate_signed_error_mm=describe(errors), parsed_coordinate_absolute_error_mm=describe(np.abs(errors)),
        exact_parsed_coordinates=sum(e == 0 for e in errors),
        first_point_coordinates=sum(s['point'] == 0 for s in scalars),
        first_point_exact_parsed_coordinates=sum(s['point'] == 0 and s['signed_integer_error'] == 0 for s in parsed),
        conditional_endpoint_error_to_reference_m=describe([r['conditional_endpoint_error_to_reference_m'] for r in rows
            if r['conditional_endpoint_error_to_reference_m'] is not None]),
        conditional_unparseable_endpoints=sum(r['conditional_endpoint_error_to_reference_m'] is None for r in rows),
        saved_free_endpoint_error_to_same_reference_m=describe([r['saved_free_endpoint_error_to_same_reference_m'] for r in rows]),
        causal_control=summary['causal_control'], body_seconds=summary['body_seconds'],
        elapsed_seconds=summary['elapsed_seconds'], gpu_hours_reserved=summary['gpu_hours_reserved'],
        peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes'], rows=rows,
        caveat='Teacher top1 fragments each condition on true prior tokens, including true earlier digits and route points. '
               'Good conditional endpoints do not establish observed target grounding or free trajectory quality. '
               'Parsed-only scalar errors have explicit failure counts. No task-level success score is computed.')
    write_json(args.output, result)
    print(json.dumps({k:result[k] for k in ('coordinate_share_of_total_nll','parsed_coordinate_scalars',
        'unparseable_coordinate_scalars','parsed_coordinate_absolute_error_mm','conditional_endpoint_error_to_reference_m',
        'conditional_unparseable_endpoints','saved_free_endpoint_error_to_same_reference_m')}, indent=2))


if __name__ == '__main__':
    main()
