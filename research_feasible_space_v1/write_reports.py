"""Render research Markdown tables directly from delivered evidence JSON."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
E=HERE/'results'


def read(name):
    return json.loads((E/(name+'.json')).read_text(encoding='utf-8'))


def metrics(v):
    return '|%.2f%%|%.4f|%.2f%%|%.4f|%.2f%%|'%(100*v['V8'],v['U8'],100*v['V4'],v['U4'],100*v['known_recall'])


def main():
    stats=read('three_seed_statistics_v1');s=read('summary_v2' if (E/'summary_v2.json').exists() else 'summary_v1')
    witness=read('fixed_witness_v1');diag=read('saved_outcome_diagnostic_v1');visibility=read('observation_visibility_v2')
    names={'xyz':'Same-information free XYZ','bounded':'Proposed bounded mapping','projection':'Shared-head projection',
           'center':'Shared-head XYZ center','boundcenter':'Shared-head bounded center','refitted_projection':'Projection, fresh head',
           'refitted_center':'XYZ center, fresh head','refitted_boundcenter':'Bounded center, fresh head'}
    report=['# Feasible-space v1 results', '',
            '**The implemented bounded decoder does not pass the preregistered mechanism gate; publication readiness is not established.**', '',
            'The final three paired generator continuations show a small benefit over same-information XYZ, but a clear deficit to returning predicted centerlines. Two measured-cause repairs improved region feasibility; neither established a necessary neural interior generator. Historical defaults remain unchanged.', '',
            '## Scope and fixed protocol', '',
            'TRAIN: 1,152 requests / 128 families / 56,920 verified positive paths. Existing DEV_MODEL: 288 requests / 32 repeatedly reused families. Seeds 0–2 are full generator continuations of shared pretrained C0, with exactly matched initialization and sampled streams within each pair. They are not independent VLM/pretraining runs. Seed 0 was reused from exploratory screening; the replication lock preceded reading its final outcome, not its launch.', '',
            'Each final generator receives 2,400 updates and a matching ordinary success head trained for 2,400 updates on fresh TRAIN outcomes. Center/projection controls receive their own hashed decoder views, fresh feedback and heads. Exactly eight candidates are decoded, followed by the complete original frozen scorer returning four. Truth geometry is used only for TRAIN supervision and independent checking.', '',
            'The unchanged practical gate requires ≥0.15 additional actual valid modes@8 with a positive paired interval against all matched controls; guards require ΔV8≥−0.01, ΔV4≥−0.005 and ΔU4≥−0.03. Conditional 10,000-draw bootstrap intervals resample seed and scene family. They do not establish untouched-test generalization.', '',
            '## Final existing-DEV comparison', '',
            '|Method, mean of three generator continuations|V8|U8|V4|U4|Known recall|',
            '|---|---:|---:|---:|---:|---:|']
    for k in ('xyz','bounded','refitted_projection','refitted_center','refitted_boundcenter'):
        report.append('|'+names[k]+metrics(stats['means'][k]))
    report += ['', 'Historical references use different training protocols and are not paired causal controls:', '',
               '|Historical method|V8|U8|V4|U4|', '|---|---:|---:|---:|---:|',
               '|C + ordinary success, five head seeds on C0|89.15%|6.9806|94.34%|3.7743|',
               '|Gate, three historical seeds|92.12%|7.2380|94.33%|3.7720|', '',
               'Bounded mapping remains below both historical references on raw coverage and returned quality. Even the stronger center-only control does not dominate Gate or recover the historical C return quality.', '',
               '|Bounded minus control|ΔU8|Crossed 95% CI|Practical gain passes|Quality guards pass|',
               '|---|---:|---|---|---|']
    for k in ('xyz','refitted_projection','refitted_center','refitted_boundcenter'):
        c=stats['comparisons']['bounded minus '+k];lo,hi=c['crossed_CI95'][0]
        report.append('|%s|%+.4f|[%+.4f, %+.4f]|%s|%s|'%(names[k],c['mean'][0],lo,hi,c['primary_gain'],c['quality_guards']))
    report += ['', '**Overall accepted: `%s`.** A statistically positive smaller effect cannot replace the +0.15 practical gate.'%stats['accepted'], '',
               '## Every generator seed', '', '|Method|Seed 0 U8|Seed 1 U8|Seed 2 U8|', '|---|---:|---:|---:|']
    for k in ('xyz','bounded','refitted_projection','refitted_center','refitted_boundcenter'):
        report.append('|%s|%s|'%(names[k],'|'.join('%.4f'%v['U8'] for v in stats['seed_results'][k])))
    report += ['', 'All three XYZ/bounded initial-tensor hashes and actual training-stream digests match within seed. No seed extension or checkpoint selection followed these results.', '',
               '## Reference representation and oracle diagnosis', '',
               'All 56,920 TRAIN witnesses admit individual geometric certificates. Two per-word prototype corridors cover 51,211 / 56,920 witnesses (89.9701%); all 20,336 prototype centerlines are task-valid and correct-word. No medoid fallback was needed. This establishes representation capacity under oracle TRAIN geometry, not deployable perception.', '',
               'In the 1,024-route TRAIN reference-cell diagnostic, center interpolation, free XYZ and bounded mapping all reach 100% validity. Correct requested words are 1,024, 1,022 and 1,023 respectively. The oracle experiment does not demonstrate a learned-interior advantage; its centerline is already a verified feasible witness.', '',
               '## Fixed cross-edit witnesses', '',
               '|Model|Retained / 2169|Retention|New losses / 1872|Recovered / 297|Same-mode repair / 270|',
               '|---|---:|---:|---:|---:|---:|']
    for k,v in witness['models'].items():
        report.append('|%s|%d|%.2f%%|%d|%d|%d|'%(k,v['retained'],100*v['retention'],v['lost_of_1872'],v['recovered_of_297'],v['adaptation_counts'].get('valid_same_mode_repair',0)))
    report += ['', 'The denominators are the unchanged parent-defined 2,169 surviving-mode and 270 invalid-old/same-mode-known-new opportunities. Bounded seed 0 repairs more opportunities than the parent but loses more surviving modes; the center control repairs still more. Requested query identity is not treated as actual realized mode identity.', '',
               '## Frozen fresh-family diagnosis', '']
    if (E/'generalization_statistics_v1.json').exists():
        fresh=read('generalization_statistics_v1')
        report += [fresh['scope']+'.', '',
                   'All model/head checkpoints were frozen before collecting these observations. No model selection, update or weight search uses them. All 16 families are new; their physical geometry hashes were checked against the old paired TRAIN/DEV registration. Eighty scenes were actually rendered; 32 additional observation variants are explicitly synthetic RGB/depth corruptions of the open scenes. This is task-level simulation evidence, not whole-arm or real-robot execution.', '',
                   '|Frozen method, three-seed mean|V8|U8|V4|U4|Sparse teacher recall|', '|---|---:|---:|---:|---:|---:|']
        for k in ('xyz','bounded','refitted_projection','refitted_center','refitted_boundcenter'):
            report.append('|'+names[k]+metrics(fresh['means'][k]))
        report += ['', '|Variation|XYZ U8|Bounded U8|Refitted XYZ center U8|', '|---|---:|---:|---:|']
        for v in sorted(fresh['seed_results']['xyz'][0]['variants']):
            means=[sum(r['variants'][v]['U8'] for r in fresh['seed_results'][k])/3 for k in ('xyz','bounded','refitted_center')]
            report.append('|%s|%.4f|%.4f|%.4f|'%(v,*means))
        report += ['', '|Bounded minus control, fresh families|ΔU8|Conditional crossed 95% CI|', '|---|---:|---|']
        for k,c in fresh['comparisons'].items():
            lo,hi=c['crossed_CI95'][0];report.append('|%s|%+.4f|[%+.4f, %+.4f]|'%(k,c['mean'][0],lo,hi))
        report += ['', 'The fresh reference set is incomplete geometric teacher evidence; its recall must not be compared directly to the old all-mode-support recall. Noise and masking diagnose robustness, not calibrated confidence. This extension does not retroactively relax or replace the failed original mechanism gate.']
    else:report += ['The fixed16-family/7-variant/336-request collection is running. No fresh-generalization success is claimed before receipt closure.']
    report += ['', '## Verification, uncertainty and cost', '',
               'Server tests cover membership, fixed endpoints, candidate-boundary independence, relative decoder gradients, exact tapered clearance and visibility unknown handling. Actual 100-step versus 50+50 recovery has identical model, optimizer, RNG, history, sample stream and settings. The public API reproduces saved paths/events/q and selected indices exactly on 16 requests, with one decoder call per request. Cached-feature API timing excludes online VLM feature extraction.', '']
    for k in ('deployment_xyz_v2','deployment_bounded_v2'):
        a=read(k);report.append('- %s: %.2f ms/request, maximum replay error %.1g; selected indices exact.'%(k,a['api_cached_feature_ms_mean'],a['maximum_path_event_q_error']))
    a=visibility['models']['refreshed_tapered_bounded_seed0']
    report += ['', 'Depth-only calibrated corner probes classify %.2f%% as unknown. Mean unknown fraction is %.2f%% on valid routes and %.2f%% on invalid routes. These sparse probes are not whole-cell certificates and do not constitute a learned uncertainty predictor.'%(100*a['probe_fractions']['unknown'],100*a['unknown_mean_valid'],100*a['unknown_mean_invalid']), '']
    if (E/'CLOSURE.json').exists():
        c=read('CLOSURE');report += ['Closed ledger: %d jobs, %d completed and %d failed; %.1f measured command seconds (%.2f hours). All failures, source hashes, actual commands, checkpoints/optimizer/RNG state and predictions are retained. Artifact-only archive/verification costs are listed separately in CLOSURE.json.'%(c['receipt_count'],c['completed'],c['failed'],c['command_seconds'],c['command_seconds']/3600), '']
    report += ['No reserved TEST_LOCKED payload or metric was inspected. Historical d0d97eb and runs/main remain preserved. Only wzy3090, physical GPU1 UUID7506746b-d0ba-f6fe-44ce-8a1f97dde2ab,35%memory and CPU0–3 were used; rendering uses one worker/software GL. No shared environment was changed.', '',
               '## Decision', '',
               'The conditional containment guarantee is implemented and independently checked; the learned region remains an estimate. H1 shows a small relative effect versus XYZ but fails meaningful-benefit and strongest-control requirements. H2 boundary invariance holds structurally, yet the center control has the same protection and better task quality. H3 must be read from fixed-witness and frozen fresh-family results; it cannot be inferred from a few favorable route plots. There is no evidence here for a publication-ready novel interior decoder or a deployment-default switch.', '',
               'Code, method/proof, full ablations, causal limitations, paper draft, real saved-prediction figures, reproduction instructions and hash receipts are delivered. The requested publication-potential goal remains unmet; no acceptance or real-robot result is invented.']
    (HERE/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

    ab=['# Ablations and repairs','', 'Every exploratory screen uses seed0 and is labeled as development evidence. Final acceptance uses the locked three-seed paired representation and fresh-head strong controls.','',
        '|Exploratory arm|V8|U8|V4|U4|Region geometry feasible|Membership|', '|---|---:|---:|---:|---:|---:|---:|']
    for n in ('screen_xyz_seed0','screen_relative_seed0','screen_bounded_seed0','screen_peer_seed0','envelope_xyz_seed0','envelope_bounded_seed0','refreshed_envelope_xyz_seed0','refreshed_envelope_bounded_seed0','tapered_xyz_seed0','tapered_bounded_seed0','refreshed_tapered_xyz_seed0','refreshed_tapered_bounded_seed0'):
        v=s['models'][n];ab.append('|%s|%.2f%%|%.4f|%.2f%%|%.4f|%.2f%%|%.2f%%|'%(n,100*v['raw']['valid_fraction'],v['raw']['distinct'],100*v['selected']['valid_fraction'],v['selected']['distinct'],100*v['corridor_feasibility'],100*v['containment']))
    ab += ['', 'Initial XYZ/bounded use identical features/modules/init/draws, differing in the mapping. Relative removes the saturating bound while retaining center/width scaling. Peer boundaries enable cross-query attention in the corridor branch. None receives extra oracle geometry at inference.', '',
           'Uniform envelope supervision addresses optimistic reference widths around shifted predicted centers. Tapered envelope supervision addresses unreachable cubes around fixed endpoints. Both repairs apply identically to XYZ and bounded arms, retain the160clearance coefficient and 2,400-update schedule, and were registered before outcomes. Region definitions differ: uniform swept boxes versus the tapered reachable set; their feasibility rates are not an invariant common-region metric.', '',
           '## Shared-head geometric views versus standalone controls','', '|View, three-seed mean|V8|U8|V4|U4|Known recall|','|---|---:|---:|---:|---:|---:|']
    for k in ('projection','refitted_projection','center','refitted_center','boundcenter','refitted_boundcenter'):
        ab.append('|'+names[k]+metrics(stats['means'][k]))
    ab += ['', 'Shared views isolate the path construction for identical queries. Fresh-head views additionally account for changed realization labels and query allocation. Main conclusions pass neither center control; inherited mismatched heads cannot explain away the negative result.', '',
           '## Companion query perturbation','', '|Arm|Mean companion path motion mm|Mean maximum boundary change mm|Valid companions lost/request|Raw words changed/request|', '|---|---:|---:|---:|---:|']
    for n in ('screen_xyz_seed0','screen_relative_seed0','screen_bounded_seed0','screen_peer_seed0','refreshed_tapered_xyz_seed0','refreshed_tapered_bounded_seed0','refreshed_tapered_bounded_seed1','refreshed_tapered_bounded_seed2'):
        v=s['stability'][n];ab.append('|%s|%.3f|%.3f|%.4f|%.4f|'%(n,1000*v['mean_companion_motion_m'],1000*v['corridor_boundary_motion_m'],v['valid_companions_lost'],v['mode_companions_changed']))
    ab += ['', 'One slot is changed while the other seven query/variant identities stay fixed. Net U8 includes the deliberately changed slot and must not be interpreted as companion-only harm. Independent boundaries remain invariant; peer-aware interior parameters can still change geometry and realized words. Returning independent centerlines eliminates these interior changes by construction.', '']
    if 'shape' in next(iter(s['models'].values())):
        ab += ['## Width, shape and frozen scorer','', '|Arm|Width p10/median/p90 mm|Cells below2mm|Mean valid length m|Mean turn rad|Degenerate segments|Missed available valid slots|','|---|---|---:|---:|---:|---:|---:|']
        for n in ('refreshed_tapered_xyz_seed0','refreshed_tapered_bounded_seed0','refitted_control_center_seed0','refitted_control_boundcenter_seed0'):
            v=s['models'][n];q=v['width_quantiles_m'];sh=v['shape'];ab.append('|%s|%.2f / %.2f / %.2f|%.2f%%|%.4f|%.4f|%.4f%%|%d|'%(n,1000*q[1],1000*q[2],1000*q[3],100*v['cells_below_2mm'],sh['mean_length_valid_m'],sh['mean_turning_angle_rad'],100*sh['degenerate_segment_fraction'],v['scorer_missed_available_valid_count']))
        ab += ['', 'Turning angle describes the delivered polyline, not smooth robot motion. Missed available valid slots uses the checker only after selection; it is an analysis upper bound, never an oracle return rule.']
    (HERE/'ABLATIONS.md').write_text('\n'.join(ab)+'\n',encoding='utf-8')

    fail=['# Failure analysis','', 'The representation is expressive under reference geometry but does not establish useful learned interior generation. The following mechanisms are supported by actual outputs; associations are separated from causal controls.','',
          '## What the controlled comparisons establish','',
          '1. Reference centerlines already solve the oracle diagnostic. Allthree approaches are100%valid; the bounded decoder does not add oracle capability and occasionally changes the requested word.',
          '2. Merely giving XYZ the same predicted geometry produces most of the task quality. Bounded mapping adds only0.08565U8 over XYZ, below the registered0.15gate.',
          '3. Removing interior generation is a stronger control. Fresh-head centers gain0.30671U8 over bounded mapping; the crossed interval excludes zero. Geometric containment alone cannot establish neural necessity.',
          '4. Boundary independence is architectural. Center-only generation shares it, so invariant boundaries are not an independent benefit of the relative neural decoder.',
          '5. Repair gains trade against retention. Boundedseed0 repairs224/270 but retains1727/2169, below historical1872/2169. Center-only repairs243/270 and retains1760/2169. Neither supports a claim that the proposed mechanism preserves prior feasible options.','',
          '## Where route validity fails','', '|Saved seed0 output|Certified + inside|Geometry violations of implication|Endpoint failures within certified cell|Wrong center raw word|Output word changed from center|','|---|---:|---:|---:|---:|---:|']
    for n,v in diag['models'].items():
        fail.append('|%s|%d|%d|%d|%d|%d|'%(n,v['certified_and_inside'],v['certified_inside_geometry_failure'],v['certified_inside_endpoint_failure'],v['wrong_center_raw_mode'],v['output_raw_mode_changed_from_center']))
    fail += ['', 'Zero certified-and-inside geometry failures independently supports the conditional geometric implication on saved outputs. It says nothing about uncertified predictions, goal correctness or passage identity. Boundedseed0 has643/2304uncertified regions and78goal failures inside certified regions. There are103wrong center raw words and59word changes induced by the output mapping. These categories overlap and are not additive failure partitions.', '',
             '## Measured-cause repairs','',
             'The initial XYZ weighted clearance gradient norm0.22190 is about11.1times its route gradient0.01997. Final-path clearance does not directly supervise the whole cell. Reference-centered widths are optimistic after center prediction errors; whole-envelope supervision raises cell feasibility from7.55%to71.18%. This is a geometric supervision improvement, not proof of task improvement.', '',
             'Uniform endpoint cubes create pressure on regions the fixed-endpoint decoder cannot visit. XYZ goal failures rise137→172 while geometry failures fall185→167. Tapered endpoint cells remove this unreachable volume and restore some route quality, but the final three-seed comparison remains below center-only and historical references.', '',
             '## Observation and deployment limitations','',
             'Single-view depth leaves unknown space. About9.23%of bounded node-corner probes are unknown, with nearly identical fractions for valid and invalid routes. Sparse probe visibility does not explain or certify all region errors. The implementation does not learn calibrated cell confidence, abstain on missing evidence or explicitly model hidden occupancy; the public API labels actual feasibility unknown.', '',
             'Adjacent regions share a node cube and are structurally connected, including zero-width endpoint nodes. This avoids a separate empty-intersection construction failure but cannot make a colliding or wrong-mode corridor physically valid. The23degree-one Bézier pieces are precisely the24-node public polyline; they provide positional continuity only. No smoothness/dynamics/full-arm guarantee is inferred.', '',
             'The frozen scorer may miss available valid routes; the ablation table records this separately. It cannot repair a weak candidate pool and was not retrained using evaluation truth. The API timing uses cached frozen VLM features, not full online RGB-D-to-result latency.', '',
             '## Failures retained','',
             'Failed server test on Sequential.weight access (test bug); failed oracle diagnostic JSON serialization (saved paths retained); failed fresh-family prepare before data creation (inherited uniqueness check before new geometry edits). Each correction uses new source/output/job IDs. Pre-launch release path typos and local test-discovery/rg glob errors are command mistakes, not model outcomes. No failed experiment is relabeled successful.', '',
             'Older figure v1 renders an oversized uniform envelope and picks a different comparison seed. The delivered actual_reachable_figures_v2 uses true tapered reachable cells and actual paired seed0 XYZ/bounded outcomes, retaining both improvement and loss cases. Visual examples never replace aggregate gates.', '',
             '## Research decision','',
             'Do not extend interior-decoder loss searches or claim a new corridor-planning principle from these results. A further mechanism would need to solve observed region/goal/mode prediction errors and justify why simple centers cannot deliver the same ability. Fresh-family results are frozen diagnosis; they are not permission to optimize against a new development set. TEST_LOCKED and whole-arm validation remain unopened. The publication-potential goal is unmet.']
    (HERE/'FAILURE_ANALYSIS.md').write_text('\n'.join(fail)+'\n',encoding='utf-8')


if __name__=='__main__':main()
