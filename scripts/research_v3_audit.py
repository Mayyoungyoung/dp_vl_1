"""Recheck only registered TRAIN/DEV and frozen nonlocked candidate pools.

Full request identity includes actual RGB and depth/camera/current-state hashes,
language, and coordinate convention. Oracle labels are audit-only inputs.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT, read, write, sha, lines
from scripts.paired_modes_data import DATA, RUN
from scripts.evaluate_paired_modes import references, check_candidates
from scripts.observed_layout_variation import crossing_signature, gap_certificates
from scripts.analyze_paired_selection import select
from routeset.geometry import segment_aabb_intersection
from routeset.geometric_modes import portal_word, encode_word


def mode(path, cfg):
    value = crossing_signature(path, cfg)
    return None if value is None else '|'.join(value)


def probes_clear(paths, centers, halves):
    points = np.concatenate([paths, (paths[:, 1:] + paths[:, :-1])/2], axis=1)
    inside = ((points[:, :, None] >= centers[None, None]-halves[None, None]-.02) &
              (points[:, :, None] <= centers[None, None]+halves[None, None]+.02)).all(-1)
    return ~inside.any((1, 2))


def exact_clear(paths, centers, halves):
    hit = segment_aabb_intersection(paths[:, :-1, None], paths[:, 1:, None],
                                   centers[None, None]-halves[None, None]-.02,
                                   centers[None, None]+halves[None, None]+.02)
    return ~hit.any((1, 2))


def coverage(words, valid, refs, selected):
    groups = {words[j] for j in selected if valid[j] and words[j] is not None}
    return dict(distinct=len(groups), recall=len(groups & refs)/len(refs) if refs else None,
                valid_fraction=float(valid[selected].mean()), any_valid=int(valid[selected].any()),
                all_valid=int(valid[selected].all()))


def average(rows):
    return {k:float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows) else None for k in rows[0]}


def main(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    inputs = {}; refs = {}; counts = Counter(); records = []; families = defaultdict(set)
    obs = lines(DATA/'export/observations.jsonl'); labels = lines(DATA/'export/supervision.jsonl')
    meta = {r['id']:r for r in lines(DATA/'export/metadata.jsonl')}
    byid = {r['id']:r for r in labels}
    for name in ('observations', 'supervision', 'metadata'):
        inputs[str(DATA/'export'/('%s.jsonl'%name))] = sha(DATA/'export'/('%s.jsonl'%name))
    image_hashes = {}; request_keys = {}; observation_hashes = {}
    for row in obs:
        assert row['split'] in ('TRAIN', 'DEV_MODEL')
        label = byid[row['id']]; assert label['split'] == row['split']
        m = meta[row['id']]; families[m['family_id']].add(row['split'])
        if row['image'] not in image_hashes: image_hashes[row['image']] = sha(row['image'])
        op = label['observation']
        if op not in observation_hashes:
            with np.load(op) as a:
                content = {k:hashlib.sha256(a[k].tobytes()).hexdigest() for k in
                           ('depth', 'camera_intrinsics', 'camera_extrinsics', 'gripper_pose', 'gripper_open')}
            observation_hashes[op] = content
        identity = dict(rgb=image_hashes[row['image']], instruction=row['instruction'],
                        observed_state=observation_hashes[op], coordinates='camera_to_world; XYZ meters')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        assert key not in request_keys, 'Duplicate complete request: '+row['id']
        request_keys[key] = row['id']
        ref = references(label); refs[row['id']] = ref
        valid = np.array(ref['reference_valid'], dtype=bool)
        modes = [mode(p, ref['config']) if v else None for p, v in zip(ref['paths'], valid)]
        known = Counter(w for w in modes if w is not None)
        words = Counter(w for w in ref['words'] if w is not None)
        certificates = gap_certificates(ref['config'])
        rec = dict(id=row['id'], split=row['split'], family=m['family_id'], variant=m['variant'],
                   complete_request_sha256=key, references=len(valid), valid_h24=int(valid.sum()),
                   relation_counts=dict(known), portal_counts=dict(words),
                   unknown_valid_relations=int(sum(v and w is None for v, w in zip(valid, modes))),
                   reference_set_complete=label['reference_set_complete'], certificates=certificates)
        records.append(rec)
        counts[row['split']+'_requests'] += 1
        for k in ('references', 'valid_h24', 'unknown_valid_relations'):counts[k] += rec[k]
        for k in ('route_config', 'verification_only'):
            if label[k] not in inputs:inputs[label[k]] = sha(label[k])
        for route in label['routes']: inputs[route] = sha(route)
    assert all(len(x)==1 for x in families.values()), 'Paired family leakage'
    role_parents = {}
    for role in ('SCORE_TRAIN', 'DEV_SCORE', 'CALIBRATION', 'FUTURE_GENERATOR_TRAIN'):
        p = ROOT/'data/observed_probability_v2'/role/'observations.jsonl'
        role_parents[role] = sorted({r['parent_id'] for r in lines(p)})
        inputs[str(p)] = sha(p)
    for a in role_parents:
        for b in role_parents:
            if a != b:assert not set(role_parents[a]) & set(role_parents[b])
    assert not set(r['parent_id'] for r in obs) & set(sum(role_parents.values(), []))
    write(out/'data_requests.json', records)
    recmap = {r['id']:r for r in records}
    results = {}; errors = {}; allrows = {}
    for seed in range(3):
        folder = RUN/'evaluation/paired_dev'/('R1_seed%d'%seed)
        with np.load(folder/'pool.npz') as a: pool = {k:a[k] for k in a.files}
        assert sha(folder/'pool.npz') == read(folder/'receipt.json')['pool_sha256']
        cp = RUN/'reliability'/('R1_seed%d'%seed)/('calibration_seed%d'%seed)/'paired_dev.npz'
        with np.load(cp) as a:
            np.testing.assert_array_equal(a['ids'], pool['ids']); q = a['q']
        inputs[str(folder/'pool.npz')] = sha(folder/'pool.npz'); inputs[str(cp)] = sha(cp)
        rows = []; taxonomy = Counter(); probe_missed = 0; mismatch = 0; grouped = defaultdict(list)
        for i, ident in enumerate(pool['ids']):
            ref = refs[str(ident)]; cfg = ref['config']; paths = pool['paths'][i]
            _, cand = check_candidates(paths, pool['events'][i], ref['label'], ref['current'], ref['truth'], cfg)
            valid = np.array([c['TipValid'] for c in cand]); assert np.array_equal(valid, pool['labels'][i])
            words = [mode(p, cfg) if v else None for p, v in zip(paths, valid)]
            portals = [encode_word(portal_word(p, cfg)) if v else None for p, v in zip(paths, valid)]
            known = set(recmap[str(ident)]['relation_counts']); refportal = set(recmap[str(ident)]['portal_counts'])
            chosen = select(paths, q[i], 4); top = int(np.argmax(q[i]))
            raw = coverage(words, valid, known, list(range(8))); sel = coverage(words, valid, known, chosen)
            rawp = coverage(portals, valid, refportal, list(range(8))); selp = coverage(portals, valid, refportal, chosen)
            hit = exact_clear(paths, ref['truth']['obstacle_centers'], ref['truth']['obstacle_halfsizes'])
            probe = probes_clear(paths, ref['truth']['obstacle_centers'], ref['truth']['obstacle_halfsizes'])
            probe_missed += int((probe & ~hit).sum())
            mismatch += int(sum(bool(h) != bool(c['post_segments_clear']) for h, c in zip(hit, cand)))
            taxonomy['zero_valid_requests'] += int(not valid.any())
            taxonomy['partial_known_relation_coverage_requests'] += int(valid.any() and raw['recall'] < 1)
            taxonomy['score_top1_miss_with_valid_pool'] += int(valid.any() and not valid[top])
            taxonomy['k4_extra_relation_loss_beyond_capacity'] += max(0, min(4, raw['distinct'])-sel['distinct'])
            taxonomy['invalid_candidates'] += int((~valid).sum())
            for c in cand:
                for key in ('semantic_goal_correct', 'post_segments_clear', 'workspace_floor_correct', 'event_state_sequence_correct'):
                    taxonomy['fail_'+key] += int(not c[key])
            duplicate = int(valid.sum())-len({w for w in words if w is not None})
            # Unclassified valid paths are not duplicates; report them separately.
            duplicate -= sum(v and w is None for v, w in zip(valid, words))
            rec = dict(id=str(ident), family=recmap[str(ident)]['family'], variant=recmap[str(ident)]['variant'],
                       target=int(str(ident).rsplit('target',1)[1]), raw=raw, selected=sel,
                       raw_portal=rawp, selected_portal=selp, q_top1_valid=int(valid[top]),
                       duplicate_known_valid_candidates=duplicate,
                       unknown_valid_relations=int(sum(bool(v) and w is None for v, w in zip(valid, words))),
                       coverage_capacity_bound=min(4,raw['distinct']), words=words, valid=valid.tolist(),
                       q=q[i].tolist(), selected_indices=chosen, candidates=cand)
            rows.append(rec); grouped[rec['variant']].append(raw)
        assert mismatch == 0
        results[str(seed)] = dict(raw=average([r['raw'] for r in rows]), selected=average([r['selected'] for r in rows]),
            raw_portal=average([r['raw_portal'] for r in rows]), selected_portal=average([r['selected_portal'] for r in rows]),
            top1_valid=float(np.mean([r['q_top1_valid'] for r in rows])),
            duplicate_valid_fraction=sum(r['duplicate_known_valid_candidates'] for r in rows)/int(pool['labels'].sum()),
            unknown_valid_relations=sum(r['unknown_valid_relations'] for r in rows),
            vertices_midpoints_miss_collision=probe_missed, continuous_checker_disagreements=mismatch,
            by_variant={k:average(v) for k,v in grouped.items()}, requests=len(rows))
        errors[str(seed)] = dict(taxonomy); allrows[str(seed)] = rows
    data = dict(counts=counts, independent_families=len(families), complete_requests=len(request_keys),
                family_role_counts=dict(Counter(next(iter(x)) for x in families.values())),
                reference_count_histogram=dict(Counter(str(r['references']) for r in records)),
                valid_relation_count_histogram=dict(Counter(str(len(r['relation_counts'])) for r in records)),
                frequency_max_min_ratios=sorted({max(r['relation_counts'].values())/min(r['relation_counts'].values())
                                               for r in records if r['relation_counts']}),
                role_parent_counts={k:len(v) for k,v in role_parents.items()}, family_leakage=False,
                rare_mode_recall=None, rare_mode_reason='Paired teacher counts do not define natural demo frequencies',
                modes_exhaustive=False, locked_access=False)
    write(out/'metrics.json', dict(data=data, generator=results, taxonomy=errors,
        mode_scope='Known positive low/over relation witnesses; report historical portal words separately. Neither is proven homotopy.',
        source_inputs_sha256=inputs, observed_image_sha256=image_hashes))
    write(out/'candidate_rows.json', allrows)
    print(json.dumps(dict(data=data, generator=results, taxonomy=errors)), flush=True)


def smoke(output):
    from scripts.run_observed_probability import torch_setup
    from routeset.observed_probability import load_scored_planner
    from routeset.factored_q import load_factored_planner
    torch = torch_setup(); out = Path(output); out.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, folder, loader in (
        ('single', RUN/'reliability/R1_seed0/deployment_seed0', load_scored_planner),
        ('dual', ROOT/'runs/factored_q_v1/conditional_endpoint_g0_s0/deployment_v2', load_factored_planner)):
        inp = torch.load(folder/'example_observed_inputs.pt', map_location='cuda', weights_only=False)
        model = loader(folder/'planner.pt', 'cuda')
        with torch.inference_mode(): pred = model(**inp, return_k=4)
        with np.load(folder/'example.npz') as a:
            for key in ('paths', 'events', 'q', 'selected_indices'):
                np.testing.assert_array_equal(pred[key][0].cpu().numpy(), a[key])
            if name == 'dual':
                for key in ('q_task', 'q_feas'):np.testing.assert_array_equal(pred[key][0].cpu().numpy(), a[key])
        results[name] = dict(exact=True, paths=list(pred['paths'].shape), q_sum=float(pred['q'].sum()),
                             bundle_sha256=sha(folder/'planner.pt'))
        del model
    write(out/'metrics.json',results); print(json.dumps(results),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['audit', 'smoke']); p.add_argument('--output', required=True)
    a = p.parse_args(); (main if a.stage=='audit' else smoke)(a.output)
