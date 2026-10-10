"""Freeze observation-only predictions before independent geometry evaluation.

This study reuses frozen models; no fitting or checkpoint selection occurs.
Every arm generates exactly eight routes, and the same complete scorer returns
four. Stochastic repetitions remain separate trials, never merged candidate pools.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT, SOURCE, read, write, sha, lines, torch_setup

RUN = ROOT / 'runs/route_portfolio_v1'
POLICY = SOURCE / 'configs/route_portfolio_v1.json'


def evaluate(name, checkpoint, dataset, sampling='adaptive', inference_seed=71239, head=None):
    torch = torch_setup()
    from routeset.mode_geometry import load_fixed_scored_mode_planner, VOCAB
    from research_realized_coverage_v1.allocator import SuccessHead
    from scripts.evaluate_paired_modes import inputs_for, references, check_candidates
    from scripts.research_v3_audit import mode, coverage, average, plain
    from scripts.mode_geometry_experiment import Q, old
    from scripts.analyze_paired_selection import select
    policy = read(POLICY)
    spec = policy['datasets'][dataset]
    data = ROOT / spec['relative_path']
    assert spec['role'] == 'DEV_MODEL'
    assert sha(data/'export/observations.jsonl') == spec['observations_sha256']
    assert sha(data/'export/supervision.jsonl') == spec['supervision_sha256']
    rows = lines(data/'export/observations.jsonl')
    assert len(rows) == spec['requests'] and all(r['split'] == 'DEV_MODEL' for r in rows)
    labels = {r['id']: r for r in lines(data/'export/supervision.jsonl')}
    assert all(labels[r['id']]['split'] == 'DEV_MODEL' for r in rows)
    assert len(set(r['id'] for r in rows)) == len(rows)
    out = RUN/name
    out.mkdir(parents=True, exist_ok=False)
    planner = load_fixed_scored_mode_planner(checkpoint, Q, 'cuda').requires_grad_(False)
    assert sha(Q) == policy['scorer_sha256']
    proposal = None
    if head is not None:
        saved = torch.load(head, map_location='cpu', weights_only=False)
        assert saved['settings']['kind'] == 'success'
        assert saved['settings']['generator_sha256'] == sha(checkpoint)
        proposal = SuccessHead().cuda().eval()
        proposal.load_state_dict(saved['model'], strict=True)
        proposal.requires_grad_(False)
    torch.manual_seed(inference_seed)
    hashes = {}
    pools = {k: [] for k in ('paths','events','q','ids','mode_ids','mode_logits','selected_indices')}
    times = []
    calls = []
    allocations = []
    hook = planner.generator.head.output.register_forward_hook(
        lambda module, args, result: calls.append(tuple(result.shape)))
    allocation_hook = planner.generator.mode_predictor.register_forward_hook(
        lambda module, args, result: allocations.append(result.detach()))
    for row in rows:
        inp = inputs_for(row, labels[row['id']], data/'export/qwen_cache', torch, hashes)
        with torch.inference_mode():
            torch.cuda.synchronize()
            tic = time.monotonic()
            if proposal is None:
                result = planner(**inp, sampling=sampling)
                p, e, q, mi = (result[k] for k in ('paths','events','q','mode_ids'))
                logits = allocations[-1]
            else:
                context, anchor = planner.generator.encode(**inp)
                logits = proposal(context)
                mi = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :8]
                p, e, _ = planner.generator.decode(context, anchor, inp['current'], mi)
                q = old.fixed_path_scores(planner.fixed_q, p, e, inp)
            torch.cuda.synchronize()
            times.append(time.monotonic()-tic)
        pp, ee, qq, mm = [v[0].cpu().numpy() for v in (p, e, q, mi)]
        for k, val in dict(paths=pp, events=ee, q=qq, ids=row['id'], mode_ids=mm,
                           mode_logits=logits[0].cpu().numpy(), selected_indices=select(pp,qq,4)).items():
            pools[k].append(val)
    hook.remove()
    allocation_hook.remove()
    assert len(calls) == len(rows) and all(s[:2] == (1,8) for s in calls)
    np.savez_compressed(out/'pool.npz', **pools)
    seal = dict(protocol=policy['protocol'], source_commit=__import__('os').environ.get('CODE_COMMIT'),
        source_sha256={str(f.relative_to(SOURCE)):sha(f) for f in (SOURCE/'research_route_portfolio_v1').glob('*.py')},
        policy_sha256=sha(POLICY),
        checkpoint=str(checkpoint), generator_sha256=sha(checkpoint), scorer_sha256=sha(Q),
        head_sha256=sha(head) if head else None, dataset=dataset,
        sampling=sampling, inference_seed=inference_seed, predictions_sha256=sha(out/'pool.npz'),
        input_hashes=hashes, data_spec=spec, locked_access=False,
        cost=dict(decoded_sets=len(calls), generated_routes=8*len(calls), returned_routes=4*len(calls),
                  observed_decode_calls=True, observed_encoder_decode_q_ms_mean=1000*float(np.mean(times)),
                  timing_scope='Cached Qwen features; RGB-D encoder, decoder and complete q. No uncached VLM timing.'))
    write(out/'PREDICTION_SEAL.json', seal)
    # Oracle configuration, geometry and reference routes are first used here.
    result_rows = []
    for i, row in enumerate(rows):
        ref = references(labels[row['id']])
        p, e, q = pools['paths'][i], pools['events'][i], pools['q'][i]
        _, cc = check_candidates(p,e,ref['label'],ref['current'],ref['truth'],ref['config'])
        valid = np.array([c['TipValid'] for c in cc])
        words = [mode(path, ref['config']) if ok else None for path, ok in zip(p,valid)]
        known = {mode(path,ref['config']) for path,ok in zip(ref['paths'],ref['reference_valid']) if ok}
        known.discard(None)
        assigned = [VOCAB[j] for j in pools['mode_ids'][i]]
        chosen = pools['selected_indices'][i]
        result_rows.append(dict(id=row['id'], family=row['parent_id'].rsplit('_',1)[0],
            variant=row['parent_id'].rsplit('_',1)[1], words=words,valid=valid.tolist(),
            assigned_modes=assigned,selected_indices=chosen.tolist(),
            raw=coverage(words,valid,known,range(8)),selected=coverage(words,valid,known,chosen),
            duplicate_valid_routes=int(valid.sum()-len(set(words)-{None})),
            distinct_queries=len(set(assigned)), condition_hit=float(np.mean([v and w==m for v,w,m in zip(valid,words,assigned)]))))
    summary = dict(raw=average([r['raw'] for r in result_rows]),
        selected=average([r['selected'] for r in result_rows]),
        duplicate_valid_routes=float(np.mean([r['duplicate_valid_routes'] for r in result_rows])),
        distinct_queries=float(np.mean([r['distinct_queries'] for r in result_rows])),
        condition_hit=float(np.mean([r['condition_hit'] for r in result_rows])),
        source_commit=seal['source_commit'], predictions_sha256=seal['predictions_sha256'],
        generator_sha256=seal['generator_sha256'], head_sha256=seal['head_sha256'],
        dataset=dataset, sampling=sampling, inference_seed=inference_seed,cost=seal['cost'],
        variants={v:dict(raw=average([r['raw'] for r in result_rows if r['variant']==v]),
                        selected=average([r['selected'] for r in result_rows if r['variant']==v]))
                  for v in sorted(set(r['variant'] for r in result_rows))},
        scope=spec['scope'], locked_access=False, robot_execution=False)
    write(out/'rows.json', plain(result_rows))
    write(out/'RESULTS.json', plain(summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--dataset',choices=['shift'],default='shift')
    p.add_argument('--sampling',choices=['adaptive','balanced','ordinary'],default='adaptive')
    p.add_argument('--inference-seed',type=int,default=71239)
    p.add_argument('--head',type=Path)
    evaluate(**vars(p.parse_args()))
