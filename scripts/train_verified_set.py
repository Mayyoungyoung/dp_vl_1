"""Stage A/B: observation-only model, audited training-only target construction.

Fresh outputs and immutable releases only. All actual job wall time is charged
by launch_verified_set_v1.sh. Replay is paired per training draw, not an expanded
unbounded corpus, and uses only targets already queried by the project arm.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from scripts.run_observed_probability import ROOT, SOURCE, read, write, sha, lines, torch_setup
from scripts import research_v3_frequency as old
from scripts.research_v3_audit import mode, plain
from scripts.evaluate_paired_modes import references, check_candidates
from routeset.verified_route_set import certified_radii, verify_reach, build_targets, build_set_targets

RUN = ROOT/'runs/verified_set_v1'
POLICY = SOURCE/'configs/verified_set_v1.json'
INITIAL = old.RUN/'safety_mean/last.pt'
PREPARED = RUN/'prepared'
CAPACITY_ONLY = True


def open_support():
    with np.load(old.SUPPORT/'support.npz') as z:
        support = {k:z[k] for k in z.files}
    rows = {r['id']:r for r in lines(old.DATA/'export/supervision.jsonl')}
    return support, rows


def checker(ref):
    def check(paths, events):
        label, truth, cfg = ref['label'], ref['truth'], ref['config']
        s = label['semantic_targets']
        valid = verify_reach(paths, events, ref['current'], s['centers'], s['target_index'],
                             truth['obstacle_centers'], truth['obstacle_halfsizes'], cfg['post_base_z']+.02)
        words = [mode(p, cfg) if v else None for p, v in zip(paths, valid)]
        return valid, words
    return check


def prepare():
    """Fixed TRAIN eligibility, certified regions, independent label replay."""
    out = PREPARED; out.mkdir(parents=True, exist_ok=False)
    cfg = read(POLICY); support, labels = open_support()
    ids = []; radii = []; rows = []; counts = Counter(); source_hashes = {}
    for i in np.flatnonzero(support['splits'] == 'TRAIN'):
        ident = str(support['ids'][i]); label = labels[ident]
        assert label['split'] == 'TRAIN'
        n = int(support['mask'][i].sum()); tags = support['modes'][i, :n]
        counts['train_requests'] += 1
        if CAPACITY_ONLY and len(set(tags)) > 8:
            counts['over_budget_teacher_requests'] += 1
            continue
        ref = references(label); p, e = support['paths'][i, :n], support['events'][i, :n]
        check = checker(ref); valid, words = check(p, e)
        original = check_candidates(p, e, label, ref['current'], ref['truth'], ref['config'])[1]
        np.testing.assert_array_equal(valid, [c['TipValid'] for c in original])
        assert valid.all() and words == tags.tolist()
        # Deliberately corrupted goals, starts, floor and events: never training inputs.
        bad = p[:1].repeat(4, 0); be = e[:1].repeat(4, 0)
        bad[0, -1, 0] += 1; bad[1, 0, 1] += .1
        bad[2, 12, 2] = ref['config']['post_base_z']-.1; be[3, 10] = 1-be[3, 10]
        bv, _ = check(bad, be)
        orig_bad = check_candidates(bad, be, label, ref['current'], ref['truth'], ref['config'])[1]
        np.testing.assert_array_equal(bv, [c['TipValid'] for c in orig_bad]); assert not bv.any()
        rr = certified_radii(p, ref['truth']['obstacle_centers'], ref['truth']['obstacle_halfsizes'],
                             ref['config']['post_base_z']+.02, cap=cfg['radius_cap_m'])
        padded = np.zeros(support['paths'].shape[1:3], dtype=np.float32); padded[:n] = rr
        radii.append(padded); ids.append(i)
        rows.append(dict(id=ident, modes=len(set(tags)), witnesses=n, mean_radius_m=float(rr.mean())))
        counts['verified_witnesses'] += n; counts['corruption_probes'] += 4
        for k in ('route_config', 'verification_only', 'observation'):
            source_hashes[label[k]] = sha(label[k])
    assert ids, 'No capacity-feasible TRAIN requests; do not silently change protocol'
    np.savez_compressed(out/'regions.npz', support_indices=np.asarray(ids), radii=np.asarray(radii))
    write(out/'manifest.json', dict(rows=rows, counts=dict(counts), support_sha256=sha(old.SUPPORT/'support.npz'),
        regions_sha256=sha(out/'regions.npz'), source_hashes=source_hashes,
        policy_sha256=sha(POLICY), eligibility=('TRAIN and original known_mode_count<=8, before model inference'
            if CAPACITY_ONLY else 'All1152 original TRAIN requests; preserve open/closed/shifted exposure'),
        independent_checker_agreement=True, locked_access=False))
    print(json.dumps(dict(counts)), flush=True)


def load_training():
    from scripts.train_paired_modes import load_data
    manifest = read(PREPARED/'manifest.json')
    assert sha(old.SUPPORT/'support.npz') == manifest['support_sha256']
    assert sha(PREPARED/'regions.npz') == manifest['regions_sha256']
    support, labels = open_support()
    with np.load(PREPARED/'regions.npz') as z:
        selected, radii = z['support_indices'], z['radii']
    data, geo, configs, centers, halves, _, labelmap = load_data()
    index = {str(k):i for i,k in enumerate(data['scene_ids'])}
    mapped = np.asarray([index[str(support['ids'][i])] for i in selected])
    assert all(data['splits'][i] == 'TRAIN' for i in mapped)
    refs = [references(labels[str(support['ids'][i])]) for i in selected]
    checks = [checker(r) for r in refs]
    return support, selected, radii, data, geo, configs, centers, halves, mapped, refs, checks


def diagnose():
    """Fixed TRAIN predictions: shape sensitivity and target-builder audit."""
    torch = torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from scripts.train_observed_geometry import batch_inputs
    out = RUN/'diagnostic'; out.mkdir(parents=True, exist_ok=False)
    s, selected, radii, data, geo, cfgs, cs, hs, mapped, refs, checks = load_training()
    ckpt = torch.load(INITIAL, map_location='cpu', weights_only=False)
    model = ProbabilisticGeometryRouteHead(**ckpt['config']['head_options']).cuda()
    model.load_state_dict(ckpt['model']); model.eval(); model.requires_grad_(False)
    # First 64 fixed eligible TRAIN requests; no model-dependent selection.
    rows = []; predictions = []; events = []; target_rows = []; verify_count = 0
    for local in range(min(64, len(selected))):
        si = selected[local]; ident = str(s['ids'][si]); n = int(s['mask'][si].sum())
        rp, re, tags = s['paths'][si, :n], s['events'][si, :n], s['modes'][si, :n]
        with torch.inference_mode():
            p, e, _ = model(**batch_inputs(data, geo, mapped[local:local+1], 'cuda'))
        p, e = p[0].cpu().numpy(), e[0].cpu().numpy(); predictions.append(p); events.append(e)
        valid, words = checks[local](p, e)
        original = check_candidates(p, e, refs[local]['label'], refs[local]['current'], refs[local]['truth'], refs[local]['config'])[1]
        np.testing.assert_array_equal(valid, [x['TipValid'] for x in original]); verify_count += len(p)
        result = dict(id=ident, valid=int(valid.sum()), known_modes=len(set(tags)), arms={})
        for arm in ('ordinary', 'gate', 'project'):
            tp, te, audit = build_targets(p, e, rp, re, tags, radii[local, :n], valid, words,
                                          arm, np.random.default_rng(0), checks[local])
            movement = np.square(tp-p).mean((1, 2))
            result['arms'][arm] = dict(mean_correction_mse=float(movement.mean()),
                correct_route_correction_mse=float(movement[valid].mean()) if valid.any() else None,
                protected=int(audit['protected'].sum()), fallbacks=int(audit['fallback'].sum()),
                oversubscribed=audit['oversubscribed'])
            target_rows.append(tp)
            checked = check_candidates(tp, te, refs[local]['label'], refs[local]['current'], refs[local]['truth'], refs[local]['config'])[1]
            assert all(x['TipValid'] for x in checked); verify_count += len(tp)
        # Fixed-mode equal-exposure replacement, and an exact duplication sanity control.
        base = np.arange(0, n, 5); variant = base+1
        changes = []
        for picked in (base, variant, np.r_[base, base]):
            tp, _, _ = build_targets(p, e, rp[picked], re[picked], tags[picked], radii[local, picked],
                valid, words, 'ordinary', np.random.default_rng(0), checks[local])
            changes.append(tp)
        np.testing.assert_array_equal(changes[0], changes[2])
        result['equivalent_reference_target_change_mse'] = float(np.square(changes[0]-changes[1]).mean())
        rows.append(result)
    np.savez_compressed(out/'predictions_targets.npz', paths=np.asarray(predictions), events=np.asarray(events),
                        targets=np.asarray(target_rows), ids=np.asarray([r['id'] for r in rows]))
    summary = dict(requests=len(rows), independently_verified_slots=verify_count,
        valid_slots=sum(r['valid'] for r in rows), rows=rows, initial_sha256=sha(INITIAL),
        source_commit=__import__('os').environ.get('CODE_COMMIT'), regions_sha256=sha(PREPARED/'regions.npz'),
        payload_sha256=sha(out/'predictions_targets.npz'), checker_agreement=True,
        interpretation='Direct target pressure only; not evidence of optimizer-induced harm or generalization', locked_access=False)
    write(out/'summary.json', plain(summary)); print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)


def train(arm, seed, resume=False, stop_after=None):
    torch = torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.segment_clearance import segment_clearance_loss
    from routeset.paired_modes import workspace_floor_loss
    from scripts.train_observed_geometry import batch_inputs
    from routeset.train_v2 import atomic_checkpoint, rng_state, restore_rng
    from routeset.observed_training_audit import tensor_state_digest, new_stream_audit, append_indices
    from scripts.train_paired_modes import source_identity
    cfg = read(POLICY); steps = cfg['steps']; batch = cfg['batch_size']
    out = RUN/('%s_seed%d' % (arm, seed)); out.mkdir(parents=True, exist_ok=resume)
    if (out/'last.pt').exists(): raise FileExistsError('Completed arm cannot be resumed')
    s, selected, radii, data, geo, configs, cs, hs, mapped, refs, checks = load_training()
    saved = torch.load(INITIAL, map_location='cpu', weights_only=False)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.default_rng(seed); lrng = np.random.default_rng(seed+6100)
    model = ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda()
    model.load_state_dict(saved['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:1.)
    settings = dict(arm=arm, seed=seed, policy=cfg, initial_sha256=sha(INITIAL),
        regions_sha256=sha(PREPARED/'regions.npz'), source_sha256=source_identity(),
        data_fingerprint=geo['fingerprint'], selected_ids_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),
        population='known_le8' if CAPACITY_ONLY else 'all_train')
    replay = None; replay_files = {}; archive = None
    if arm == 'replay':
        parent = RUN/('project_seed%d' % seed)
        parent_summary = read(parent/'summary.json')
        assert parent_summary['steps'] == steps
        replay_files = parent_summary['replay_sha256']
        for name, value in replay_files.items(): assert sha(parent/name) == value
        replay = {k:np.load(parent/('replay_'+k+'.npy'), mmap_mode='r') for k in ('paths', 'events', 'words', 'ids')}
        settings['replay_source_sha256'] = replay_files
    if arm in ('project', 'set_point', 'set_project'):
        specs = dict(paths=((steps,batch,8,24,3),np.float32), events=((steps,batch,8,24),np.float32),
                     words=((steps,batch,8),'U40'), ids=((steps,batch),np.int64))
        archive = {k:np.lib.format.open_memmap(out/('replay_'+k+'.npy'), mode='r+' if resume else 'w+',
                                              dtype=dtype, shape=shape) for k,(shape,dtype) in specs.items()}
    initial = tensor_state_digest(model.state_dict())
    audit = new_stream_audit(model, rng.bit_generator.state, torch.get_rng_state())
    count = Counter(); history=[]; start=0
    if resume:
        state = torch.load(out/'recovery.pt', map_location='cpu', weights_only=False)
        assert state['settings'] == settings
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); restore_rng(state['rng'], rng)
        lrng.bit_generator.state = state['loss_rng']; audit=state['sampler']; count=Counter(state['counts'])
        history=state['history']; start=state['step']
    else: write(out/'config.json', settings)
    tc, th = torch.tensor(cs,device='cuda'), torch.tensor(hs,device='cuda')
    floors = torch.tensor([c['post_base_z']+.02 for c in configs], device='cuda')
    tic = time.monotonic(); end=min(steps, stop_after or steps)
    def snapshot(step):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng_state(rng), loss_rng=lrng.bit_generator.state, sampler=audit, counts=dict(count),
            settings=settings, step=step, history=history, config=dict(head_options=saved['config']['head_options']))
    for step in range(start+1, end+1):
        local=rng.integers(len(selected),size=batch); ids=mapped[local]; audit=append_indices(audit,ids)
        inp=batch_inputs(data,geo,ids,'cuda'); model.train(); xyz,event,details=model(**inp)
        p,e=xyz.detach().cpu().numpy(),event.detach().cpu().numpy()
        target=[]; targete=[]; groundp=[]; grounde=[]; taglist=[]
        for j, li in enumerate(local):
            si=selected[li]; n=int(s['mask'][si].sum())
            rp,re,tags=s['paths'][si,:n],s['events'][si,:n],s['modes'][si,:n]
            groundp.append(rp); grounde.append(re)
            if replay is not None:
                assert replay['ids'][step-1,j] == ids[j]
                # Same original positives + all eight already-queried targets for this draw.
                rp=np.concatenate((rp,replay['paths'][step-1,j]))
                re=np.concatenate((re,replay['events'][step-1,j]))
                tags=np.r_[tags,replay['words'][step-1,j]]
                if np.any(tags == ''): raise ValueError('Unclassified replay target cannot be invented as a mode')
            valid, words=checks[li](p[j],e[j]); count['verified_raw_slots']+=8
            builder=build_set_targets if arm in ('set_point','set_project') else build_targets
            tp,te,a=builder(p[j],e[j],rp,re,tags,radii[li,:n],valid,words,
                           'ordinary' if arm=='replay' else arm,lrng,checks[li])
            count['protected_slots']+=int(a['protected'].sum()) if arm in ('gate','project','set_point','set_project') else 0
            count['fallback_slots']+=int(a['fallback'].sum()); count['oversubscribed_draws']+=int(a['oversubscribed'])
            count['verified_target_slots']+=8*(1+int(a['fallback'].any()))
            count['reference_slots_processed']+=len(rp)
            target.append(tp); targete.append(te); taglist.append(a['target_words'])
        tx=torch.tensor(np.asarray(target),device='cuda'); te=torch.tensor(np.asarray(targete),device='cuda')
        regression=(torch.cat((xyz[:,:,1:]-tx[:,:,1:],.2*(event[:,:,1:]-te[:,:,1:])[...,None]),-1).square().mean())
        gp,ge,gm=old.pad_targets(groundp,grounde)
        ground=positive_endpoint_attention_loss(details['attention'],inp['world_xyz'],inp['valid_mask'],
            torch.tensor(gp[:,:,-1],device='cuda'),torch.tensor(gm,device='cuda'),.025)
        clear=segment_clearance_loss(xyz,tc[ids],th[ids])+workspace_floor_loss(xyz,floors[ids])
        loss=regression+.02*ground+160*clear
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step(); scheduler.step()
        if archive is not None:
            archive['paths'][step-1]=np.asarray(target); archive['events'][step-1]=np.asarray(targete)
            archive['words'][step-1]=np.asarray(taglist); archive['ids'][step-1]=ids
        if old.record_history(step, steps):
            row=dict(step=step,loss=float(loss),regression=float(regression),clearance=float(clear),
                     elapsed_seconds=time.monotonic()-tic,counts=dict(count))
            history.append(row); print(json.dumps(row),flush=True)
        if step%100==0 or step==end:
            if archive is not None:
                for a in archive.values(): a.flush()
            atomic_checkpoint(out/'recovery.pt',snapshot(step))
    if end==steps:
        atomic_checkpoint(out/'last.pt',snapshot(end))
        replay_hashes={}
        if archive is not None:
            for k,a in archive.items(): a.flush(); replay_hashes['replay_'+k+'.npy']=sha(out/('replay_'+k+'.npy'))
        write(out/'summary.json',dict(steps=end,counts=dict(count),sampler=audit,initial_sha256=initial,
            final_sha256=tensor_state_digest(model.state_dict()),last_sha256=sha(out/'last.pt'),
            elapsed_seconds=time.monotonic()-tic,replay_sha256=replay_hashes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),seed_scope='paired continuation randomness, shared historical seed0 initialization'))


def evaluate(arm, seed):
    # Preserve old evaluator, full DEV denominator and complete fixed q encoder.
    old.evaluate('%s_seed%d' % (arm, seed), fixed_q=True, paired_score=True, generator_root=RUN)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','diagnose','train','evaluate'])
    parser.add_argument('--arm',choices=['ordinary','gate','project','replay','set_point','set_project']); parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--stop-after',type=int)
    parser.add_argument('--population',choices=['known_le8','all'],default='known_le8')
    a=parser.parse_args()
    if a.population=='all':
        RUN=RUN/'all_population_v2'; PREPARED=RUN/'prepared'; CAPACITY_ONLY=False
    if a.stage=='prepare': prepare()
    elif a.stage=='diagnose': diagnose()
    elif a.stage=='train': train(a.arm,a.seed,a.resume,a.stop_after)
    else: evaluate(a.arm,a.seed)
