"""Fresh export of old64 TRAIN/reused12 DEV plus exactly extension32 TRAIN.

No model/cache work. Mechanical checks complete before either corpus raw is read.
The historical 225 observation/label rows remain an exact byte prefix.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts import collect_two_row_extension as extension
from scripts import collect_two_row_formal as formal
from scripts import export_two_row_observations as old_exporter
from scripts.snapshot_multitask_observations import child_path, digest, write_json

PROTOCOL = 'observed_two_row_composite108_export_v1'
SELECTION_PROTOCOL = 'observed_two_row_composite108_selection_v1'
OLD_MANIFEST_SHA256 = '309966e192a3c582cde503151e607b5ede679eaec596bd8ab2bc1fd06ffdbab1'
OUTPUT_FILES = old_exporter.OUTPUT_FILES
JSONL_FILES = OUTPUT_FILES[:3]


def selection_guard(spec):
    expected = dict(protocol=SELECTION_PROTOCOL, old_indices=list(range(76)),
                    extension_indices=list(range(32)), requested_parents=108,
                    requested_train_parents=96, requested_reused_dev_parents=12,
                    requested_inputs=324, requested_routes=2916,
                    old_actual_inputs=225, old_actual_train_inputs=189,
                    old_reused_dev_inputs=36, failed_parent_replacements=0,
                    old_export_manifest_sha256=OLD_MANIFEST_SHA256,
                    require_all_selected_closed=True, extension_dev_raw_allowed=False,
                    cache_or_training_authorized=False)
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('Only prospectively fixed composite108 selection is permitted')
    return expected


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Missing or malformed mechanical SHA256')
    return value


def _closure_budget(row):
    names = ('completed_slots', 'attempted_lower', 'attempted_upper', 'unattempted_lower', 'unattempted_upper')
    if any(type(row.get(k)) is not int for k in names) or row.get('requested_routes') != 27:
        raise ValueError('Invalid closure budget types or requested slots')
    done, lo, hi, ulo, uhi = (row[k] for k in names)
    if not (0 <= done <= lo <= hi <= 27 and hi - lo <= 1 and ulo == 27-hi and uhi == 27-lo):
        raise ValueError('Invalid closure budget bounds')


def mechanical_gate(old_source, extension_source, spec):
    """Read registrations/closures/mechanical receipts only, never outcome pools.

    Registered exact and 1mm groups span ALL plans, including sealed roles. Actual
    closure evidence supplies only 1mm hashes; no actual-exact claim is made.
    """
    selection_guard(spec)
    roots = dict(old=Path(old_source).resolve(), extension=Path(extension_source).resolve())
    if roots['old'] == roots['extension']:
        raise ValueError('Two distinct registered corpora are required')
    sources, groups, actual_groups, blocked, missing = {}, defaultdict(list), defaultdict(list), set(), []
    for name, module, indices in (('old', formal, spec['old_indices']),
                                  ('extension', extension, spec['extension_indices'])):
        root = roots[name]
        registration, manifest = module.verify_corpus(root)
        gate = module.live_layout_gate(root)
        rows = extension.checked_closures(root, registration)
        by_index = {r['index']: r for r in rows}
        plans = registration['parent_plan']
        if len(plans) != (116 if name == 'old' else 288):
            raise ValueError('Unexpected corpus registration size')
        selected = []
        for i in indices:
            plan = plans[i]
            role = 'DEV_MODEL' if name == 'old' and i >= 64 else 'TRAIN'
            seed = (283200 if name == 'old' else 400000) + i
            if (plan['index'], plan['parent_id'], plan['role'], plan['config']['split']) != (
                    i, 'two_row_reach_%d' % seed, role, role):
                raise ValueError('Selected parent identity/role differs from fixed registration')
            selected.append(plan)
            if i not in by_index:
                missing.append('%s:%s' % (name, plan['parent_id']))
                continue
            row = by_index[i]
            _closure_budget(row)
            actual = row.get('actual_geometry_1mm_sha256')
            if row['initial_observation_saved'] and not actual:
                raise ValueError('Observed parent has no measured geometry identity')
            if actual and actual != plan['registered_geometry_1mm_sha256']:
                blocked.add((name, plan['parent_id']))
        for plan in plans:
            identity = (name, plan['parent_id'])
            if not plan.get('registration_eligible', False):
                blocked.add(identity)
            for kind in ('exact', '1mm'):
                h = _sha(plan['registered_geometry_%s_sha256' % kind])
                groups[(kind, h)].append(dict(corpus=name, parent_id=plan['parent_id'],
                                             role=plan['role'], index=plan['index']))
        for row in rows:
            h = row.get('actual_geometry_1mm_sha256')
            if h:
                actual_groups[_sha(h)].append(dict(corpus=name, parent_id=row['parent_id'],
                                                   role=row['role'], index=row['index']))
        blocked.update((name, p) for p in gate['blocked_parent_ids'])
        sources[name] = dict(root=root, registration=registration, manifest=manifest,
                             gate=gate, rows=rows, selected=selected,
                             closures={r['index']: r for r in rows})
    duplicates = []
    for (kind, h), members in groups.items():
        if len(members) > 1:
            duplicates.append(dict(kind='registered_'+kind, sha256=h, members=members))
            blocked.update((r['corpus'], r['parent_id']) for r in members)
    for h, members in actual_groups.items():
        if len(members) > 1:
            duplicates.append(dict(kind='actual_1mm', sha256=h, members=members))
            blocked.update((r['corpus'], r['parent_id']) for r in members)
    selected_ids = {(name, p['parent_id']) for name, source in sources.items() for p in source['selected']}
    result = dict(ready=not missing and not (selected_ids & blocked), missing=missing,
                  selected_blocked=[dict(corpus=a, parent_id=b) for a,b in sorted(selected_ids & blocked)],
                  blocked_members=[dict(corpus=a, parent_id=b) for a,b in sorted(blocked)],
                  duplicate_groups=duplicates, mechanical_only=True,
                  exact_hash_scope='registered plans only; measured closure hash is 1mm only',
                  extension_dev_raw_opened=False, old_reserved_raw_opened=False)
    return sources, result


def _ready(old_source, extension_source, spec):
    sources, gate = mechanical_gate(old_source, extension_source, spec)
    if gate['missing']:
        raise ValueError('All 108 registered parents must be fully closed: '+','.join(gate['missing']))
    if not gate['ready']:
        raise ValueError('Composite selected mechanical layout gate blocked: '+str(gate['selected_blocked']))
    return sources, gate


def _capture_metadata(sources):
    hashes, metadata = {}, {}
    for name, source in sources.items():
        root = source['root']
        paths = [root/'registration.json', root/'corpus_manifest.json']
        paths += [root/'closures'/('%03d.json' % r['index']) for r in source['rows']]
        hashes.update({str(p): digest(p) for p in paths})
        metadata[name] = dict(source_dataset=str(root), corpus_protocol=source['manifest']['protocol'],
            registration_sha256=digest(root/'registration.json'),
            corpus_manifest_sha256=digest(root/'corpus_manifest.json'),
            collector_source_sha256=source['manifest']['source_sha256'],
            selected_indices=[p['index'] for p in source['selected']],
            selected_parent_ids=[p['parent_id'] for p in source['selected']],
            mechanical_closure_files_sha256={str(p): hashes[str(p)] for p in paths[2:]},
            selected_raw_roots=[str(root/'parents'/p['role']/p['parent_id']) for p in source['selected']])
    return hashes, metadata


def _jsonl(blob):
    if blob and not blob.endswith(b'\n'):
        raise ValueError('Historical JSONL lacks terminal newline; cannot append without changing bytes')
    return [json.loads(line) for line in blob.decode('utf-8').splitlines() if line.strip()]


def _rows_guard(inputs, labels, plans):
    allowed = {p['parent_id']: p['role'] for p in plans}
    ids = []
    if len(inputs) != len(labels):
        raise ValueError('Input and supervision counts differ')
    for x, y in zip(inputs, labels):
        parent = x['parent_id']
        if (parent not in allowed or x['split'] != allowed[parent] or
                set(x) != old_exporter.INPUT_KEYS or
                x['id'] not in {parent+'_target%d' % t for t in range(3)} or
                any(x[k] != y[k] for k in ('id', 'parent_id', 'split'))):
            raise ValueError('Unexpected input/label identity or role')
        ids.append(x['id'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate input identity')


def _old_bytes(old_export, spec, source):
    old_export = Path(old_export).resolve()
    if digest(old_export/'export_manifest.json') != spec['old_export_manifest_sha256']:
        raise ValueError('Historical prefix76 export manifest changed')
    manifest, _ = old_exporter.verify_export(old_export)
    if (old_exporter.selection_guard(manifest['selection']) != 64 or
            Path(manifest['source_dataset']).resolve() != source['root'] or
            (manifest['actual_inputs'], manifest['requested_parents']) != (225, 76)):
        raise ValueError('Historical export does not identify the original 225 inputs')
    blobs = {name: child_path(old_export,name).read_bytes() for name in JSONL_FILES}
    rows = {name: _jsonl(blob) for name, blob in blobs.items()}
    x, y = rows['observations.jsonl'], rows['supervision.jsonl']
    _rows_guard(x, y, source['selected'])
    if len(x) != 225 or sum(r['split']=='TRAIN' for r in x) != 189 or sum(r['split']=='DEV_MODEL' for r in x) != 36:
        raise ValueError('Historical role counts changed')
    parents = json.loads(child_path(old_export,'parent_inventory.json').read_text())
    return manifest, blobs, rows, parents


def _sources_unchanged(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Closed source changed: '+str(path))


def _raw_root_guard(root, plan):
    # The old reader resolves children safely, but resolving an already-symlinked
    # role root could otherwise change which corpus/role constitutes its boundary.
    paths = [root/'parents', root/'parents'/plan['role'], root/'parents'/plan['role']/plan['parent_id']]
    if any(p.is_symlink() for p in paths):
        raise ValueError('Selected raw parent/role directory must not be a symlink')
    if paths[-1].resolve() != root.resolve()/'parents'/plan['role']/plan['parent_id']:
        raise ValueError('Selected raw root escapes its declared corpus/role')


def _budget_summary(parents):
    keys = ('attempted_lower','attempted_upper','completed_slots','unattempted_lower','unattempted_upper')
    return {key: sum(p['closure'][key] for p in parents) for key in keys}


def build(old_source, extension_source, old_export, spec, output):
    output = Path(output).resolve(); staging = output.with_name(output.name+'.staging')
    if output.exists() or staging.exists():
        raise FileExistsError('Fresh atomic composite export required')
    sources, gate = _ready(old_source, extension_source, spec)
    # The historical export verifier hashes its raw source paths. Check every
    # selected role/parent boundary before invoking that verifier, not just before
    # decoding through read_parent below.
    for source in sources.values():
        for plan in source['selected']:
            _raw_root_guard(source['root'], plan)
    hashes, metadata = _capture_metadata(sources)
    old_export = Path(old_export).resolve()
    old_manifest, blobs, rows, old_parents = _old_bytes(old_export, spec, sources['old'])
    hashes.update(old_manifest['source_files_sha256'])
    hashes.update({str(old_export/name): digest(old_export/name) for name in OUTPUT_FILES+('export_manifest.json',)})
    all_rows, parents = {}, []
    for name, source in sources.items():
        inputs, labels, attempts, inventory = [], [], [], []
        for plan in source['selected']:
            _raw_root_guard(source['root'], plan)
            x,y,z,p = old_exporter.read_parent(source['root'], plan, source['closures'][plan['index']], hashes)
            inputs.extend(x); labels.extend(y); attempts.extend(z); inventory.append(p)
        _rows_guard(inputs, labels, source['selected'])
        if name == 'old' and (inputs != rows['observations.jsonl'] or labels != rows['supervision.jsonl'] or
                              attempts != rows['attempts.jsonl'] or inventory != old_parents):
            raise ValueError('Revalidated old input/label/attempt/inventory differs from original export')
        all_rows[name] = dict(zip(JSONL_FILES, (inputs,labels,attempts)))
        parents.extend(inventory)
    combined_x = all_rows['old']['observations.jsonl'] + all_rows['extension']['observations.jsonl']
    combined_y = all_rows['old']['supervision.jsonl'] + all_rows['extension']['supervision.jsonl']
    _rows_guard(combined_x, combined_y, sources['old']['selected']+sources['extension']['selected'])
    if any(hashes.get(path) != value for path,value in old_manifest['source_files_sha256'].items()):
        raise ValueError('Historical raw source hash was changed during revalidation')
    _sources_unchanged(hashes)
    # Recheck live gates after raw validation; future closure additions are allowed,
    # but selected source bytes and geometry must still be valid.
    _, final_gate = _ready(old_source, extension_source, spec)
    output.parent.mkdir(parents=True, exist_ok=True); staging.mkdir()
    preservation = {}
    for name in JSONL_FILES:
        added = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'
                        for r in all_rows['extension'][name]).encode('utf-8')
        (staging/name).write_bytes(blobs[name]+added)
        preservation[name] = dict(old_bytes=len(blobs[name]), old_sha256=hashlib.sha256(blobs[name]).hexdigest(),
                                  old_rows=len(rows[name]), appended_rows=len(all_rows['extension'][name]))
    write_json(staging/'parent_inventory.json', parents)
    manifest = dict(protocol=PROTOCOL, created_at=datetime.now(timezone.utc).isoformat(), selection=spec,
        sources=metadata, historical_export=str(old_export), historical_export_manifest_sha256=spec['old_export_manifest_sha256'],
        requested_parents=108, requested_train_parents=96, requested_reused_dev_parents=12,
        requested_inputs=324, requested_routes=2916, actual_inputs=len(combined_x),
        actual_inputs_by_role={role: sum(r['split']==role for r in combined_x) for role in ('TRAIN','DEV_MODEL')},
        actual_observed_train_parents=len({r['parent_id'] for r in combined_x if r['split']=='TRAIN'}),
        positive_references=sum(p['positive_references'] for p in parents),
        unobserved_requested_inputs=sum(p['unobserved_requested_inputs'] for p in parents),
        actual_attempt_records=sum(len(r['attempts.jsonl']) for r in all_rows.values()),
        route_slot_accounting=_budget_summary(parents),
        original_bytes_preserved=preservation, reused_dev_input_ids=[r['id'] for r in combined_x if r['split']=='DEV_MODEL'],
        extension_added_input_ids=[r['id'] for r in all_rows['extension']['observations.jsonl']],
        source_files_sha256=hashes, output_files_sha256={name: digest(staging/name) for name in OUTPUT_FILES},
        mechanical_gate_at_export=gate, mechanical_gate_after_raw_validation=final_gate,
        exporter_sha256=digest(__file__), reused_reader_sha256=digest(old_exporter.__file__),
        live_gate_required_before_model_use=True, training_authorized=False, cache_generated=False,
        extension_dev_raw_opened=False, old_reserved_raw_opened=False,
        reference_policy='All validated known positives including unknown types; missing references are not negatives',
        cache_reuse_policy='Not performed; future code must prove per-ID arrays/source identity and account only new encoding')
    write_json(staging/'export_manifest.json', manifest)
    _sources_unchanged(hashes)
    staging.rename(output)
    return manifest


def verify_export(output):
    """Revalidate bytes and current dual mechanical gates without decoding raw."""
    output = Path(output).resolve()
    m = json.loads((output/'export_manifest.json').read_text())
    if m.get('protocol') != PROTOCOL:
        raise ValueError('Explicit composite export protocol required')
    selection_guard(m['selection'])
    if (m['requested_parents'],m['requested_train_parents'],m['requested_reused_dev_parents'],
            m['requested_inputs'],m['requested_routes']) != (108,96,12,324,2916):
        raise ValueError('Composite requested denominators changed')
    if (m.get('training_authorized') is not False or m.get('cache_generated') is not False or
            m.get('extension_dev_raw_opened') is not False or m.get('old_reserved_raw_opened') is not False or
            m.get('live_gate_required_before_model_use') is not True):
        raise ValueError('Composite export-only/sealed-role policy changed')
    if digest(__file__) != m['exporter_sha256'] or digest(old_exporter.__file__) != m['reused_reader_sha256']:
        raise ValueError('Composite exporter/reader source changed')
    sources, gate = _ready(m['sources']['old']['source_dataset'],m['sources']['extension']['source_dataset'],m['selection'])
    for name, source in sources.items():
        saved=m['sources'][name]
        if (saved['registration_sha256'] != digest(source['root']/'registration.json') or
                saved['corpus_manifest_sha256'] != digest(source['root']/'corpus_manifest.json') or
                saved['collector_source_sha256'] != source['manifest']['source_sha256'] or
                saved['selected_indices'] != [p['index'] for p in source['selected']] or
                saved['selected_parent_ids'] != [p['parent_id'] for p in source['selected']] or
                saved['selected_raw_roots'] != [str(source['root']/'parents'/p['role']/p['parent_id']) for p in source['selected']]):
            raise ValueError('Composite dual-source identity changed')
        for plan in source['selected']:
            _raw_root_guard(source['root'],plan)
    if set(m['output_files_sha256']) != set(OUTPUT_FILES):
        raise ValueError('Incomplete composite output hashes')
    _sources_unchanged(m['source_files_sha256'])
    for name, value in m['output_files_sha256'].items():
        if digest(child_path(output,name)) != value:
            raise ValueError('Composite output changed')
    _, blobs, rows, old_parents = _old_bytes(m['historical_export'],m['selection'],sources['old'])
    if m['historical_export_manifest_sha256'] != m['selection']['old_export_manifest_sha256']:
        raise ValueError('Historical export SHA receipt changed')
    for name in JSONL_FILES:
        combined=child_path(output,name).read_bytes()
        if not combined.startswith(blobs[name]):
            raise ValueError('Historical byte prefix changed')
        receipt=dict(old_bytes=len(blobs[name]), old_sha256=hashlib.sha256(blobs[name]).hexdigest(),
                     old_rows=len(rows[name]), appended_rows=len(_jsonl(combined[len(blobs[name]):])))
        if receipt != m['original_bytes_preserved'][name]:
            raise ValueError('Historical byte prefix receipt changed')
    x = _jsonl(child_path(output,'observations.jsonl').read_bytes())
    y = _jsonl(child_path(output,'supervision.jsonl').read_bytes())
    _rows_guard(x,y,sources['old']['selected']+sources['extension']['selected'])
    if len(x) != m['actual_inputs'] or 324-len(x) != m['unobserved_requested_inputs']:
        raise ValueError('Composite actual/missing input denominators changed')
    if sum(r['split']=='DEV_MODEL' for r in x) != 36:
        raise ValueError('Reused DEV identity count changed')
    parents=json.loads(child_path(output,'parent_inventory.json').read_text())
    if len(parents)!=108 or parents[:76]!=old_parents:
        raise ValueError('Historical/composite parent inventory changed')
    for inventory,(name,plan) in zip(parents,[(n,p) for n,s in sources.items() for p in s['selected']]):
        if (inventory['parent_id'],inventory['index'],inventory['split'],inventory['closure']) != (
                plan['parent_id'],plan['index'],plan['role'],sources[name]['closures'][plan['index']]):
            raise ValueError('Composite parent inventory identity changed')
    derived=dict(actual_inputs_by_role={r:sum(v['split']==r for v in x) for r in ('TRAIN','DEV_MODEL')},
        actual_observed_train_parents=len({v['parent_id'] for v in x if v['split']=='TRAIN'}),
        positive_references=sum(len(v['routes']) for v in y),
        actual_attempt_records=len(_jsonl(child_path(output,'attempts.jsonl').read_bytes())),
        route_slot_accounting=_budget_summary(parents),
        reused_dev_input_ids=[v['id'] for v in x if v['split']=='DEV_MODEL'],
        extension_added_input_ids=[v['id'] for v in x[225:]])
    if any(m.get(k)!=value for k,value in derived.items()):
        raise ValueError('Composite actual counts/identity receipt changed')
    return m, gate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-source',type=Path,required=True);p.add_argument('--extension-source',type=Path,required=True)
    p.add_argument('--old-export',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--readiness-only',action='store_true')
    a=p.parse_args(argv);spec=json.loads(a.selection.read_text())
    if a.readiness_only:
        _, result=mechanical_gate(a.old_source,a.extension_source,spec)
        print(json.dumps(result,ensure_ascii=False));return 0 if result['ready'] else 2
    if a.output is None:p.error('--output required for export')
    result=build(a.old_source,a.extension_source,a.old_export,spec,a.output)
    print(json.dumps({k:result[k] for k in ('requested_parents','actual_inputs','unobserved_requested_inputs','positive_references')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
