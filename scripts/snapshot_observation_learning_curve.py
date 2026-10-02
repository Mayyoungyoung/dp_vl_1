"""Atomically snapshot complete new TRAIN parents plus explicitly reused DEV.

This tool never rewrites source datasets, reassigns a source split, or caches
future trajectory tokens. It creates new strict observation manifests pointing
to already completed immutable parent files, with exact source-row/file hashes.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile


INPUT_KEYS = {'id', 'parent_id', 'split', 'image', 'instruction'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def row_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_complete_rows(path):
    """Ignore only an unterminated concurrent writer suffix, never full lines."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n')+1]
    return [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]


def inventory(dataset, split):
    dataset = Path(dataset).resolve()
    grouped = {key:defaultdict(list) for key in ('observations', 'supervision', 'attempts')}
    for key in grouped:
        for row in read_complete_rows(dataset/(key+'.jsonl')):
            if 'parent_id' in row:
                grouped[key][row['parent_id']].append(row)
    complete, incomplete = [], []
    for parent in sorted(grouped['observations']):
        observations, supervision, attempts = [grouped[key][parent] for key in ('observations', 'supervision', 'attempts')]
        if any(set(row) != INPUT_KEYS for row in observations):
            raise ValueError('observation input whitelist violated: '+parent)
        if {row['split'] for row in observations} != {split}:
            continue
        expected = {parent+'_target'+str(index) for index in range(3)}
        identifiers = {row['id'] for row in observations}
        reason = None
        if len(observations) != 3 or identifiers != expected:
            reason = 'three observation instructions not complete'
        elif len(supervision) != 3 or {row['id'] for row in supervision} != expected:
            reason = 'three target supervision records not complete'
        elif any(row.get('split', split) != split for row in supervision):
            raise ValueError('source split mismatch: '+parent)
        elif len(attempts) != 9 or {(row.get('input_id'), row.get('attempt')) for row in attempts} != {(item, attempt) for item in expected for attempt in range(3)}:
            reason = 'all nine proposal outcomes not complete'
        elif any(row.get('attempted', True) is False for row in attempts):
            reason = 'parent setup prevented observation collection'
        if reason is None:
            labels = {row['id']:row for row in supervision}
            for identifier in expected:
                actual = {row['route_file'] for row in attempts if row['input_id'] == identifier and row.get('success')}
                if actual != set(labels[identifier].get('routes', [])):
                    raise ValueError('saved reference list differs from successful attempts: '+identifier)
            complete.append(dict(parent_id=parent, observations=observations, supervision=supervision, attempts=attempts))
        else:
            incomplete.append(dict(parent_id=parent, reason=reason))
    setup_failures = [row for row in read_complete_rows(dataset/'parents.jsonl') if not row.get('setup_success', True)]
    return complete, incomplete, setup_failures


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')


def build(new_train, old_dev, output, train_parents=32, dev_parents=8):
    new_train, old_dev, output = Path(new_train).resolve(), Path(old_dev).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('snapshot already exists; never replace an existing learning-curve dataset')
    available, unfinished, setup_failures = inventory(new_train, 'TRAIN')
    development, old_unfinished, _ = inventory(old_dev, 'DEV_MODEL')
    if len(available) < train_parents or len(development) != dev_parents or old_unfinished:
        raise ValueError('not ready: complete new TRAIN parents=%d (need %d); complete reused DEV parents=%d (need exactly %d)' %
                         (len(available), train_parents, len(development), dev_parents))
    selected = [(new_train, item) for item in available[:train_parents]]+[(old_dev, item) for item in development]
    ids = [item['parent_id'] for _,item in selected]
    if len(set(ids)) != len(ids):
        raise ValueError('parent overlap between new training and reused development')
    observations, supervision, attempts, provenance = [], [], [], []
    file_hashes = {}
    for source, item in selected:
        parent_hashes = {}
        # A closed parent contains future paths and failure evidence as well as
        # current observations. Hashes record them but do not add them to input.
        for path in sorted((source/item['parent_id']).rglob('*')):
            if path.is_file():
                parent_hashes[str(path)] = digest(path)
        for row in item['observations']:
            changed = dict(row, image=str((source/row['image']).resolve()))
            if not Path(changed['image']).is_file():
                raise ValueError('missing completed parent image')
            observations.append(changed)
        for row in item['supervision']:
            changed = dict(row, observation=str((source/row['observation']).resolve()),
                           routes=[str((source/name).resolve()) for name in row.get('routes', [])])
            if 'verification_only' in changed:
                changed['verification_only'] = str((source/changed['verification_only']).resolve())
            for path in [changed['observation']]+changed['routes']:
                if not Path(path).is_file():
                    raise ValueError('missing completed parent current/route file')
            supervision.append(changed)
        attempts.extend(dict(row, source_dataset=str(source)) for row in item['attempts'])
        provenance.append(dict(parent_id=item['parent_id'], source_dataset=str(source), source_split=item['observations'][0]['split'],
            source_observation_rows=item['observations'], source_supervision_rows=item['supervision'], source_attempt_rows=item['attempts'],
            source_rows_sha256=row_digest(item), source_files_sha256=parent_hashes))
        file_hashes.update(parent_hashes)
    # Fail closed if a supposed completed parent changed while the snapshot
    # was being assembled. No writer lock is taken on the live collector.
    for filename, expected in file_hashes.items():
        if digest(filename) != expected:
            raise RuntimeError('source parent changed during snapshot: '+filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name+'.staging')
    if staging.exists():
        raise ValueError('staging directory exists; inspect it instead of overwriting')
    staging.mkdir()
    write_jsonl(staging/'observations.jsonl', observations)
    write_jsonl(staging/'supervision.jsonl', supervision)
    write_jsonl(staging/'attempts.jsonl', attempts)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), evaluation_protocol='observation_eval_v2',
        purpose='exploratory development learning curve; not independent confirmation',
        source_new_dataset=str(new_train), source_reused_dev_dataset=str(old_dev),
        parent_selection='lexicographically first requested number of fully closed initialized TRAIN parents, independent of number of successful routes',
        parent_completion='three observation records, three supervision records, all nine attempted proposal outcomes; successful route lists agree exactly',
        train_parents=train_parents, dev_model_parents=dev_parents, observations=len(observations),
        train_supervised_examples=sum(row['split'] == 'TRAIN' and bool(row['routes']) for row in supervision),
        dev_semantic_examples=sum(row['split'] == 'DEV_MODEL' for row in supervision),
        dev_reference_examples=sum(row['split'] == 'DEV_MODEL' and bool(row['routes']) for row in supervision),
        reused_dev_warning='These original eight DEV parents have repeatedly informed model and method selection. No held-out or final-test claim.',
        original_256_split_unchanged=True, incomplete_new_parents_excluded=unfinished,
        new_parent_setup_failures_recorded=setup_failures, source_parents=provenance,
        input_contract=sorted(INPUT_KEYS), input_manifest_sha256=digest(staging/'observations.jsonl'),
        supervision_manifest_sha256=digest(staging/'supervision.jsonl'), script_sha256=digest(Path(__file__)),
        cache_policy='Regenerate true frozen Qwen3-VL cache for this exact five-key manifest using the same official revision/processor. No route, goal, mode, acceptance or future token enters the prompt.')
    (staging/'snapshot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    staging.rename(output)
    return {key:manifest[key] for key in ('created_at', 'train_parents', 'dev_model_parents', 'observations',
                                         'train_supervised_examples', 'dev_semantic_examples', 'dev_reference_examples', 'input_manifest_sha256')}


def self_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, split, parents in [('new', 'TRAIN', ['p0','p1']), ('old', 'DEV_MODEL', ['d0'])]:
            source = root/name; source.mkdir()
            observations, supervision, attempts = [], [], []
            for parent in parents:
                folder = source/parent; folder.mkdir()
                (folder/'front.png').write_bytes(b'unit-test-image')
                (folder/'observation.npz').write_bytes(b'unit-test-current-not-scientific-evidence')
                for target in range(3):
                    identifier = parent+'_target%d' % target
                    observations.append(dict(id=identifier, parent_id=parent, split=split, image=parent+'/front.png', instruction='fixture '+str(target)))
                    routes = []
                    for attempt in range(3):
                        success = target != 2
                        row = dict(parent_id=parent, input_id=identifier, attempt=attempt, success=success, attempted=True)
                        if success:
                            route = parent+'/target%d_route%d.npz' % (target, attempt)
                            (source/route).write_bytes(b'unit-test-future-path')
                            row['route_file'] = route; routes.append(route)
                        attempts.append(row)
                    supervision.append(dict(id=identifier, parent_id=parent, split=split, observation=parent+'/observation.npz', routes=routes))
            write_jsonl(source/'observations.jsonl', observations)
            write_jsonl(source/'supervision.jsonl', supervision)
            write_jsonl(source/'attempts.jsonl', attempts)
        # Remove the final committed target record: p1 is still incomplete.
        path = root/'new/supervision.jsonl'; original = path.read_text()
        path.write_text('\n'.join(original.splitlines()[:-1])+'\n')
        before = digest(path)
        result = build(root/'new', root/'old', root/'one', train_parents=1, dev_parents=1)
        assert result['observations'] == 6 and result['dev_semantic_examples'] == 3 and result['dev_reference_examples'] == 2
        assert result['train_supervised_examples'] == 2 and digest(path) == before
        assert all(set(row) == INPUT_KEYS for row in read_complete_rows(root/'one/observations.jsonl'))
        try:
            build(root/'new', root/'old', root/'two', train_parents=2, dev_parents=1)
        except ValueError as exc:
            assert 'not ready' in str(exc)
        else:
            raise AssertionError('incomplete parent must never be published')
        assert not (root/'two').exists()
        path.write_text(original)
        result = build(root/'new', root/'old', root/'two', train_parents=2, dev_parents=1)
        assert result['observations'] == 9
        with (root/'new/observations.jsonl').open('a') as stream:
            stream.write('{"incomplete":')
        assert len(read_complete_rows(root/'new/observations.jsonl')) == 6
    print(json.dumps(dict(status='snapshot_self_test_passed', checks=['incomplete parent rejected', 'zero-reference instructions retained',
        'exact input whitelist', 'source unchanged', 'atomic final output', 'concurrent partial JSON suffix ignored'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-train', type=Path)
    parser.add_argument('--old-dev', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-parents', type=int, choices=(32,64), default=32)
    parser.add_argument('--dev-parents', type=int, default=8)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.new_train or not args.old_dev or not args.output:
        parser.error('new-train, old-dev and output required')
    print(json.dumps(build(args.new_train, args.old_dev, args.output, args.train_parents, args.dev_parents)), flush=True)


if __name__ == '__main__':
    main()
