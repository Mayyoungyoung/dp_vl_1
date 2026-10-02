"""Reconcile actual job snapshots without deleting historical registry evidence."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = json.loads((root/'JOBS.json').read_text(encoding='utf-8-sig'))
    path = root/'experiments/registry.jsonl'
    previous = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    # Recorded path uniquely identifies same-named seeds across experimental families.
    records = {record.get('record_path', record.get('log', record['run_id'])): record for record in previous}
    for record in snapshot['records']:
        record = dict(record, snapshot_utc=snapshot['snapshot_utc'])
        records[record['record_path']] = record
        # Old entries without a status path refer to the same immutable log.
        for key in list(records):
            if key != record['record_path'] and records[key].get('log') and records[key].get('log') == record.get('log'):
                records.pop(key)
    path.write_text(''.join(json.dumps(record, ensure_ascii=False)+'\n' for record in records.values()), encoding='utf-8')
    print('Reconciled %d recorded jobs; retained historical unmatched records.' % len(records))


if __name__ == '__main__':
    main()
