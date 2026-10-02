"""Copy small measured reports and index binary artifacts without committing them."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--reports', required=True)
    args = parser.parse_args()
    source, target = Path(args.runs), Path(args.reports)
    artifacts = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.suffix in ('.json', '.jsonl'):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        if path.suffix in ('.pt', '.npz'):
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b''):
                    digest.update(chunk)
            artifacts.append(dict(path=str(path).replace('\\', '/'), bytes=path.stat().st_size,
                                  sha256=digest.hexdigest()))
    target.mkdir(parents=True, exist_ok=True)
    (target/'artifact_index.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')
    print('Exported measured JSON and indexed %d binary artifacts.' % len(artifacts))


if __name__ == '__main__':
    main()
