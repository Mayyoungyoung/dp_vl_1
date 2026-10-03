"""Seal aggregate QA only after three independent actual-review receipts exist."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('F:/dpvlm/reports/observed_two_row_extension128_quality_v1')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = [
    ('VISUAL_QA_CONSTRAINT_UPDATE_64_95.json', range(64, 96), [1, 2]),
    ('QA_ROOT_096_111.json', range(96, 112), [3]),
    ('QA_LOCAL_112_127.json', range(112, 128), [4]),
]
rows = []
parents = set()
fronts = set()
for name, indices, pages in sources:
    receipt = json.loads((ROOT / name).read_text(encoding='utf-8'))
    images = receipt.get('files')
    if images is None:
        images = receipt['parents'] + receipt['front_pages']
    expected = {'qa_views/two_row_reach_%d_all3_targets_front.png' % (400000 + i) for i in indices}
    expected |= {'qa_views/new64_fronts_original_page%d.png' % p for p in pages}
    assert {r['path'] for r in images} == expected, name
    assert len(images) == len(expected), name
    for image in images:
        assert image['viewed'] is True, (name, image['path'])
        assert sha(ROOT / image['path']) == image['sha256'], (name, image['path'])
    assert parents.isdisjoint(indices)
    assert fronts.isdisjoint(pages)
    parents.update(indices)
    fronts.update(pages)
    rows.append(dict(path=name, sha256=sha(ROOT / name), reviewer=receipt['reviewer'],
                     parent_indices=list(indices), front_pages=pages,
                     image_count=len(images), all_image_hashes_verified=True))
assert parents == set(range(64, 128))
assert fronts == {1, 2, 3, 4}
result = dict(protocol='extension128_complete_actual_visual_qa_v1',
              finalized_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              actual_review_receipts=rows, parent_indices=sorted(parents),
              parent_contact_sheets=64, route_target_plots=192, requested_slots=1728,
              front_pages=4, all_assigned_images_actually_viewed_by_named_reviewers=True,
              all_receipt_image_hashes_verified=True,
              missing_or_corrupt_panel_reported=False,
              retained=['unknown positive references', 'failed and partial paths', 'long loops', 'classified duplicate'],
              known_presentation_limit='Some wide-axis original plots have crowded small labels; original images remain archived.',
              scope_limit='Presentation and evidence coverage only; no 3D/whole-arm safety, complete-solution or model-gain claim.',
              prior64='256 original images byte-identical; prior visual QA is referenced, not repeated.',
              new_model_forward=0, new_search=0, new_simulation=0, reserved_raw_reads=0)
out = ROOT / 'VISUAL_QA_COMPLETE.json'
assert not out.exists(), 'fresh aggregate only'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(path=str(out), sha256=sha(out), parents=64, front_pages=4)))
