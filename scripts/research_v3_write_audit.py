"""Generate the phase-one reports only from successfully sealed audit JSON."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def main(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    d=json.loads((source/'metrics.json').read_text());g=d['generator'];t=d['taxonomy'];data=d['data']
    mean=lambda key,field:float(np.mean([v[key][field] for v in g.values()]))
    total=lambda field:sum(v.get(field,0) for v in t.values())
    shutil.copyfile(source/'metrics.json',out/'metrics.json')
    shutil.copyfile(source/'data_requests.json',out/'data_requests.json')
    shutil.copyfile(source/'candidate_rows.json',out/'candidate_rows.json')
    table='| Seed | OracleValid@8 | ValidDistinct@8 | KnownModeRecall@8 | ValidDistinct@4 | KnownModeRecall@4 | Top1 valid |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for seed,v in g.items():
        table+='| %s | %.3f%% | %.4f | %.3f%% | %.4f | %.3f%% | %.3f%% |\n'%(seed,100*v['raw']['any_valid'],v['raw']['distinct'],
            100*v['raw']['recall'],v['selected']['distinct'],100*v['selected']['recall'],100*v['top1_valid'])
    (out/'DATA_AUDIT.md').write_text('''# Complete-input multimodality audit

Actual audit source b6727ff; audit_v4 completed exit0 on 2026-10-09 (Asia/Shanghai).
Full identity hashes RGB pixels, depth, camera matrices, current gripper state,
instruction and world-metric convention. No grouping by task category alone.

There are **1440 unique complete requests**, 1152 TRAIN and 288 DEV_MODEL,
from **160 independent layout families** (128/32). Every request has 6, 7 or 9
verified references. Counts: 960 requests have9, 240 have6, 240 have7. All11760
stored H24 references pass the current full-line checker. These are geometric
teachers over actual rendered RGB-D, not execution demonstrations. The historical
22-reference training exclusion came from other merged corpora, not these paired
references. Historical raw data remain untouched.

Each witnessed operational passage has exactly one base teacher reference.
Max/min teacher-count ratio is1. Thus the present corpus does not test frequency
imbalance. Natural RareModeRecall is undefined until an explicit controlled
frequency regime is imposed; a singleton teacher is not a rare-mode estimate.

Modes: first forward crossing per registered obstacle row. Above the 2cm-inflated
top is `over`; otherwise the lateral free gap is `gap0/1/2`. This separates central
low and high passages and removes arbitrary unknown height strips. It is an
operational passage vector, not a proof of different homotopy classes. Historical
portal words are retained separately: their lateral-first precedence merges some
central-low/high references. No old metric or artifact is rewritten.

Zero paired-family role overlap and zero parent-ID overlap across SCORE_TRAIN64,
DEV_SCORE32, CALIBRATION32, FUTURE_GENERATOR_TRAIN96 and the paired corpus. This
rechecks these explicitly registered exports; it does not claim an exhaustive
image-duplicate audit of every historical corpus. Old development parents stay
reused development evidence. TEST_LOCKED payloads and results were not opened.

The witness set is incomplete. A closed expanded low gap has a local geometric
absence certificate for that gap; it does not eliminate above or exterior paths.
For open/shifted inputs, nine known witnesses exceed M8, so full recall is
impossible at that budget. Use capacity bounds in coverage interpretation.

Reproduce with `python -m scripts.research_v3_audit audit --output NEW_DIRECTORY`
from an immutable source export on the authorized server. Actual commands and
source hashes are in runs/research_v3_local/jobs. Request details, reference mode
counts and certificates are in data_requests.json; metrics and input hashes in
metrics.json. Inputs are oracle audit labels; model forward sees observations only.
''',encoding='utf-8')
    text='''# Raw generator upper bound and selection audit

The same three historical R1 models generate M8/H24 without oracle repair or
filtering. All6912 candidate validity labels were independently recomputed with
the continuous checker; disagreements0. Returned K4 uses each generator's
existing independent single-q scorer and calibration, unchanged. This is a
selection analysis, not a new generator training result.

'''+table+'''\nThree-seed means: OracleValid@8 **%.3f%%**, ValidDistinct@8 **%.4f**,
known witness recall@8 **%.3f%%**; K4 distinct **%.4f**, witness recall **%.3f%%**.
'''%(100*mean('raw','any_valid'),mean('raw','distinct'),100*mean('raw','recall'),mean('selected','distinct'),100*mean('selected','recall'))
    text+='''
Candidate validity %.3f%%; selected candidate validity %.3f%%. All original
portal counts are present in metrics.json and must not be confused with the
new above-first passage definition. Neither recall denominator is exhaustive.
RareModeRecall is undefined for this uniformly counted base teacher corpus.

Main bottleneck: missed witnessed passages and continuous collisions, rather
than Top1 scoring alone. Most K4 coverage reduction is its four-slot capacity;
extra loss below min(4, raw distinct) totals%d mode slots across864 requests.
Next experiment falsifies frequency-retention motivation against ordinary
balancing and distinct matching before inventing a correspondence mechanism.
'''%(100*mean('raw','valid_fraction'),100*mean('selected','valid_fraction'),total('k4_extra_relation_loss_beyond_capacity'))
    (out/'GENERATOR_ORACLE.md').write_text(text,encoding='utf-8')
    text='''# Failure decomposition

Counts cover288 requests x3 trained models, not864 independent scenes. All
targets and edits belong to32 parent families. Failure categories overlap.

| Category | Actual count | Meaning |
|---|---:|---|
| Generator empty | %d/864 requests | No valid candidate before scoring |
| Partial known-mode coverage | %d/864 requests | Valid pool misses at least one witness; includes M8 capacity limit |
| Scorer miss with valid pool | %d/864 requests | Highest q invalid despite an available valid candidate |
| K4 loss beyond capacity | %d mode slots | min(4, raw distinct) minus returned valid distinct |
| Invalid candidates | %d/6912 | Full task and geometry event false |
| Post collision failures | %d/6912 | Continuous inflated-box intersection |
| Semantic goal failures | %d/6912 | Wrong intended endpoint |
| Floor/event failures | 0/6912 | Under current reach contract |

'''%(total('zero_valid_requests'),total('partial_known_relation_coverage_requests'),total('score_top1_miss_with_valid_pool'),
       total('k4_extra_relation_loss_beyond_capacity'),total('invalid_candidates'),total('fail_post_segments_clear'),total('fail_semantic_goal_correct'))
    missed=sum(v['vertices_midpoints_miss_collision'] for v in g.values())
    text+='''47 probe samples (24 vertices +23 midpoints) miss%d continuous post collisions,
%.3f%% of candidates and %.3f%% of all post collision failures. Existing checker
labels are correct; the issue is incomplete observation-feature sampling, not
182 mislabeled successes. Nearest visible surface distance is also not occupancy.

The independent checker intersects each complete closed segment with physical
AABBs inflated by2cm on every axis (a conservative L-infinity tip envelope).
World XYZ is meter-scale, RGB-D is calibrated camera-to-world. Floor is0.775m;
linear segments cannot dip below both endpoint heights. Goal tolerance3cm,
start tolerance5mm, reach events preserved. This covers registered posts and
the workspace lower bound, not full-arm collision, execution error, table mesh,
arbitrary occluded objects or unknown space. Those are outside label Y.

Method implications: ordinary mode-balanced coverage and injective matching
must be tested first. Low-cost exact single-q/dual model replay passes for
all paths, events, scores and chosen indices; historical dual-factor superiority
remains rejected. Failed audit attempts and their receipts remain available.
'''%(missed,100*missed/6912,100*missed/total('fail_post_segments_clear'))
    (out/'FAILURE_TAXONOMY.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.source,a.output)
