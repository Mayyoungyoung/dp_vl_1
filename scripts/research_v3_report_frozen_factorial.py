"""Local, read-only scientific figure/report from sealed factorial artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=root/'frozen_factorial_analysis_v1/RESULTS.json'
    result=json.loads(source.read_text())
    primary=json.loads((root/'frozen_augmentation_analysis_v1/RESULTS.json').read_text())
    names=['safety_mean','margin_mean','verified_edit_augmented','frozen_encoder_plain','frozen_encoder_augmented']
    labels=['Parent','Unfrozen plain','Unfrozen augmented','Frozen plain','Frozen augmented']
    text='# Frozen input encoders x verified-positive augmentation\n\n'
    text+='Ordinary seed0 factorial on288 DEV_MODEL requests/32families; no new algorithm or independent test claim. All models use the same complete paired-domain q seed0.\n\n'
    text+='| Model | Valid@8 | Any valid | Distinct@8 | Rare recall | Brier | Retained/2169 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for n,label in zip(names,labels):
        m=result['metrics'][n]
        text+='|%s|%.6f|%.6f|%.6f|%.6f|%.6f|%d|\n'%(label,m['raw']['valid_fraction'],m['raw']['any_valid'],m['raw']['distinct'],m['rare_recall8'],m['reliability']['brier'],result['retained'][n])
    r=primary['fixed_source_retention'];inter=result['augmentation_retention_interaction']
    text+='\nFrozen augmentation-minus-plain retention: %.6f,95%%CI%s; registered gate **%s**. Interaction (frozen augmentation effect minus unfrozen augmentation effect): %.6f,95%%CI%s. Ordinary plain-freezing versus parent gate **%s**.\n\n'%(r['delta'],r['CI95'],result['ordinary_frozen_augmentation_gate'],inter['delta'],inter['CI95'],result['ordinary_plain_freezing_gate'])
    text+='Frozen geometry/feature/state tensors have identical initial/final hashes. Grounding loss is computed but constant with respect to remaining trainable parameters. Bounded endpoint residuals still train; freezing alone does not guarantee correct endpoints. Same1200 updates and actual input/group RNG; augmented targets require more processing.\n\n'
    text+='The retention denominator uses fixed safety_mean source witnesses for all destination models. Confidence intervals resample families, not generator seeds. Earlier unfrozen arms are reused sealed runs. Shared loader materializes permitted DEV caches, but only TRAIN enters updates. No TEST_LOCKED, q retraining, robot execution claim or post-result threshold change.\n\n'
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),sharey=True)
    for ax,key,title in zip(axes,['valid','retention'],['Candidate validity change','Fixed-source retention change']):
        ax.axvline(0,color='#999999',lw=1)
        for i,n in enumerate(names[1:]):
            v=result['route_versus_parent'][n]['valid'] if key=='valid' else result['retention_versus_parent'][n]
            d=100*v['delta'];lo,hi=np.array(v['CI95'])*100
            ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color=('#b35b38' if i<2 else '#24788b'),capsize=4)
        ax.set_title(title);ax.set_xlabel('Percentage points vs unchanged parent')
        ax.set_yticks(range(4));ax.set_yticklabels(labels[1:]);ax.grid(axis='x',alpha=.18)
    axes[0].invert_yaxis();fig.suptitle('Ordinary encoder-freezing control | seed0 DEV_MODEL, family 95% intervals')
    fig.tight_layout();fig.savefig(out/'factorial.png',dpi=180);fig.savefig(out/'factorial.svg');plt.close(fig)
    text+='![Family intervals](factorial.png)\n\nFull metrics, direction counts, failure details and provenance remain in the closed artifact archive; compact results accompany this report.\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    fullhash=hashlib.sha256(source.read_bytes()).hexdigest()
    result.pop('input_sha256',None)
    for m in result['metrics'].values():m.pop('input_sha256',None)
    result.update(full_result_path=str(source.resolve()),full_result_sha256=fullhash)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'PRIMARY.json').write_text(json.dumps({k:primary[k] for k in ['fixed_source_retention','by_direction','ordinary_augmentation_gate','route_contrast']},indent=2)+'\n')
    provenance=dict(source=str(source.resolve()),source_sha256=fullhash,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    command=['python','-m','scripts.research_v3_report_frozen_factorial','--root',str(root),'--output',str(out)])
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
