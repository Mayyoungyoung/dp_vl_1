"""Static scientific figure from the completed reserved96 DEV audit."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'reports/observed_obstacle_reserved96_analysis_v1/analysis.json'
    latency=root/'reports/observed_obstacle_reserved96_full_inference_v1/cached_vs_online_audit.json'
    result=json.loads(source.read_text());timing=json.loads(latency.read_text())
    planner=result['traditional_planner']
    print('Planner metrics keys:',list(planner))
    metrics=[result['results'][key]['metrics'] for key in ('soft_best','peak_best','peak_last')]
    planner_report=json.loads((root/'reports/observation_astar_obstacle_reserved96_v2/dev_model/tip_evaluation/metrics.json').read_text())
    metrics.append(planner_report)
    names=['Soft\nbest / last3000','Peak\nbest1250','Peak\nlast3000','Observed A*\nclosed instructions']
    keys=['UniqueClassifiedTipValidAtK','DuplicateClassifiedTipValidCount','UnknownTypeTipValidCount']
    labels=['Distinct classified valid','Repeated classified valid','Valid, type unknown','Invalid / failed slot']
    colors=['#237a64','#8fc8b5','#e8bd59','#d4d8de']
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.2),gridspec_kw={'width_ratios':[1.35,1]})
    values=np.array([[m[k] for k in keys]+[4*(1-m['TipValidAtK'])] for m in metrics])
    np.testing.assert_allclose(values.sum(1),4,atol=1e-7)
    bottom=np.zeros(4)
    for column,(label,color) in enumerate(zip(labels,colors)):
        axes[0].bar(np.arange(4),values[:,column],bottom=bottom,label=label,color=color,width=.64)
        bottom+=values[:,column]
    axes[0].set_xticks(np.arange(4),names);axes[0].set_ylabel('Candidates per instruction (K = 4)')
    axes[0].set_ylim(0,4.65);axes[0].set_title('A. Full candidate accounting')
    for index,m in enumerate(metrics):
        axes[0].text(index,4.12,f"{100*m['TipValidAtK']:.1f}% valid",ha='center',fontsize=9)
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False,fontsize=8.5)
    points=[(timing['arms']['soft']['all_requests_ms_median'],metrics[0]['UniqueClassifiedTipValidAtK'],'Soft best', '#677998'),
            (timing['arms']['peak']['all_requests_ms_median'],metrics[1]['UniqueClassifiedTipValidAtK'],'Peak best','#237a64'),
            (planner['observation_to_routes_ms_median'],metrics[3]['UniqueClassifiedTipValidAtK'],'Observed A*','#b88038')]
    for x,y,name,color in points:
        axes[1].scatter(x,y,c=color,s=65)
        axes[1].annotate(f'{name}\n{x:.0f} ms',xy=(x,y),xytext=(7,-5 if name=='Soft best' else 8),textcoords='offset points',fontsize=9)
    axes[1].set_xscale('log');axes[1].set_xlim(40,7000);axes[1].set_ylim(0,1.6)
    axes[1].set_xlabel('Measured observation-to-path milliseconds (log)')
    axes[1].set_ylabel('Distinct classified tip-valid routes / instruction')
    axes[1].set_title('B. Quality and measured request cost')
    axes[1].text(.02,.04,'Qwen models: GPU1 + CPU1\nA*: CPU1, includes observed proxy checks\nNo equal-time or equal-hardware claim',transform=axes[1].transAxes,fontsize=8.5)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Fresh obstacle DEV: 8 parents, 24 instructions, 1 training seed',fontsize=13)
    fig.text(.5,.015,'All zero-reference inputs and failed slots retained. Box-tip check only; no full-arm, scoring, or execution certificate.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.065,right=.97,top=.86,bottom=.27,wspace=.32)
    out=root/'reports/observed_obstacle_reserved96_analysis_v1/figures';out.mkdir(parents=True,exist_ok=True)
    outputs=[out/'quality_cost.png',out/'quality_cost.pdf']
    for path in outputs:fig.savefig(path,dpi=180)
    plt.close(fig)
    index={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,latency,*outputs,Path(__file__)]}
    (out/'provenance.json').write_text(json.dumps(index,indent=2)+'\n')


if __name__=='__main__':main()
