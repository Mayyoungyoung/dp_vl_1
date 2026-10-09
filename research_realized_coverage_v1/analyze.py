"""Reuse exactly the immutable parent opportunity definitions."""
import argparse
from research_realized_coverage_v1.core import RUN,write
from scripts import analyze_mode_geometry as inherited

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--figures',action='store_true');p.add_argument('--labels',nargs='+');a=p.parse_args()
    if a.labels:
        assert len(a.labels)==len(a.names) and len(set(a.labels))==len(a.labels)
        original_plot=inherited.plot
        def plot(out,names,results,records,pools,models,refs):
            rename=dict(zip(names,a.labels));rename['parent']='parent'
            remap=lambda d:{rename[k]:v for k,v in d.items()}
            original_plot(out,[rename[n] for n in names],remap(results),remap(records),remap(pools),remap(models),refs)
            write(out/'figure_model_labels.json',{v:k for k,v in rename.items()})
        inherited.plot=plot
    inherited.RUN=RUN
    inherited.main(a.names,a.output,a.figures)
