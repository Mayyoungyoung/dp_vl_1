"""Reuse exactly the immutable parent opportunity definitions."""
import argparse
from research_realized_coverage_v1.core import RUN
from scripts import analyze_mode_geometry as inherited

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--figures',action='store_true');a=p.parse_args()
    inherited.RUN=RUN
    inherited.main(a.names,a.output,a.figures)
