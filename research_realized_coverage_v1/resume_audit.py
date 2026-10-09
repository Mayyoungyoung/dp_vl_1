import argparse
from scripts import diagnose_mode_geometry as old
from research_realized_coverage_v1.core import RUN

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs=2,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    old.RUN=RUN;old.resume_audit(*a.names,a.output)
