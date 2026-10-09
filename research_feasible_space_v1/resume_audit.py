import argparse
from research_feasible_space_v1.prepare import RUN
from scripts import diagnose_mode_geometry as inherited
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs=2,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    inherited.RUN=RUN;inherited.resume_audit(*a.names,a.output)
