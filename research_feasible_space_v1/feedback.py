"""Reuse unchanged16word/two-slot design, fresh outputs bound to new decoder."""
import argparse
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.model import load_model
from research_realized_coverage_v1 import feedback,train_allocation

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=type(RUN));p.add_argument('--pool');a=p.parse_args()
    if a.checkpoint:
        feedback.RUN=RUN;feedback.load_generator=load_model;feedback.collect(a.name,'TRAIN',checkpoint=a.checkpoint)
    else:
        train_allocation.RUN=RUN;train_allocation.train('success',a.name,a.pool,0,2400)
