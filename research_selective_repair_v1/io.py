"""Lightweight study IO shared with the existing Torch-free simulator runtime."""
import numpy as np
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha,lines

RUN=ROOT/'runs/selective_repair_v1'
DATA=ROOT/'data/paired_modes_v1_v2'

def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v
