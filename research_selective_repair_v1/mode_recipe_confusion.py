"""Strong empirical conditional native-word control from shared TRAIN labels.

Nominal word comes from current inferred posts/path. Recipe is an internal
proposal index, not an old scene or oracle edit ID. Diagnostic held families
are excluded; fixed half-count smoothing is specified before any results.
"""
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.execution_semantics import word
from research_selective_repair_v1.body_feedback_data import WORDS

def nominal(paths,completed,base):
    labels=[]
    for p,c in zip(paths,completed):
        h=np.clip(c[:,2]-base,.025,.4)
        cfg=dict(row_x=[float(c[j:j+2,0].mean()) for j in (0,2)],post_y=[c[j:j+2,1].tolist() for j in (0,2)],
            post_heights=[float(h[j:j+2].max()) for j in (0,2)],post_base_z=base,tip_clearance_m=.02)
        w=word(p,cfg);labels.append(0 if w not in WORDS else WORDS.index(w)+1)
    return np.asarray(labels)

def fit(dataset,name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    manifest=read(dataset.parent/'MANIFEST.json');assert manifest['no_DEV_feedback'] and manifest['samples_sha256']==sha(dataset)
    with np.load(dataset) as z:d={k:z[k] for k in ('paths','completed','labels','families','roles','options')}
    assert set(d['roles'])=={'TRAIN'};families=sorted(set(d['families']));use=~np.isin(d['families'],families[-2:])
    base=read(RUN/'constraints_seed0_v1/config.json')['post_base'];nom=nominal(d['paths'],d['completed'],base)
    table=np.full((int(d['options'].max())+1,17,16),.5,dtype=np.float64);counts=np.zeros(table.shape[:2],int)
    for i in np.flatnonzero(use&(d['labels']>0)):
        table[d['options'][i],nom[i],d['labels'][i]-1]+=1;counts[d['options'][i],nom[i]]+=1
    table/=table.sum(-1,keepdims=True)
    write(out/'MODEL.json',dict(conditional_mass=table.tolist(),counts=counts.tolist(),post_base=base,dataset_sha256=sha(dataset),
        fit_families=list(map(str,families[:-2])),diagnostic_TRAIN_families=list(map(str,families[-2:])),
        smoothing=.5,condition='Current inferred nominal word and internal proposal recipe',native_queries=0,locked_access=False))
    return table,nom

def compose(success,planned,recipes,table):
    planned=np.asarray(planned);recipes=np.broadcast_to(np.asarray(recipes),planned.shape)
    positive=np.asarray(success)[...,None]*np.asarray(table)[recipes,planned]
    return np.concatenate([1-positive.sum(-1,keepdims=True),positive],-1).astype(np.float32)
