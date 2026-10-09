import numpy as np
import torch
from routeset.mode_geometry import ModeGeometryHead,VOCAB
from scripts.mode_geometry_experiment import ROOT,PREP,Q,DATA,old,read,write,sha,lines,torch_setup
from scripts.analyze_paired_selection import select
from scripts.research_v3_audit import mode
from scripts.train_verified_set import checker
from routeset.observed_probability import route_observation_features,load_scored_planner

RUN=ROOT/'runs/realized_coverage_v1'
BASE=ROOT/'runs/mode_geometry_v1/canonical_C_seed0/last.pt'

def load_generator(path=BASE):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=ModeGeometryHead(**ck['config']['head_options'],conditional=True).cuda()
    model.load_state_dict(ck['model']);model.freeze_encoders();model.eval()
    return model,ck

def variants(modes):
    return np.array([sum(modes[:j]==m) for j,m in enumerate(modes)],np.int64)

def replace_query(modes,variant,slot,new_mode,new_variant=0):
    m=modes.copy();v=variant.copy();m[slot]=new_mode;v[slot]=new_variant
    return m,v

def interventions(m,v,index):
    # Every word has two rotating slots, independently of model/known positives.
    result=[(m,v,-1,-1)]
    for word in range(16):
        for slot in ((index+word)%8,(index+word+4)%8):
            if m[slot]==word and v[slot]==0:continue
            mm,vv=replace_query(m,v,slot,word);result.append((mm,vv,slot,word))
    return result

def assess(paths,events,q,ref):
    valid,words=checker(ref)(paths.reshape(-1,24,3),events.reshape(-1,24))
    valid=valid.reshape(-1,8);words=np.array([VOCAB.index(w) if w in VOCAB else -1 for w in words]).reshape(-1,8)
    raw=np.array([VOCAB.index(w) if w in VOCAB else -1 for w in [mode(p,ref['config']) for p in paths.reshape(-1,24,3)]]).reshape(-1,8)
    chosen=np.array([select(p,s,4) for p,s in zip(paths,q)])
    utility=[];present=[]
    for i in range(len(paths)):
        whole=set(words[i,valid[i]])-{-1};small=set(words[i,chosen[i]][valid[i,chosen[i]]])-{-1}
        utility.append([len(whole),valid[i].sum(),len(small),valid[i,chosen[i]].sum()])
        present.append([j in whole for j in range(16)])
    return dict(valid=valid,words=words,raw_words=raw,selected=chosen,utility=np.array(utility,np.float32),present=np.array(present))

class SceneRunner:
    """Cache only observed encodings; q encoder is distinct and unchanged."""
    def __init__(self,model,scorer,inp):
        self.model=model;self.scorer=scorer;self.inp=inp
        with torch.inference_mode():
            self.context,self.anchor=model.encode(**inp)
            self.qgeo=scorer.generator.geometry(**inp,return_point_features=True)
            self.qcontext=scorer.generator.head.feature_encoder(inp['features'])+scorer.generator.head.state_encoder(inp['current'])+self.qgeo['context']
            _,_,d=model.decode(self.context,self.anchor,inp['current'])
            self.base=d['mode_ids'][0].cpu().numpy();self.base_variants=variants(self.base)
    def run(self,modes,var):
        paths=[];events=[];scores=[]
        with torch.inference_mode():
            for start in range(0,len(modes),8):
                ms=torch.as_tensor(modes[start:start+8],device='cuda');vs=torch.as_tensor(var[start:start+8],device='cuda');n=len(ms)
                expand=lambda x:x.expand(n,*x.shape[1:])
                p,e,_=self.model.decode(expand(self.context),expand(self.anchor),expand(self.inp['current']),ms,variant_ids=vs)
                nodes,ctx=route_observation_features(p,e,expand(self.inp['current']),expand(self.inp['world_xyz']),expand(self.inp['rgb']),expand(self.inp['valid_mask']),expand(self.qgeo['point_features']),expand(self.qcontext),expand(self.qgeo['anchor_xyz']))
                q=(self.scorer.scorer((nodes-self.scorer.nodes_mean)/self.scorer.nodes_std,(ctx-self.scorer.context_mean)/self.scorer.context_std)/self.scorer.temperature).sigmoid()
                paths.extend(p.cpu().numpy());events.extend(e.cpu().numpy());scores.extend(q.cpu().numpy())
        return np.array(paths),np.array(events),np.array(scores)
