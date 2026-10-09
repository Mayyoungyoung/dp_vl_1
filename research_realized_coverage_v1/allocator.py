"""Observation/query-only allocation. Never reads routes or verification inputs."""
import torch
from torch import nn

class SuccessHead(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,192),nn.SiLU(),nn.Linear(192,16))
    def forward(self,context):return self.net(context)

class SetUtility(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.context=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64),nn.SiLU())
        self.word=nn.Embedding(16,32);self.variant=nn.Embedding(8,8)
        self.token=nn.Sequential(nn.Linear(104,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(144,64),nn.SiLU(),nn.Linear(64,4))
        self.register_buffer('scales',torch.tensor([8.,8.,4.,4.]))
    def forward(self,context,modes,variants):
        h=self.context(context)
        t=self.token(torch.cat((h[:,None].expand(-1,8,-1),self.word(modes),self.variant(variants)),-1))
        hist=torch.nn.functional.one_hot(modes,16).float().mean(1)
        return self.value(torch.cat((h,t.mean(1),t.amax(1),hist),-1)).sigmoid()*self.scales

def occurrence(modes):
    return torch.stack([(modes[:,:i]==modes[:,i:i+1]).sum(-1) for i in range(8)],1)

class RealizationUtility(nn.Module):
    """Dense actual-mode outcomes; product union is an approximation, no guarantee."""
    def __init__(self,peers=True,width=128):
        super().__init__();self.peers=peers
        self.context=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,64),nn.SiLU())
        self.word=nn.Embedding(16,32);self.variant=nn.Embedding(8,8)
        self.token=nn.Sequential(nn.Linear(104,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.interaction=nn.MultiheadAttention(32,4,batch_first=True) if peers else nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,32))
        self.outcome=nn.Linear(32,17)
        self.returned=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
    def components(self,context,modes,variants):
        h=self.context(context);t=self.token(torch.cat((h[:,None].expand(-1,8,-1),self.word(modes),self.variant(variants)),-1))
        t=t+(self.interaction(t,t,t,need_weights=False)[0] if self.peers else self.interaction(t))
        logits=self.outcome(t);prob=logits.softmax(-1)[...,:16]
        u=(1-(1-prob).prod(1)).sum(-1);v=prob.sum((1,2))
        ret=self.returned(torch.cat((h,t.mean(1),t.amax(1)),-1)).sigmoid()*4
        return torch.cat((u[:,None],v[:,None],ret),-1),logits
    def forward(self,context,modes,variants):return self.components(context,modes,variants)[0]

def make_allocator(kind):
    if kind=='success':return SuccessHead()
    if kind=='net':return SetUtility()
    if kind in ('dense','no_peer'):return RealizationUtility(peers=kind=='dense')
    raise ValueError(kind)

def allocate(kind,head,context,base,variant,max_swaps=2):
    """Two token-only sweeps then caller decodes once. No route generation here."""
    if kind=='success':
        m=torch.argsort(head(context),dim=-1,descending=True,stable=True)[:,:8]
        return m,occurrence(m),dict(replacements=0,query_sets_evaluated=0)
    m=base.clone();v=variant.clone();changes=0;evaluated=0
    # Batch-independent deterministic refinement; full one-swap neighborhood.
    for b in range(len(context)):
        visited={tuple(m[b].tolist()+v[b].tolist())}
        for _ in range(max_swaps):
            mm=m[b:b+1].expand(128,-1).clone();vv=v[b:b+1].expand(128,-1).clone()
            slots=torch.arange(8,device=m.device).repeat_interleave(16);words=torch.arange(16,device=m.device).repeat(8)
            mm[torch.arange(128,device=m.device),slots]=words;vv[torch.arange(128,device=m.device),slots]=0
            score=head(context[b:b+1].expand(128,-1),mm,vv)-head(context[b:b+1],m[b:b+1],v[b:b+1])
            feasible=(score[:,0]>.1)&(score[:,1]>=-.02)&(score[:,2]>=-.02)&(score[:,3]>=-.02)
            for j in range(128):
                if tuple(mm[j].tolist()+vv[j].tolist()) in visited:feasible[j]=False
            evaluated+=129
            if not feasible.any():break
            j=score[:,0].masked_fill(~feasible,-float('inf')).argmax()
            m[b]=mm[j];v[b]=vv[j];visited.add(tuple(m[b].tolist()+v[b].tolist()));changes+=1
    return m,v,dict(replacements=changes,query_sets_evaluated=evaluated)
