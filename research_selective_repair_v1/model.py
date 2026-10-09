"""Per-route edit support and displacement; independent of companion routes."""
import torch
from torch import nn
from research_selective_repair_v1.core import center_model,masks

class RepairHead(nn.Module):
    def __init__(self,width,arm='selective'):
        super().__init__();self.arm=arm
        self.context=nn.Linear(width,64);self.mode=nn.Embedding(16,16)
        self.net=nn.Sequential(nn.Conv1d(29+64+16,128,1),nn.SiLU(),nn.Conv1d(128,128,3,padding=1),nn.SiLU(),nn.Conv1d(128,128,3,padding=1),nn.SiLU())
        self.output=nn.Conv1d(128,4,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        with torch.no_grad():self.output.bias[3]=-2

    def forward(self,context,modes,drafts,local,hard=False,threshold=.5,scale=1.):
        b,k,h,_=drafts.shape
        ctx=self.context(context)[:,None,None].expand(-1,k,h,-1)
        mod=self.mode(modes)[:,:,None].expand(-1,-1,h,-1)
        x=torch.cat([local,ctx,mod],-1).reshape(b*k,h,-1).transpose(1,2)
        value=self.output(self.net(x)).transpose(1,2).reshape(b,k,h,4)
        delta=.08*value[...,:3].tanh();prob=value[...,3].sigmoid()
        endpoints=torch.ones_like(prob);endpoints[:,:,0]=0;endpoints[:,:,-1]=0
        if self.arm=='selective':support=masks(prob,threshold,12) if hard else prob
        else:support=torch.ones_like(prob)
        support=support*endpoints
        path=drafts+scale*support[...,None]*delta
        return path,dict(prob=prob,logits=value[...,3],delta=delta,support=support)

def load_repair(path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    center,_=center_model();head=RepairHead(center.base.mode_embedding.embedding_dim,ck['settings']['arm'])
    head.load_state_dict(ck['repair']);head=head.to(device).eval()
    if ck.get('center') is not None:center.load_state_dict(ck['center'])
    return center,head,ck
