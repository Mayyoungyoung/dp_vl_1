"""Per-route edit support and displacement; independent of companion routes."""
import torch
from torch import nn
from research_selective_repair_v1.core import center_model,masks

class RepairHead(nn.Module):
    def __init__(self,width,arm='selective',goal_tail=False):
        super().__init__();self.arm=arm;self.goal_tail=goal_tail
        self.context=nn.Linear(width,64);self.mode=nn.Embedding(16,16)
        self.net=nn.Sequential(nn.Conv1d((33 if goal_tail else 29)+64+16,128,1),nn.SiLU(),nn.Conv1d(128,128,3,padding=1),nn.SiLU(),nn.Conv1d(128,128,3,padding=1),nn.SiLU())
        self.output=nn.Conv1d(128,4,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        with torch.no_grad():self.output.bias[3]=-2

    def forward(self,context,modes,drafts,local,hard=False,threshold=.5,scale=1.,decoupled_gate=False):
        b,k,h,_=drafts.shape
        ctx=self.context(context)[:,None,None].expand(-1,k,h,-1)
        mod=self.mode(modes)[:,:,None].expand(-1,-1,h,-1)
        x=torch.cat([local,ctx,mod],-1).reshape(b*k,h,-1).transpose(1,2)
        value=self.output(self.net(x)).transpose(1,2).reshape(b,k,h,4)
        delta=(.4 if self.goal_tail else .08)*value[...,:3].tanh();prob=value[...,3].sigmoid()
        endpoints=torch.ones_like(prob);endpoints[:,:,0]=0
        if not self.goal_tail:endpoints[:,:,-1]=0
        if self.arm=='selective' and self.goal_tail:
            # Release one goal tail using evidence from the current observation.
            gate=(prob[:,:,-1]>=threshold).float() if hard else torch.ones_like(prob[:,:,-1]) if decoupled_gate else prob[:,:,-1]
            s=torch.linspace(0,1,7,device=prob.device)[1:];s=s*s*(3-2*s)
            support=torch.zeros_like(prob);support[:,:,18:]=gate[:,:,None]*s
        elif self.arm=='selective':support=masks(prob,threshold,12) if hard else prob
        else:support=torch.ones_like(prob)
        support=support*endpoints
        path=drafts+scale*support[...,None]*delta
        return path,dict(prob=prob,logits=value[...,3],delta=delta,support=support)

def load_repair(path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    center,_=center_model();head=RepairHead(center.base.mode_embedding.embedding_dim,ck['settings']['arm'],ck['settings'].get('goal_tail',False))
    head.load_state_dict(ck['repair']);head=head.to(device).eval()
    if ck.get('center') is not None:center.load_state_dict(ck['center'])
    return center,head,ck
