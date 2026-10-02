"""Exact teacher-forced causal loss without retaining full-sequence vocab logits."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def causal_chunked_loss(hidden,labels,lm_head,chunk_size=64):
    if hidden.ndim!=3 or labels.shape!=hidden.shape[:2] or chunk_size<1:
        raise ValueError('Expected matching BxL hidden states/labels and positive chunk size')
    if any(parameter.requires_grad for parameter in lm_head.parameters()):
        raise ValueError('This exact memory-efficient baseline assumes a frozen vocabulary head')
    # Token at position t predicts label t+1. Prompt and padding are all -100.
    targets=labels[:,1:].reshape(-1)
    valid=targets!=-100
    if not bool(valid.any()):raise ValueError('No supervised answer tokens')
    states=hidden[:,:-1].reshape(-1,hidden.shape[-1])[valid]
    targets=targets[valid]
    def block_loss(states,targets):
        return F.cross_entropy(lm_head(states).float(),targets,reduction='sum')
    total=hidden.new_zeros((),dtype=torch.float32)
    for begin in range(0,len(targets),chunk_size):
        states_block=states[begin:begin+chunk_size];target_block=targets[begin:begin+chunk_size]
        if torch.is_grad_enabled() and states_block.requires_grad:
            loss=checkpoint(block_loss,states_block,target_block,use_reentrant=False)
        else:loss=block_loss(states_block,target_block)
        total=total+loss
    return total/len(targets)
