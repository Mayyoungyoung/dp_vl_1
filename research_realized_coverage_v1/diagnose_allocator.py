"""Prediction errors on already-generated neighborhoods, no new oracle calls."""
import argparse
import json
import numpy as np
import torch
from research_realized_coverage_v1.core import *
from research_realized_coverage_v1.allocator import SetUtility

def main(head,folders,output):
    torch_setup();saved=torch.load(RUN/head,map_location='cpu',weights_only=False)
    model=SetUtility().cuda();model.load_state_dict(saved['model']);model.eval();result={}
    for folder in folders:
        summary=read(RUN/folder/'SUMMARY.json');pred=[];actual=[];chosen=[];best=[]
        for row in summary['files']:
            with np.load(RUN/folder/(row['id']+'.npz')) as z:
                with torch.no_grad():
                    c=torch.as_tensor(z['context'][None],device='cuda').expand(len(z['modes']),-1)
                    p=model(c,torch.as_tensor(z['modes'],device='cuda'),torch.as_tensor(z['variants'],device='cuda')).cpu().numpy();delta=p-p[0]
                truth=z['gain'];allowed=(delta[:,0]>.1)&(delta[:,1:]>=-.02).all(-1)
                ix=int(np.argmax(np.where(allowed,delta[:,0],-1e9))) if allowed.any() else 0
                pred.extend(delta[1:]);actual.extend(truth[1:]);chosen.append(truth[ix]);best.append(truth[:,0].max())
        p=np.array(pred);y=np.array(actual);positive=p[:,0]>.1;safe=positive&(p[:,1:]>=-.02).all(-1)
        result[folder]=dict(records=len(y),gain_mse=float(np.mean((p[:,0]-y[:,0])**2)),gain_correlation=float(np.corrcoef(p[:,0],y[:,0])[0,1]),
            predicted_safe_positive=int(safe.sum()),true_net_gain_precision=float(np.mean(y[safe,0]>0)) if safe.any() else None,
            positive_recall=float(np.mean(safe[y[:,0]>0])),within_sampled_neighborhood_mean_gain=np.mean(chosen,0).tolist(),oracle_local_gain=float(np.mean(best)))
    write(RUN/output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--head',default='net_fit0/last.pt');p.add_argument('--folders',nargs='+',default=['feedback_C_train','diagnostic_C_dev']);p.add_argument('--output',default='allocation_diagnostic.json');a=p.parse_args();main(**vars(a))
