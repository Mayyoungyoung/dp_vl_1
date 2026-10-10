"""Compose forecast execution words using only current inferred geometry."""
import numpy as np
from research_selective_repair_v1.execution_semantics import word,tip_clear
from research_selective_repair_v1.body_feedback_data import WORDS

def compose(prediction,paths,cfg):
    paths=np.asarray(paths);n=len(paths);tips=prediction['tip']
    assert tips.shape[:1]==(n,) and tips.shape[-3:]==(23,4,3)
    weights=np.exp(prediction['log_weights']);hazard=prediction['hazard']
    continuation=1/(1+np.exp(np.clip(hazard,-40,40)))
    survival=continuation.prod(-1)
    result=np.zeros((n,17),np.float64);queries=0
    for i in range(n):
        for k in range(tips.shape[1]):
            trajectory=np.concatenate([paths[i,:1],tips[i,k].reshape(-1,3)])
            signature=word(trajectory,cfg);queries+=1
            # Predicted sampled path clearance, never a body safety certificate.
            clear=tip_clear(trajectory,cfg);queries+=1
            reaches=np.linalg.norm(trajectory[-1]-paths[i,-1])<=.03
            outcome=WORDS.index(signature)+1 if signature in WORDS and clear and reaches else 0
            mass=weights[i,k]*survival[i,k]
            result[i,outcome]+=mass
            result[i,0]+=weights[i,k]*(1-survival[i,k])
    np.testing.assert_allclose(result.sum(-1),1,atol=1e-5)
    return result.astype(np.float32),dict(prefix_hypotheses_per_route=tips.shape[1],
        predicted_signature_and_tip_proxy_calls=queries,actual_controller_queries=0,
        scope='Public-FK forecast with current inferred boxes; not actual geometry/execution validation')
