"""Compose forecast execution words using only current inferred geometry."""
import numpy as np
from research_selective_repair_v1.execution_semantics import word,tip_clear
from research_selective_repair_v1.body_feedback_data import WORDS

def computation_counts(settings,routes):
    hypotheses=int(settings['hypotheses']);steps=int(settings.get('conditioning',0))
    state_frames=23*(4+2)
    conditional_frames=1+23*steps*(1+2) if steps else 0
    return dict(internal_route_forwards=routes,trajectory_hypotheses=routes*hypotheses,
        public_FK_chain_state_evaluations=routes*hypotheses*(state_frames+conditional_frames),
        public_terminal_6D_linear_solves=routes*hypotheses*23*steps,
        public_crossing_scalar_updates=routes*hypotheses*23*2*steps,
        native_IK_planner_or_controller_queries=0,
        scope='Internal analytic public robot computations,all counted; only8final geometric candidates reach q')

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
            # Predicted sampled path clearance, never a body safety certificate.
            clear=tip_clear(trajectory,cfg);queries+=1
            reaches=np.linalg.norm(trajectory[-1]-paths[i,-1])<=.03
            event_probability=1/(1+np.exp(np.clip(-prediction['events'][i,k],-40,40)))
            previous=np.concatenate([np.ones((1,2)),np.cumprod(1-event_probability,axis=0)[:-1]],0)
            first=event_probability*previous
            row_mass=np.zeros((2,4))
            for row in range(2):
                x=cfg['row_x'][row]
                rowcfg=dict(cfg,row_x=[x],post_y=[cfg['post_y'][row]],post_heights=[cfg['post_heights'][row]])
                for segment in range(23):
                    point=prediction['event_tip'][i,k,segment,row]
                    probe=np.repeat(point[None],2,axis=0);probe[:,0]=[x-1e-5,x+1e-5]
                    category=word(probe,rowcfg);queries+=1
                    if category not in ('gap0','gap1','gap2','over'):continue
                    compatibility=np.exp(-.5*((point[0]-x)/.02)**2)
                    row_mass[row,('gap0','gap1','gap2','over').index(category)]+=first[segment,row]*compatibility
            # Residual row dependence conditional on propagated latent state is
            # an approximation; marginal composition is not a joint certificate.
            mass=weights[i,k]*survival[i,k]*float(clear and reaches)
            result[i,1:]+=mass*np.outer(row_mass[0],row_mass[1]).reshape(16)
        result[i,0]=1-result[i,1:].sum()
    np.testing.assert_allclose(result.sum(-1),1,atol=1e-5)
    return result.astype(np.float32),dict(prefix_hypotheses_per_route=tips.shape[1],
        predicted_signature_and_tip_proxy_calls=queries,actual_controller_queries=0,
        scope='Public-FK first-crossing event composition with inferred boxes; conditional row independence approximation,not execution validation')
