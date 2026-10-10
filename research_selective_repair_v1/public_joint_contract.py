"""Read-only native joint intervals from one registered TRAIN initialization."""
import argparse,copy,os
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,SOURCE,read,write,sha

def capture(name):
    from scripts import collect_observed_layout_variation as physical
    from scripts.collect_observed_layout_hash_recovery import validate_initial_geometry
    registration=read(ROOT/'data/selective_repair_interventions_v1/registration.json')
    plan=copy.deepcopy(next(p for p in registration['parent_plan'] if p['role']=='TRAIN'))
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    def callback(task,posts,targets,saved,plan,output,phase,counts,gripper_shapes,external_shapes):
        arm=task._robot.arm
        cyclic,intervals=arm.get_joint_intervals()
        q=np.asarray(arm.get_joint_positions())
        interval=np.asarray(intervals);lower=interval[:,0];upper=lower+interval[:,1]
        assert q.shape==(7,) and interval.shape==(7,2) and np.all(upper>lower)
        assert np.all((q>=lower)&(q<=upper))
        write(out/'PUBLIC_JOINT_CONTRACT.json',dict(cyclic=list(map(bool,cyclic)),lower=lower.tolist(),upper=upper.tolist(),q0=q.tolist(),
            source_commit=os.environ.get('CODE_COMMIT'),native_API='arm.get_joint_intervals/get_joint_positions',
            scope='Fixed robot hardware/public initial state only; no environment geometry,IK,execution or simulator step added by callback',
            interval_semantics='Native [minimum,range],converted to lower/upper radians',locked_access=False))
        return [],[],False
    physical.collect_routes=callback;physical.validate_initial_geometry=validate_initial_geometry
    physical.PROTOCOL='public_fixed_robot_joint_contract_v1'
    file=out/'plan.json';write(file,plan)
    result=physical.physical_worker(registration,{},plan,file,out/'initialization')
    assert result['status']=='collection_finished' and result['route_attempts']==0
    write(out/'SUMMARY.json',dict(status='completed',route_attempts=0,contract_sha256=sha(out/'PUBLIC_JOINT_CONTRACT.json'),
        actual_initialization_seconds=result['elapsed_seconds'],locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);capture(**vars(p.parse_args()))
