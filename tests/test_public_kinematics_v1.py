import unittest
import numpy as np
from research_selective_repair_v1.public_kinematics import chain,forward

class PublicFK(unittest.TestCase):
    def test_rotated_base_and_batch(self):
        base=np.array([[0,-1,0,2],[1,0,0,3],[0,0,1,4],[0,0,0,1]],float)
        frames=np.repeat(base[None],7,axis=0)
        # All translations after first revolute joint: analytic one-axis case.
        tip=base.copy();tip[:3,3]+=np.array([0,1,0])
        b,r=chain(frames,tip)
        q=np.zeros((3,7));q[:,0]=[0,np.pi/2,np.pi]
        actual=forward(q,np.zeros(7),b,r)
        np.testing.assert_allclose(actual,[[2,4,4],[1,3,4],[2,2,4]],atol=1e-12)

    def test_reference_pose_and_translation(self):
        frames=np.repeat(np.eye(4)[None],7,axis=0)
        frames[:,0,3]=np.arange(7)*.1
        tip=np.eye(4);tip[0,3]=.8
        q0=np.linspace(-1,1,7);base,relative=chain(frames,tip)
        np.testing.assert_allclose(forward(q0,q0,base,relative),tip[:3,3])
        offset=np.array([.2,-.4,.8]);frames[:,:3,3]+=offset;tip[:3,3]+=offset
        b,r=chain(frames,tip)
        np.testing.assert_allclose(forward(q0,q0,b,r),[1,-.4,.8])

if __name__=='__main__':unittest.main()
