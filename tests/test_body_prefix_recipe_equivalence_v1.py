import unittest
import numpy as np
from research_selective_repair_v1.mode_recipe_confusion import effective_training_recipes,effective_inference_recipes,compose

class RecipeEquivalenceTests(unittest.TestCase):
    def test_identity_fallback_has_identical_prediction(self):
        candidates=np.zeros((8,3,24,3));recipes=effective_inference_recipes(candidates)
        table=np.full((6,17,16),1/16);table[5,:,0]=.7;table[5,:,1:]=.3/15
        prob=compose(np.full((8,3),.8),np.ones((8,3),int),recipes,table)
        np.testing.assert_array_equal(prob[:,0],prob[:,2])
        candidates[:,2,2,2]=.02
        self.assertTrue((effective_inference_recipes(candidates)[:,2]==5).all())
    def test_fit_fallback_is_pooled_into_identity(self):
        paths=np.zeros((3,24,3));paths[2,1,2]=.04
        d=dict(ids=np.array(['a','a','a']),slots=np.array([0,0,0]),options=np.array([0,4,5]),paths=paths)
        np.testing.assert_array_equal(effective_training_recipes(d),[0,0,5])

if __name__=='__main__':unittest.main()
