"""Export metadata must not inherit the raw collector's larger DEV denominator."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scripts.evaluate_observed_obstacles import evaluate,sha256


class ReservedObstacleEvaluationTests(unittest.TestCase):
    def test_equal_geometry_metrics_and_reserved_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'export';data.mkdir();source=root/'raw';source.mkdir()
            metadata=dict(acceptance=dict(tip_polyline_clearance_m=.02),parents_requested=128,dev_parents=32)
            (source/'manifest.json').write_text(json.dumps(metadata))
            observations=[dict(id='parent_target0',parent_id='parent',split='DEV_MODEL',image='unused.png',instruction='fixture')]
            label=dict(id='parent_target0',parent_id='parent',split='DEV_MODEL',observation='current.npz',verification_only='boxes.npz',
                semantic_targets=dict(centers=[[0.,0.,-.4]],target_index=0,tolerance=.03),routes=[],route_types=[])
            (data/'observations.jsonl').write_text(json.dumps(observations[0])+'\n')
            (data/'supervision.jsonl').write_text(json.dumps(label)+'\n')
            np.savez(data/'current.npz',gripper_pose=np.array([0.,0.,.4,0.,0.,0.,1.]),gripper_open=1.)
            np.savez(data/'boxes.npz',obstacle_centers=np.array([[0.,0.,0.]]),obstacle_halfsizes=np.array([[.1,.1,.1]]))
            path=np.array([[0.,0.,.4],[-.3,0.,.15],[-.3,0.,-.15],[0.,0.,-.4]])
            np.savez(data/'predictions.npz',scene_ids=np.array(['parent_target0']),parent_ids=np.array(['parent']),
                     paths=path[None,None],gripper_open=np.ones((1,1,4)))
            export=dict(source_dataset=str(source),requested_roles=['TRAIN','DEV_MODEL'],requested_parent_counts={'TRAIN':96,'DEV_MODEL':1},
                reservation_sha256='fixture',input_manifest_sha256=sha256(data/'observations.jsonl'),
                supervision_manifest_sha256=sha256(data/'supervision.jsonl'))
            (data/'export_manifest.json').write_text(json.dumps(export))
            actual=evaluate(data,data/'predictions.npz',root/'reserved_result')
            self.assertEqual(actual['TipValidAtK'],1.)
            self.assertEqual(actual['examples_without_reference'],1)
            self.assertEqual(actual['requested_collection_parents_for_split'],1)
            self.assertEqual(actual['collection_parents_without_observations'],0)
            with self.assertRaisesRegex(ValueError,'not explicitly exported'):
                evaluate(data,data/'predictions.npz',root/'forbidden',split='CALIBRATION')
            # Existing datasets follow their unchanged metadata and same checks.
            (data/'manifest.json').write_text(json.dumps(metadata))
            historical=evaluate(data,data/'predictions.npz',root/'historical_result')
            for key in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','TipClearAtK','semantic_goal_accuracy'):
                self.assertEqual(actual[key],historical[key])
            self.assertEqual(historical['requested_collection_parents_for_split'],32)


if __name__=='__main__':unittest.main()
