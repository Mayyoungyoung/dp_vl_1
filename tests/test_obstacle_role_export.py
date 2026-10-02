"""Reservation boundaries and closure accounting for the original obstacle log."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_observation_roles import build, write_rows


class ObstacleRoleExportTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'data/obstacle'; source.mkdir(parents=True)
        reservation=root/'reservation.json'
        reservation.write_text(json.dumps(dict(registered_utc='fixture',collections=[dict(
            source='data/obstacle',parent_prefix='parent_',roles_inclusive={
                'TRAIN':[1,2],'DEV_MODEL':[3,3],'TEST_LOCKED':[4,4]})])))
        observations=[]; supervision=[]; attempts=[]
        for number in (1,3):
            parent='parent_%06d'%number; folder=source/parent; folder.mkdir()
            for name in ('front.png','observation.npz','verification_only.npz'):
                (folder/name).write_bytes(b'fixture')
            for target in range(3):
                identifier=parent+'_target'+str(target); routes=[]
                for attempt in range(4):
                    success=target!=2 and attempt==0
                    row=dict(parent_id=parent,split='RAW',input_id=identifier,attempt=attempt,success=success)
                    if success:
                        filename=parent+'/route%d_%d.npz'%(target,attempt)
                        (source/filename).write_bytes(b'route');routes.append(filename);row['route_file']=filename
                    attempts.append(row)
                observations.append(dict(id=identifier,parent_id=parent,split='RAW',image=parent+'/front.png',instruction='fixture'))
                supervision.append(dict(id=identifier,parent_id=parent,split='RAW',observation=parent+'/observation.npz',
                    verification_only=parent+'/verification_only.npz',routes=routes))
        attempts.append(dict(parent_id='parent_000002',split='RAW',phase='parent_setup',success=False,error='fixture failure'))
        for key,rows in [('observations',observations),('supervision',supervision),('attempts',attempts)]:
            write_rows(source/(key+'.jsonl'),rows)
            with (source/(key+'.jsonl')).open('ab') as stream:
                stream.write(b'{"parent_id":"parent_000004","must_not_decode":BROKEN_LOCKED_PAYLOAD}\n')
        return source,reservation,attempts

    def test_reserved_export_preserves_failures_zero_refs_and_original_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,reservation,_=self.fixture(root)
            before={path:path.read_bytes() for path in source.glob('*.jsonl')}
            manifest=build(root,reservation,'data/obstacle',['TRAIN','DEV_MODEL'],root/'export',3,4,'obstacle_v3')
            manifest=json.loads((root/'export/export_manifest.json').read_text())
            self.assertEqual(manifest['requested_parent_counts'],{'TRAIN':2,'DEV_MODEL':1})
            self.assertEqual(manifest['setup_failed_parents'],['parent_000002'])
            self.assertEqual(manifest['unattempted_proposals'],12)
            self.assertEqual(manifest['observations'],6)
            self.assertEqual(manifest['supervised_examples_by_role'],{'TRAIN':2,'DEV_MODEL':2})
            labels=[json.loads(line) for line in (root/'export/supervision.jsonl').read_text().splitlines()]
            self.assertEqual(sum(not row['routes'] for row in labels),2)
            self.assertTrue(all(Path(row['verification_only']).parent.name==row['parent_id'] for row in labels))
            for path,value in before.items():self.assertEqual(path.read_bytes(),value)
            with self.assertRaisesRegex(ValueError,'only explicit'):
                build(root,reservation,'data/obstacle',['TEST_LOCKED'],root/'forbidden',3,4,'obstacle_v3')

    def test_partial_parent_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,reservation,attempts=self.fixture(root)
            write_rows(source/'attempts.jsonl',attempts[1:])
            with self.assertRaisesRegex(ValueError,'all requested proposal'):
                build(root,reservation,'data/obstacle',['TRAIN'],root/'partial',3,4,'obstacle_v3')
            self.assertFalse((root/'partial').exists())


if __name__=='__main__':unittest.main()
