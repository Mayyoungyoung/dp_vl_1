"""Actual all-positive sampling, paired streams, and resumed updates."""
import copy
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from routeset.observed_diffusion_stream import PairedPositiveStream
    from routeset.observed_training_audit import append_indices


def fixture():
    rng=np.random.default_rng(10)
    mask=np.array([[True,False,False,False,False,False],
                   [True,False,True,False,False,False],
                   [True,True,True,True,True,True],
                   [True,True,True,True,False,False]])
    paths=rng.normal(size=(4,6,24,3)).astype(np.float32)
    events=rng.integers(0,2,size=(4,6,24)).astype(np.float32)
    paths[~mask]=np.nan;events[~mask]=np.nan
    return dict(paths=paths,events=events,path_mask=mask,splits=np.array(['TRAIN']*4),
        scene_ids=np.array(['a','b','c','d']),parent_ids=np.array(['p0','p0','p1','p1']),
        fingerprint='fixture',modes=np.full((4,6),-1))


class FixtureTests(unittest.TestCase):
    def test_fixture_contains_sparse_unknown_positive_support(self):
        data=fixture()
        self.assertEqual(data['path_mask'].sum(axis=1).tolist(),[1,2,6,4])
        self.assertTrue((data['modes'][data['path_mask']]==-1).all())


@unittest.skipIf(torch is None,'requires actual Torch RNG and optimizer')
class StreamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data=fixture();self.ids=np.arange(4)

    def test_unknown_sparse_positives_and_subset_rules(self):
        stream=PairedPositiveStream(self.data,self.ids)
        batch=stream.draw(128)
        for idx,refs in zip(batch['indices'],batch['reference_indices']):
            self.assertTrue(self.data['path_mask'][idx,refs].all())
            expected=set(np.flatnonzero(self.data['path_mask'][idx]).tolist())
            if len(expected)<=4:self.assertEqual(set(refs.tolist()),expected)
            else:self.assertEqual(len(set(refs.tolist())),4)
        audit=stream.audit()
        self.assertEqual(audit['target_path_states'],512)
        self.assertEqual(sum(map(sum,audit['reference_frequencies'])),512)
        self.assertEqual(audit['initial_sampler_state_sha256'],
            '48a64e771100689dadc6faec20d1a9d312e0e35cbd828ef0c10e20de352a56b6')

    def test_actual_pair_and_ordinary_parent_stream_match(self):
        first=PairedPositiveStream(self.data,self.ids)
        second=PairedPositiveStream(self.data,self.ids)
        ordinary=np.random.default_rng(100000)
        for _ in range(4):
            a,b=first.draw(7),second.draw(7)
            for key in a:
                np.testing.assert_array_equal(np.asarray(a[key]),np.asarray(b[key]))
            np.testing.assert_array_equal(a['indices'],ordinary.choice(self.ids,7,replace=True))
        self.assertEqual(first.audit(),second.audit())

    def test_four_vs_two_restore_two_actual_stream_exact(self):
        full=PairedPositiveStream(self.data,self.ids)
        for _ in range(4):full.draw(3)
        first=PairedPositiveStream(self.data,self.ids)
        for _ in range(2):first.draw(3)
        saved=copy.deepcopy(first.state_dict())
        second=PairedPositiveStream(self.data,self.ids)
        second.load_state_dict(saved)
        for _ in range(2):second.draw(3)
        self.assertEqual(full.audit(),second.audit())
        for key in ('times','noise'):self.assertTrue(torch.equal(full.state_dict()[key],second.state_dict()[key]))
        self.assertEqual(full.state_dict()['parents'],second.state_dict()['parents'])
        self.assertEqual(full.state_dict()['positives'],second.state_dict()['positives'])
        # Advance both restored/continuous streams exactly once for a future batch.
        a,b=full.draw(5),second.draw(5)
        for key in a:np.testing.assert_array_equal(np.asarray(a[key]),np.asarray(b[key]))

    def test_changed_data_seed_population_and_corrupt_counts_rejected(self):
        stream=PairedPositiveStream(self.data,self.ids);stream.draw(3)
        state=stream.state_dict()
        changed=copy.deepcopy(self.data);changed['paths'][0,0,3,0]+=.001
        with self.assertRaises(ValueError):PairedPositiveStream(changed,self.ids).load_state_dict(state)
        with self.assertRaises(ValueError):PairedPositiveStream(self.data,self.ids,seed=1).load_state_dict(state)
        with self.assertRaises(ValueError):PairedPositiveStream(self.data,self.ids[::-1])
        bad=copy.deepcopy(state);bad['audit']['target_path_states']+=4
        with self.assertRaises(ValueError):PairedPositiveStream(self.data,self.ids).load_state_dict(bad)

    def test_empty_positive_and_unapproved_population_rejected(self):
        data=copy.deepcopy(self.data);data['path_mask'][1]=False
        with self.assertRaises(ValueError):PairedPositiveStream(data,self.ids)
        data=copy.deepcopy(self.data);data['splits'][3]='DEV_MODEL'
        with self.assertRaises(ValueError):PairedPositiveStream(data,self.ids)
        with self.assertRaises(ValueError):PairedPositiveStream(self.data,self.ids[:3])

    def test_global_rng_independence(self):
        before=torch.get_rng_state().clone()
        stream=PairedPositiveStream(self.data,self.ids);stream.draw(3)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        stream.load_state_dict(stream.state_dict())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_tiny_true_adam_four_vs_two_restore_two(self):
        # The driver tests the real model/checkpoint/journal. This tests the
        # stream's exact coupling to a real differentiable Torch update.
        def start():
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(51)
                model=torch.nn.Linear(4,4)
            return model,torch.optim.AdamW(model.parameters(),lr=3e-4),PairedPositiveStream(self.data,self.ids)
        def steps(model,opt,stream,count):
            for _ in range(count):
                draw=stream.draw(2)
                ids,refs=draw['indices'],draw['reference_indices']
                target=torch.from_numpy(self.data['paths'][ids[:,None],refs,1:])
                features=draw['epsilon'].reshape(-1,4)
                wanted=torch.cat((target,torch.zeros(2,4,23,1)),dim=-1).reshape(-1,4)
                loss=(model(features)-wanted).square().mean()
                opt.zero_grad();loss.backward();opt.step()
        full,optfull,streamfull=start();steps(full,optfull,streamfull,4)
        half,opthalf,streamhalf=start();steps(half,opthalf,streamhalf,2)
        checkpoint=copy.deepcopy(dict(model=half.state_dict(),optimizer=opthalf.state_dict(),stream=streamhalf.state_dict()))
        resumed,optresume,streamresume=start()
        resumed.load_state_dict(checkpoint['model']);optresume.load_state_dict(checkpoint['optimizer']);streamresume.load_state_dict(checkpoint['stream'])
        steps(resumed,optresume,streamresume,2)
        for key,value in full.state_dict().items():self.assertTrue(torch.equal(value,resumed.state_dict()[key]))
        for idx,state in optfull.state_dict()['state'].items():
            for key,value in state.items():
                other=optresume.state_dict()['state'][idx][key]
                if torch.is_tensor(value):self.assertTrue(torch.equal(value,other))
                else:self.assertEqual(value,other)
        self.assertEqual(streamfull.audit(),streamresume.audit())


if __name__=='__main__':unittest.main()
