"""Paired all-positive TRAIN stream, separate from controlled mode filtering.

No mode field is read. Observation indices, selected references, permutation,
time and actual epsilon bytes are audited identically for both model arms.
This logical stream is checkpointable; the caller separately journals issued
model calls so a hard crash cannot silently replay extra computation.
"""
import copy
import hashlib
import json

import numpy as np
import torch

from .observed_training_audit import PROTOCOL as INDEX_PROTOCOL, append_indices


PROTOCOL = 'observed_all_positive_diffusion_stream_v1'


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf-8')


def _array(digest,value):
    value = np.ascontiguousarray(value)
    digest.update(_json([str(value.dtype),list(value.shape)]))
    digest.update(value.tobytes())


def _hex(value):
    try:
        return isinstance(value,str) and len(value)==64 and len(bytes.fromhex(value))==32
    except ValueError:
        return False


class PairedPositiveStream:
    def __init__(self,data,train_ids,seed=0,k=4):
        if type(seed) is not int or seed < 0 or k != 4:
            raise ValueError('nonnegative seed and fixed K4 required')
        ids = np.asarray(train_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or len(set(ids.tolist())) != len(ids):
            raise ValueError('nonempty unique TRAIN index vector required')
        splits = np.asarray(data['splits'])
        if not np.array_equal(ids,np.flatnonzero(splits=='TRAIN')):
            raise ValueError('all TRAIN observations in canonical order required; no filtering')
        mask = np.asarray(data['path_mask'])
        paths,events = np.asarray(data['paths']),np.asarray(data['events'])
        if mask.dtype != np.bool_ or mask.ndim != 2 or paths.shape != mask.shape+(24,3) or events.shape != mask.shape+(24,):
            raise ValueError('H24 paths/events and boolean positive mask required')
        if not mask[ids].any(axis=1).all():
            raise ValueError('empty positive sets cannot be converted to negative labels or silently skipped')
        if any(len(data[key])!=len(splits) for key in ('scene_ids','parent_ids')) or len(mask)!=len(splits):
            raise ValueError('dataset metadata length mismatch')
        self.train_ids = ids.astype(np.int64,copy=True)
        self.scene_ids = [str(data['scene_ids'][i]) for i in ids]
        self.parent_ids = [str(data['parent_ids'][i]) for i in ids]
        if len(set(self.scene_ids)) != len(ids):
            raise ValueError('unique observation identities required')
        self.valid_refs = [np.flatnonzero(mask[i]).astype(np.int64) for i in ids]
        self.row_position = {int(idx):position for position,idx in enumerate(ids)}
        digest = hashlib.sha256(PROTOCOL.encode())
        digest.update(_json(dict(scene_ids=self.scene_ids,parent_ids=self.parent_ids,
            data_fingerprint=data.get('fingerprint'),train_ids=ids.tolist())))
        for i,refs in zip(ids,self.valid_refs):
            if not np.isfinite(paths[i,refs]).all() or not np.isfinite(events[i,refs]).all():
                raise ValueError('nonfinite positive reference')
            _array(digest,refs.astype('<i8'))
            _array(digest,paths[i,refs].astype('<f4'))
            _array(digest,events[i,refs].astype('<f4'))
        self.identity = dict(protocol=PROTOCOL,seed=seed,k=k,horizon=24,diffusion_steps=100,
            data_sha256=digest.hexdigest(),population=len(ids),train_ids=ids.tolist(),
            seed_offsets=dict(observation=100000,positive=200000,time=300000,epsilon=400000))
        self.parents = np.random.default_rng(seed+100000)
        self.positives = np.random.default_rng(seed+200000)
        self.times = torch.Generator(device='cpu').manual_seed(seed+300000)
        self.noise = torch.Generator(device='cpu').manual_seed(seed+400000)
        self.initial_sampler_state_sha256 = hashlib.sha256(json.dumps(self.parents.bit_generator.state,sort_keys=True).encode()).hexdigest()
        self.index_audit = dict(batches=0,observation_draws=0,index_chain_sha256=hashlib.sha256(INDEX_PROTOCOL.encode()).hexdigest())
        self.chain = hashlib.sha256(_json(self.identity)).hexdigest()
        self.target_states = self.pool_visits = self.duplicate_targets = 0
        self.reference_frequencies = [[0]*mask.shape[1] for _ in ids]

    def draw(self,batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('positive integer batch size required')
        indices = self.parents.choice(self.train_ids,batch_size,replace=True)
        chosen,permutations,counts = [],[],[]
        for idx in indices:
            position = self.row_position[int(idx)]
            refs = self.valid_refs[position]
            if len(refs)>=4:
                base = self.positives.choice(refs,4,replace=False)
            else:
                base = np.concatenate((refs,self.positives.choice(refs,4-len(refs),replace=True)))
            permutation = self.positives.permutation(4)
            target = base[permutation]
            chosen.append(target);permutations.append(permutation);counts.append(len(refs))
            self.pool_visits += len(refs)
            self.duplicate_targets += 4-len(set(target.tolist()))
            for ref in target:
                self.reference_frequencies[position][int(ref)] += 1
        references = np.asarray(chosen,dtype=np.int64)
        permutations = np.asarray(permutations,dtype=np.int64)
        timesteps = torch.randint(0,100,(batch_size,),generator=self.times,dtype=torch.long)
        epsilon = torch.randn((batch_size,4,23,4),generator=self.noise,dtype=torch.float32)
        digest = hashlib.sha256(bytes.fromhex(self.chain))
        digest.update(_json(dict(ids=[self.scene_ids[self.row_position[int(i)]] for i in indices],
                                parents=[self.parent_ids[self.row_position[int(i)]] for i in indices])))
        for array in (indices.astype('<i8'),references.astype('<i8'),permutations.astype('<i8'),
                      timesteps.numpy().astype('<i8'),epsilon.numpy().astype('<f4')):
            _array(digest,array)
        self.chain = digest.hexdigest()
        self.index_audit = append_indices(self.index_audit,indices)
        self.target_states += batch_size*4
        return dict(indices=indices,reference_indices=references,reference_permutations=permutations,
            reference_counts=np.asarray(counts,dtype=np.int64),timesteps=timesteps,epsilon=epsilon)

    def audit(self):
        return dict(identity=copy.deepcopy(self.identity),batches=self.index_audit['batches'],
            observation_draws=self.index_audit['observation_draws'],target_path_states=self.target_states,
            all_positive_pool_visits=self.pool_visits,sampled_duplicate_targets=self.duplicate_targets,
            actual_stream_sha256=self.chain,ordinary_index_chain_sha256=self.index_audit['index_chain_sha256'],
            initial_sampler_state_sha256=self.initial_sampler_state_sha256,
            reference_frequencies=copy.deepcopy(self.reference_frequencies),
            positive_reference_counts=[len(x) for x in self.valid_refs],
            scene_ids=list(self.scene_ids),parent_ids=list(self.parent_ids))

    def state_dict(self):
        return dict(identity=copy.deepcopy(self.identity),parents=copy.deepcopy(self.parents.bit_generator.state),
            positives=copy.deepcopy(self.positives.bit_generator.state),times=self.times.get_state().clone(),
            noise=self.noise.get_state().clone(),audit=self.audit())

    def load_state_dict(self,state):
        if not isinstance(state,dict) or set(state)!=set(self.state_dict()) or state['identity']!=self.identity:
            raise ValueError('stream resume protocol/data/seed/budget mismatch')
        audit = state['audit']
        if not isinstance(audit,dict) or set(audit)!=set(self.audit()) or audit['identity']!=self.identity:
            raise ValueError('stream audit identity mismatch')
        for key in ('initial_sampler_state_sha256','positive_reference_counts','scene_ids','parent_ids'):
            if audit[key]!=self.audit()[key]:
                raise ValueError('stream fixed audit metadata differs: '+key)
        for key in ('batches','observation_draws','target_path_states','all_positive_pool_visits','sampled_duplicate_targets'):
            if type(audit[key]) is not int or audit[key]<0:
                raise ValueError('invalid stream counts')
        if audit['observation_draws']<audit['batches'] or audit['target_path_states']!=4*audit['observation_draws']:
            raise ValueError('stream exposure mismatch')
        if audit['all_positive_pool_visits']<audit['observation_draws'] or audit['sampled_duplicate_targets']>3*audit['observation_draws']:
            raise ValueError('stream positive counts mismatch')
        frequencies = audit['reference_frequencies']
        if len(frequencies)!=len(self.train_ids) or any(len(row)!=len(self.reference_frequencies[i]) for i,row in enumerate(frequencies)):
            raise ValueError('reference frequency shape mismatch')
        for refs,counts in zip(self.valid_refs,frequencies):
            if any(type(x) is not int or x<0 for x in counts) or any(counts[i] for i in range(len(counts)) if i not in refs):
                raise ValueError('frequency for invalid/unrecorded positive')
        if sum(sum(row) for row in frequencies)!=audit['target_path_states']:
            raise ValueError('reference frequency exposure mismatch')
        if not _hex(audit['actual_stream_sha256']) or not _hex(audit['ordinary_index_chain_sha256']):
            raise ValueError('invalid stream digest')
        # Validate RNG states on temporary generators before mutating this stream.
        parents,positives = np.random.default_rng(),np.random.default_rng()
        parents.bit_generator.state = copy.deepcopy(state['parents'])
        positives.bit_generator.state = copy.deepcopy(state['positives'])
        times,noise = torch.Generator(device='cpu'),torch.Generator(device='cpu')
        times.set_state(state['times'].cpu());noise.set_state(state['noise'].cpu())
        self.parents,self.positives,self.times,self.noise = parents,positives,times,noise
        self.index_audit = dict(batches=audit['batches'],observation_draws=audit['observation_draws'],
            index_chain_sha256=audit['ordinary_index_chain_sha256'])
        self.chain,self.target_states = audit['actual_stream_sha256'],audit['target_path_states']
        self.pool_visits,self.duplicate_targets = audit['all_positive_pool_visits'],audit['sampled_duplicate_targets']
        self.reference_frequencies = copy.deepcopy(frequencies)
