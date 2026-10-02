"""Optional, resumable audit of actual observation-index draws; no RNG calls."""
import hashlib
import json
import struct

import numpy as np


PROTOCOL = 'actual_observation_indices_sha256_chain_v1'


def tensor_state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        # Flatten before uint8 view also supports scalar optimizer buffers.
        digest.update(tensor.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def new_stream_audit(model, sampler_state, torch_rng):
    return dict(protocol=PROTOCOL, batches=0, observation_draws=0,
        initial_model_sha256=tensor_state_digest(model.state_dict()),
        initial_torch_cpu_rng_sha256=hashlib.sha256(torch_rng.cpu().numpy().tobytes()).hexdigest(),
        initial_sampler_state_sha256=hashlib.sha256(json.dumps(sampler_state, sort_keys=True).encode()).hexdigest(),
        index_chain_sha256=hashlib.sha256(PROTOCOL.encode()).hexdigest())


def append_indices(audit, indices):
    """Hash the returned integer index array, preserving actual order/repeats."""
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or np.any(values < 0):
        raise ValueError('Actual sampled indices must be a nonnegative integer vector')
    result = dict(audit)
    digest = hashlib.sha256(bytes.fromhex(audit['index_chain_sha256']))
    digest.update(struct.pack('<Q', len(values)))
    digest.update(values.astype('<u8', copy=False).tobytes())
    result.update(batches=audit['batches'] + 1,
        observation_draws=audit['observation_draws'] + len(values), index_chain_sha256=digest.hexdigest())
    return result


def restore_stream_audit(enabled, saved_config, saved, initial, step, batch_size):
    if bool(enabled) != bool(saved_config.get('sample_stream_audit', False)):
        raise ValueError('resume config mismatch: sample_stream_audit')
    if not enabled:
        return None
    if not isinstance(saved, dict) or set(saved) != set(initial):
        raise ValueError('Missing or malformed checkpoint sample-stream audit')
    if saved['protocol'] != PROTOCOL or saved['batches'] != step or saved['observation_draws'] != step * batch_size:
        raise ValueError('Checkpoint sample-stream count/protocol mismatch')
    if any(saved[name] != initial[name] for name in initial if name.startswith('initial_')):
        raise ValueError('Reconstructed initialization differs from audited initial state')
    try:
        if len(bytes.fromhex(saved['index_chain_sha256'])) != 32:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('Malformed checkpoint sample-stream digest')
    return dict(saved)
