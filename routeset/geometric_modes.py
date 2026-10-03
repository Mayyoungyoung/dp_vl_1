"""Fixed evaluation-only 3D portal words for registered two-row scenes.

Oracle row geometry is exclusively an evaluation label, NEVER model input.
No manual type, endpoint-distance cluster, or fitted threshold is used.
"""
import numpy as np


def portal_word(path, config):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('finite full 3D polyline required')
    margin = config['tip_clearance_m']
    half = config['post_size_xyz'][1] / 2 + margin
    bottom = config['post_base_z'] - margin
    top = config['post_base_z'] + config['post_size_xyz'][2] + margin
    word = []
    for a, b in zip(path[:-1], path[1:]):
        crossings = []
        for row, x in enumerate(config['row_x']):
            # Symbolic half-open side convention: x==plane is on its + side.
            if (a[0] < x) == (b[0] < x):
                continue
            t = (x-a[0])/(b[0]-a[0])
            point = a+t*(b-a)
            low, high = config['post_y'][row]
            y, z = point[1:]
            if y < low-half:
                portal = 'negative_y'
            elif low+half < y < high-half:
                portal = 'middle'
            elif y > high+half:
                portal = 'positive_y'
            elif z > top:
                portal = 'above_post_%d' % (0 if y <= (low+high)/2 else 1)
            elif z < bottom:
                portal = 'below_post_%d' % (0 if y <= (low+high)/2 else 1)
            else:
                # A valid path cannot pass the interior of an expanded post.
                raise ValueError('route crosses blocked portal: validity/geometry mismatch')
            crossings.append((t, (row, 1 if b[0] > a[0] else -1, portal)))
        for _, token in sorted(crossings):
            if word and word[-1][0] == token[0] and word[-1][2] == token[2] and word[-1][1] == -token[1]:
                word.pop()
            else:
                word.append(token)
    return tuple(word)


def encode_word(word):
    return '|'.join('%s:%+d:%s' % token for token in word) or 'no_row_crossing'


def summarize_modes(valid, words, reference_words, selected):
    valid = np.asarray(valid, dtype=bool)
    modes = {words[i] for i in selected if valid[i]}
    refs = set(reference_words)
    return dict(ValidCount=int(sum(valid[i] for i in selected)),
                GeometricModeCount=len(modes), TwoDistinctValid=int(len(modes) >= 2),
                ReferenceModesHit=len(modes & refs),
                ReferenceModeCoverage=len(modes & refs)/len(refs) if len(refs) >= 2 else None)
