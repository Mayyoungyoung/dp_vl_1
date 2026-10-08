"""Operational passage labels, not a complete or strict homotopy classifier.

At each registered row's FIRST FORWARD crossing, above the inflated row top
has precedence; otherwise the lateral free interval defines gap0/1/2.
No narrow unclassified height strip and no above-post left/right subdivision.
Only classify checker-valid paths. Oracle row parameters are labels, not inputs.
"""
import numpy as np


def passage_signature(path, config):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path)<2 or not np.isfinite(path).all():return None
    labels=[]; margin=config['tip_clearance_m']
    for r,x in enumerate(config['row_x']):
        ys=config['post_y'][r]
        height=config.get('post_heights',[config.get('post_size_xyz',[0,0,0])[2]]*len(config['row_x']))[r]
        top=config['post_base_z']+height+margin
        found=None
        for a,b in zip(path[:-1],path[1:]):
            if not a[0] < x <= b[0]:continue
            point=a+(x-a[0])/(b[0]-a[0])*(b-a)
            y,z=point[1:]
            if z>top:found='over'
            elif y<ys[0]-.0175-margin:found='gap0'
            elif y>ys[-1]+.0175+margin:found='gap%d'%len(ys)
            else:
                gaps=[j for j in range(1,len(ys)) if ys[j-1]+.0175+margin<y<ys[j]-.0175-margin]
                if len(gaps)==1:found='gap%d'%gaps[0]
            break
        if found is None:return None
        labels.append(found)
    return tuple(labels)
