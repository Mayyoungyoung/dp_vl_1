"""Static scientific figure of actual pilot RGB and certified teacher paths."""
import argparse
import json
from pathlib import Path
import numpy as np


def plot(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    fig,axes=plt.subplots(2,3,figsize=(12,7),gridspec_kw={'height_ratios':[1,1.1]},layout='constrained')
    for col,variant in enumerate(('open','closed','shifted')):
        parent='paired_family_520000_'+variant;folder=root/parent;scene=folder/parent
        axes[0,col].imshow(Image.open(scene/'front.png'));axes[0,col].axis('off');axes[0,col].set_title(variant.title())
        ax=axes[1,col]
        with np.load(scene/'verification_only.npz') as a:
            for c,h in zip(a['obstacle_centers'],a['obstacle_halfsizes']):
                ax.add_patch(Rectangle((c[0]-h[0],c[1]-h[1]),2*h[0],2*h[1],facecolor='.3',zorder=5))
                ax.add_patch(Rectangle((c[0]-h[0]-.02,c[1]-h[1]-.02),2*h[0]+.04,2*h[1]+.04,fill=False,linestyle=':',edgecolor='.5'))
            goals=a['target_centers'];ax.scatter(goals[:,0],goals[:,1],c=['dodgerblue','silver','black'],s=60,zorder=6)
        for file in sorted(scene.glob('target1_route*.npz')):
            with np.load(file) as a:path=a['gripper_pose'][:,:3]
            above=bool(np.max(path[:,2])>.91)
            ax.plot(path[:,0],path[:,1],ls='--' if above else '-',alpha=.8,lw=1.5)
        ax.set(xlabel='x (m)',ylabel='y (m)',xlim=(-.02,.52),ylim=(-.3,.3));ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Actual paired RGB-D scenes and geometric teacher paths\nTop-down paths target the same center sphere; geometry labels are training/evaluation only',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.root,a.output)
