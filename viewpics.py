import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
from paths import DATASETS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import data_config as data_cfg
from easydict import EasyDict as edict

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def view_pics(args):
    datacfg=data_cfg.get_config(args.dataset)
    generated_pics=np.load(os.path.join(DATASETS,args.dataset,'X_gan_%s.npy'%args.model_name))
    np.reshape(generated_pics,[-1,datacfg.dataset.image_size,datacfg.dataset.image_size,datacfg.dataset.channels])
    gs=gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)
    imgs_to_show=np.array(generated_pics[-1:datacfg.dataset.image_size*datacfg.dataset.image_size*datacfg.dataset.channels])

    imgs_to_show=np.reshape(imgs_to_show,[-1,datacfg.dataset.image_size*datacfg.dataset.image_size*datacfg.dataset.channels])
    imgs_to_show=imgs_to_show[-16:]
    plt.figure(figsize=(4,4))

    for i,img in enumerate(imgs_to_show):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        img=np.reshape(img,[datacfg.dataset.image_size,datacfg.dataset.image_size,datacfg.dataset.channels])
        plt.imshow(img)
    return

args=edict({'dataset':'cifar10',
            'model_name':'mlp',
            'image_size':32})

if __name__ =='__main__':
    view_pics(args)
    plt.show()









