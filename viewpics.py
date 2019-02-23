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
import scipy.misc
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def view_pics(args,from_disk,generated_imgs,index):
    datacfg=data_cfg.get_config(args.dataset)
    if from_disk:
       generated_pics=np.load(os.path.join(DATASETS,args.dataset,'X_gan_%s.npy'%args.model_name))
    else:
        generated_pics=generated_imgs
    gs=gridspec.GridSpec(8,8)
    gs.update(wspace=0.05, hspace=0.05)
    imgs_to_show=np.array(generated_pics)
    imgs_to_show=np.reshape(imgs_to_show,[-1,datacfg.dataset.image_size*datacfg.dataset.image_size*datacfg.dataset.channels])
    imgs_to_show=imgs_to_show[-16:]
    plt.figure(figsize=(4,4))

    for i,img in enumerate(imgs_to_show):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal') 
        img=np.reshape(img,[datacfg.dataset.image_size,datacfg.dataset.image_size,3]) 
        plt.imshow(img)
    plt.savefig('generated_pic%i.eps'%index,dpi=250)
    return

args=edict({'dataset':'cifar10',
            'model_name':'dcgan',
            'image_size':32})
def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')
def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def inverse_transform(images):
  return (images+1.)/2.

def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w

generated_pic_dir='./samples/dcgan/cifar10/X_gan_dcgan.npy'
real_pic_dit='./datasets/cifar10/X_train.npy'

def denormalize(images):
  images= (images+1)*127.5
  images=images.astype(np.uint8)
  return images


denorm=True


if __name__ =='__main__':
    generated_pics=np.load(generated_pic_dir)
    #print(generated_pics)
    gs=gridspec.GridSpec(8,8)
    gs.update(wspace=0.05, hspace=0.05)
    imgs_to_show=np.array(generated_pics)
    imgs_to_show=np.reshape(imgs_to_show,[-1,32*32*3])
    imgs_to_show=imgs_to_show[-64:]
    #imgs_to_show=inverse_transform(imgs_to_show)
    plt.figure(figsize=(8,8))
    if denorm:
      imgs_to_show=denormalize(imgs_to_show)
    print(imgs_to_show)
    for i,img in enumerate(imgs_to_show):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal') 
        img=np.reshape(img,[32,32,3]) 
        plt.imshow(img)
    plt.savefig('generated_pics',dpi=250)
    plt.show()









