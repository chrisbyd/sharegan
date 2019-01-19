import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

cifar10_config=edict(
    {
        'evaluation':{},
        'training':{},
        'dataset':{}
    }
)
cifar10_config.dataset.name='cifar10'
cifar10_config.dataset.num_classes=10
cifar10_config.dataset.image_size= 32
cifar10_config.dataset.channels=3
cifar10_config.evaluation.batch_size=500
cifar10_config.training.batch_size=128
cifar10_config.training.max_iterations=64000
cifar10_config.training.ckpt_step=1000
cifar10_config.training.eval_step=5000
cifar10_config.training.print_step=500
cifar10_config.training.summary_step=500

cifar100_config=edict(cifar10_config)
cifar100_config.dataset.name='cifar100'
cifar100_config.dataset.num_classes=100
cifar100_config.dataset.channels=3

mnist_config=edict(
    {
        'dataset':{},
        'evaluation':{},
        'training':{}
    }
)
mnist_config.dataset.name='mnist'
mnist_config.dataset.image_size=28
mnist_config.dataset.channels=1
mnist_config.evaluation.batch_size=500
mnist_config.training.batch_size=128
mnist_config.training.max_iterations=64000
mnist_config.training.ckpt_step=1000
mnist_config.training.eval_step=5000
mnist_config.training.print_step=500
mnist_config.training.summary_step=500



def get_config(dataset):
    if dataset=='mnist':
        return edict(mnist_config)
    elif dataset=='cifar10':
        return edict(cifar10_config)
    elif dataset=='cifar100':
        return edict(cifar100_config)
    else:
        raise NotImplementedError



