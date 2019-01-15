from data_process import cifar
from data_process import mnist
import tensorflow as tf

def load_dataset(split,batch_size,dataset,size):
    if dataset in ['cifar10','cifar100']:
        if size!=32:
            raise ValueError("CIFAR is only avaliable in 32")
        images,_,init_fn=cifar.load_dataset(split,batch_size,dataset)
    elif dataset == 'mnist':
        images,_,_init_fn=cifar.load_dataset(split,batch_size,dataset)
    else:
        raise NotImplementedError
    return images,init_fn

if __name__ =='__main__':
    load_dataset(128,'cifar10',32)