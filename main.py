import tensorflow as tf
import numpy as np
from sgan import SGAN,HParams
from utility import *
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
SHOW_INTERVAL=100
BATCH_SIZE=128
NOISE_DIM=96
NUM_EPOCHES=10
PRINT_INTERVAL=50
LEARNING_RATE=1e-3

def constructParams():
    return HParams(
       batch_size=BATCH_SIZE,
       img_dim=784,
       noise_dim=NOISE_DIM,
       learning_rate=LEARNING_RATE,
       beta1=0.5
    )

mnist=input_data.read_data_sets("./data/",one_hot=False)
show_images(mnist.train.next_batch(16)[0])
hparams=constructParams()
sgan=SGAN(hparams=hparams)
max_iter=int(mnist.train.num_examples*NUM_EPOCHES/BATCH_SIZE)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(450):
        if iter %SHOW_INTERVAL ==0:
            samples=sgan.get_fake_image(sess)
        minibatch,_=mnist.train.next_batch(BATCH_SIZE)
        loss_d=sgan.train_Discriminator(sess,minibatch)
        loss_g=sgan.train_Generator(sess)
        if iter % PRINT_INTERVAL==0:
            print('Iter:{},D:{:.4},G:{:.4}'.format(iter,loss_d,loss_g))
    print("final img")
    samples=sgan.get_fake_image(sess)
    fig = show_images(samples[:16])
    plt.show()

