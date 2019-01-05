import tensorflow as tf
import numpy as np
from utility import *
import six
class HParams:
    def __init__(self,**kwargs):
        for name,value in six.iteritems(kwargs):
            self.add_param(name,value)

    def add_param(self,name,value):
        if getattr(self,name,None) is not None:
            raise ValueError("Parameter name is reserved")
        else:
            setattr(self,name,value)


class SGAN(object):
    def __init__(self,hparams):
        self.HParams=hparams
        self.Input_img=tf.placeholder(tf.float32,[self.HParams.batch_size,self.HParams.img_dim])
        self.Input_noise=self.sample_noise()
        with tf.variable_scope("discriminator"):
            W1_conv=tf.get_variable("W1_conv",shape=[5,5,1,32])
            b1_conv=tf.get_variable("b1_conv",shape=32)
            W2_conv=tf.get_variable("W2_conv",shape=[5,5,32,64])
            b2_conv=tf.get_variable("b2_conv",shape=64)
            W1=tf.get_variable("W1",shape=[7*7*64,7*7*64])
            b1=tf.get_variable("b1",shape=7*7*64)
            W2=tf.get_variable("W2",shape=[7*7*64,1])
            b2=tf.get_variable("b2",shape=1)

        with tf.variable_scope("generator"):
            W1=tf.get_variable("W1",shape=[self.HParams.noise_dim,1024])
            b1=tf.get_variable("b1",shape=1024)
            W2=tf.get_variable("W2",shape=[1024,7*7*128])
            b2=tf.get_variable("b2",shape=7*7*128)
            W1_deconv=tf.get_variable("W1_deconv",shape=[4,4,64,128])
            b1_deconv=tf.get_variable("b1_deconv",shape=64)
            W2_deconv=tf.get_variable("W2_deconv",shape=[4,4,1,64])
            b2_deconv=tf.get_variable("b2_deconv",shape=1)
        self.Input_fake_img = self.generator()
        self.logits_real = self.discriminator(preprocess_img(self.Input_img))
        self.logits_fake = self.discriminator(self.Input_fake_img)
        # with tf.variable_scope("") as scope:
        #     self.logits_real = self.discriminator(preprocess_img(self.Input_img))
        #     scope.reuse_variables()
        #     self.logits_fake = self.discriminator(self.Input_fake_img)
        self.get_gan_loss()
        #self.get_ls_gan_loss()
        self.get_optimizer()
        self.get_train_op()


    def sample_noise(self):
        return tf.random_uniform([self.HParams.batch_size,self.HParams.noise_dim],minval=-1,maxval=1)

    def generator(self):
        with tf.variable_scope("generator",reuse=True):
            W1=tf.get_variable("W1")
            b1=tf.get_variable("b1")
            W2=tf.get_variable("W2")
            b2=tf.get_variable("b2")
            W1_deconv=tf.get_variable("W1_deconv")
            b1_deconv=tf.get_variable("b1_deconv")
            W2_deconv=tf.get_variable("W2_deconv")
            b2_deconv=tf.get_variable("b2_deconv")

        a1 = tf.nn.relu(tf.matmul(self.Input_noise, W1) + b1)

        a2 = tf.layers.batch_normalization(a1, training=True)
        a3 = tf.nn.relu(tf.matmul(a2, W2) + b2)
        a4 = tf.layers.batch_normalization(a3, training=True)
        a5 = tf.reshape(a4, [-1, 7, 7, 128])
        a6 = tf.nn.relu(tf.nn.conv2d_transpose(a5, W1_deconv, strides=[1, 2, 2, 1], padding='SAME',
                                               output_shape=[tf.shape(a5)[0], 14, 14, 64]) + b1_deconv)
        a7 = tf.layers.batch_normalization(a6, training=True)
        a8 = tf.nn.tanh(tf.nn.conv2d_transpose(a7, W2_deconv, strides=[1, 2, 2, 1], padding='SAME',
                                               output_shape=[tf.shape(a7)[0], 28, 28, 1]) + b2_deconv)
        a9 = tf.reshape(a8, [-1, 28 * 28])
        img=a9
        return img


    def discriminator(self,x):
        with tf.variable_scope("discriminator",reuse=True):
            W1_conv = tf.get_variable("W1_conv")
            b1_conv = tf.get_variable("b1_conv")
            W2_conv = tf.get_variable("W2_conv")
            b2_conv = tf.get_variable("b2_conv")
            W1 = tf.get_variable("W1")
            b1 = tf.get_variable("b1")
            W2 = tf.get_variable("W2")
            b2 = tf.get_variable("b2")
        x=tf.reshape(x,[-1,28,28,1])
        a1 = tf.nn.leaky_relu(tf.nn.conv2d(x, W1_conv, strides=[1, 1, 1, 1], padding='SAME') + b1_conv)
        a2 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        a3 = tf.nn.leaky_relu(tf.nn.conv2d(a2, W2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv)
        a4 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(a4.get_shape())
        a5 = tf.reshape(a4, [-1, 7 * 7 * 64])
        a6 = tf.nn.leaky_relu(tf.matmul(a5, W1) + b1)
        a7 = tf.matmul(a6, W2) + b2
        logits = a7
        return logits

    def get_gan_loss(self):
        labels_real = tf.ones_like(self.logits_real)
        labels_fake = tf.zeros_like(self.logits_fake)
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=self.logits_real))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=self.logits_fake))
        labels_fake = tf.ones_like(self.logits_fake)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=self.logits_fake))
        self.D_loss=D_loss
        self.G_loss=G_loss

    def get_ls_gan_loss(self):
        D_loss=1/2*tf.reduce_mean(tf.square(self.logits_real-1))+1/2*tf.reduce_mean(tf.square(self.logits_fake))
        G_loss=1/2*tf.reduce_mean(tf.square(self.logits_fake-1))
        self.D_loss=D_loss
        self.G_loss=G_loss


    def get_wessern_gan_loss(self):
        pass

    def get_optimizer(self):
        self.G_solver=tf.train.AdamOptimizer(learning_rate=self.HParams.learning_rate,beta1=self.HParams.beta1)
        self.D_solver=tf.train.AdamOptimizer(learning_rate=self.HParams.learning_rate,beta1=self.HParams.beta1)

    def get_train_op(self):
        D_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
        G_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')
        self.D_train_step=self.D_solver.minimize(self.D_loss,var_list=D_vars)
        self.G_train_step=self.G_solver.minimize(self.G_loss,var_list=G_vars)

    def train_Discriminator(self,sess,minibatch):
        _,d_loss_curr=sess.run([self.D_train_step,self.D_loss],feed_dict={self.Input_img:minibatch})
        return d_loss_curr

    def train_Generator(self,sess):
        _,g_loss_curr=sess.run([self.G_train_step,self.G_loss])
        return g_loss_curr

    def connect_Generator_with_discriminator(self):
        pass


    def get_fake_image(self,sess):
        img=sess.run(self.Input_fake_img)
        return img










