import tensorflow as tf
from utility import *
import numpy as np
class WSGAN(object):
    def __init__(self,hparams):
        self.HParams=hparams
        self.Input_img=tf.placeholder(tf.float32,[self.HParams.batch_size,self.HParams.img_dim])
        self.Input_noise=self.sample_noise()
        with tf.variable_scope("discriminator"):
            tf.get_variable("W1_conv",shape=[4,4,1,64])
            tf.get_variable("b1_conv",shape=64)
            tf.get_variable("W2_conv",shape=[4,4,64,128])
            tf.get_variable("b2_conv",shape=128)
            tf.get_variable("W1",shape=[7*7*64,7*7*64])
            tf.get_variable("b1",shape=7*7*64)
            tf.get_variable("W2",shape=[7*7*64,1])
            tf.get_variable("b2",shape=1)

        with tf.variable_scope("generator"):
            tf.get_variable("W1",shape=[self.HParams.noise_dim,1024])
            tf.get_variable("b1",shape=1024)
            tf.get_variable("W2",shape=[1024,7*7*128])
            tf.get_variable("b2",shape=7*7*128)
            tf.get_variable("W1_deconv",shape=[4,4,64,128])
            tf.get_variable("b1_deconv",shape=64)
            tf.get_variable("W2_deconv",shape=[4,4,1,64])
            tf.get_variable("b2_deconv",shape=1)
        self.logits_real = self.discriminator(preprocess_img(self.Input_img))
        self.Input_fake_img = self.generator()

        self.logits_fake = self.discriminator(self.Input_fake_img)
        self.output_fake_img = self.output_generator()
        self.get_wganp_loss()
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
                                               output_shape=[tf.shape(a5)[0], 14, 14, 64]) + b1_deconv)#[-1,14,14,64]
        a6=a6+tf.layers.dropout(self.da1,0.3)
        a7 = tf.layers.batch_normalization(a6, training=True)
        a8 = tf.nn.tanh(tf.nn.conv2d_transpose(a7, W2_deconv, strides=[1, 2, 2, 1], padding='SAME',
                                               output_shape=[tf.shape(a7)[0], 28, 28, 1]) + b2_deconv)
        a9 = tf.reshape(a8, [-1, 28 * 28])
        img=a9
        return img

    def output_generator(self):
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
                                               output_shape=[tf.shape(a5)[0], 14, 14, 64]) + b1_deconv)#[-1,14,14,64]
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
        self.da1 = tf.nn.leaky_relu(tf.nn.conv2d(x, W1_conv, strides=[1, 2, 2, 1], padding='SAME') + b1_conv)#[-1,14,14,64]
        da2 = tf.nn.leaky_relu(tf.nn.conv2d(self.da1, W2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv)#[-1,7,7,128]
        da3=tf.layers.batch_normalization(da2)
        da4= tf.reshape(da3, [-1, 7 * 7 * 64])
        da5 = tf.nn.leaky_relu(tf.matmul(da4, W1) + b1)
        da6 = tf.matmul(da5, W2) + b2
        logits = da6
        return logits

    def get_wganp_loss(self):
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        self.D_loss=-tf.reduce_mean(self.logits_real)+tf.reduce_mean(self.logits_fake)
        self.G_loss=-tf.reduce_mean(self.logits_fake)
        self.D_clip=[p.assign(tf.clip_by_value(p,-0.01,0.01)) for p in D_vars]



    def get_optimizer(self,mode="RMSprop"):
        if mode =="Adam":
            self.G_solver=tf.train.AdamOptimizer(learning_rate=self.HParams.learning_rate,beta1=self.HParams.beta1)
            self.D_solver=tf.train.AdamOptimizer(learning_rate=self.HParams.learning_rate,beta1=self.HParams.beta1)
        elif mode =="RMSprop":
            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.HParams.learning_rate)
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.HParams.learning_rate)


    def get_train_op(self):
        D_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
        G_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')
        self.D_train_step=self.D_solver.minimize(self.D_loss,var_list=D_vars)
        self.G_train_step=self.G_solver.minimize(self.G_loss,var_list=G_vars)

    def train_Discriminator(self,sess,minibatch):
        _,d_loss_curr,_=sess.run([self.D_train_step,self.D_loss,self.D_clip],feed_dict={self.Input_img:minibatch})
        return d_loss_curr

    def train_Generator(self,sess,minibatch):
        _,g_loss_curr=sess.run([self.G_train_step,self.G_loss],feed_dict={self.Input_img:minibatch})
        return g_loss_curr

    def get_fake_image(self,sess):
        img=sess.run(self.output_fake_img)
        return img
