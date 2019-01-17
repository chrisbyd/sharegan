import tensorflow as tf
import numpy as np

def construct_variables(noise_dim,data_cfg):
    with tf.variable_scope("discriminator"):
        tf.get_variable("W1_conv", shape=[4, 4, data_cfg.dataset.channels, 64])
        tf.get_variable("b1_conv", shape=64)
        tf.get_variable("W2_conv", shape=[4, 4, 64, 128])
        tf.get_variable("b2_conv", shape=128)
        tf.get_variable("W1", shape=[8 * 8 * 64,8 * 8 * 64])
        tf.get_variable("b1", shape=8 * 8 * 64)
        tf.get_variable("W2", shape=[8 * 8 * 64, 1])
        tf.get_variable("b2", shape=1)

    with tf.variable_scope("generator"):
        tf.get_variable("W1", shape=[noise_dim, 1024])
        tf.get_variable("b1", shape=1024)
        tf.get_variable("W2", shape=[1024, 8* 8 * 128])
        tf.get_variable("b2", shape=8 * 8 * 128)
        tf.get_variable("W1_deconv", shape=[4, 4, 64, 128])
        tf.get_variable("b1_deconv", shape=64)
        tf.get_variable("W2_deconv", shape=[4, 4, data_cfg.dataset.channels, 64])
        tf.get_variable("b2_deconv", shape=data_cfg.dataset.channels)

def get_generator(input_noise,data_cfg):
    with tf.variable_scope("generator", reuse=True):
        W1 = tf.get_variable("W1")
        b1 = tf.get_variable("b1")
        W2 = tf.get_variable("W2")
        b2 = tf.get_variable("b2")
        W1_deconv = tf.get_variable("W1_deconv")
        b1_deconv = tf.get_variable("b1_deconv")
        W2_deconv = tf.get_variable("W2_deconv")
        b2_deconv = tf.get_variable("b2_deconv")
    a1 = tf.nn.relu(tf.matmul(input_noise, W1) + b1)
    a2 = tf.layers.batch_normalization(a1, training=True)
    a3 = tf.nn.relu(tf.matmul(a2, W2) + b2)
    a4 = tf.layers.batch_normalization(a3, training=True)
    a5 = tf.reshape(a4, [-1, 8, 8, 128])
    a6 = tf.nn.relu(tf.nn.conv2d_transpose(a5, W1_deconv, strides=[1, 2, 2, 1], padding='SAME',
                                           output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b1_deconv)  # [-1,14,14,64]
    a7 = tf.layers.batch_normalization(a6, training=True)
    print(data_cfg.dataset.image_size)
    a8 = tf.nn.tanh(tf.nn.conv2d_transpose(a7, W2_deconv, strides=[1, 2, 2, 1], padding='SAME',
                                           output_shape=[tf.shape(a7)[0],
                                                         data_cfg.dataset.image_size,
                                                         data_cfg.dataset.image_size,
                                                         data_cfg.dataset.channels])+b2_deconv)
    a9 = tf.reshape(a8, [-1, data_cfg.dataset.image_size*data_cfg.dataset.image_size*data_cfg.dataset.channels])
    fake_img = a9
    return fake_img


def get_discriminator(img,data_cfg):
    with tf.variable_scope("discriminator",reuse=True):
        W1_conv = tf.get_variable("W1_conv")
        b1_conv = tf.get_variable("b1_conv")
        W2_conv = tf.get_variable("W2_conv")
        b2_conv = tf.get_variable("b2_conv")
        W1 = tf.get_variable("W1")
        b1 = tf.get_variable("b1")
        W2 = tf.get_variable("W2")
        b2 = tf.get_variable("b2")
    x=tf.reshape(img,[-1,32,32,3])
    da1 = tf.nn.leaky_relu(tf.nn.conv2d(x, W1_conv, strides=[1, 1, 1, 1], padding='SAME') + b1_conv)
    da2 = tf.nn.max_pool(da1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#[-1,14,14,64]
    da3 = tf.nn.leaky_relu(tf.nn.conv2d(da2, W2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv)#
    da4 = tf.nn.max_pool(da3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#[-1,7,7,128]
    #print(self.da4.get_shape())
    da5 = tf.reshape(da4, [-1, 8 * 8 * 64])
    da6 = tf.nn.leaky_relu(tf.matmul(da5, W1) + b1)
    da7 = tf.matmul(da6, W2) + b2
    logits = da7
    return logits

def get_loss(logits_real,logits_fake):
    labels_real = tf.ones_like(logits_real)
    labels_fake = tf.zeros_like(logits_fake)
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))
    labels_fake = tf.ones_like(logits_fake)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))
    return D_loss,G_loss
