from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from data_process.cifar10_loader import load_training_data
import viewpics
from viewpics import image_manifold_size

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# shared category size
CATGORY_INFO = 128
BATCH_SIZE = 128
IMG_SIZE = [-1, 32, 32, 3]
CKPT_STEP=10000
NUM_GENERATED_BATCHES=500

checkpoint_dir='./checkpoints/hgan/cifar10'
sample_dir = './samples/layergan/cifar10'
sample_dir2= './samples/layergan/cifar10R'
LAYERS = 3


# A bunch of utility functions


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([32, 32, 3]))
    return


def preprocess_img(x):
     return 2 * x - 1.0


def denormalize(images):
  images= (images+1)*127.5
  images=images.astype(np.uint8)
  return images

def damage_img(x):
    shp = 10
    x = tf.reshape(x, [-1, 32, 32, 3])
    noise = np.random.normal(0, 0.1, BATCH_SIZE * shp * shp * 3)
    noise = noise.reshape([-1, shp, shp, 3])
    xstart = np.random.randint(0, 32 - shp - 1)
    ystart = np.random.randint(0, 32 - shp - 1)
    print(xstart, ystart)
    blank = np.zeros(BATCH_SIZE * 32 * 32 * 3).reshape([-1, 32, 32, 3])
    part = blank[:, xstart:xstart + shp, ystart:ystart + shp, :]
    print(blank.shape, part.shape)
    part += noise
    return x + blank


def deprocess_img(x):
    return (x + 1.0) / 2.0


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./datasets/mnist/', one_hot=False)


# show a batch
# show_images(mnist.train.next_batch(16)[0])


def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.

    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU

    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    return tf.maximum(x, alpha * x)


def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.

    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the the noise to generate

    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    # TODO: sample and return noise
    return np.random.uniform(size=[batch_size, dim], low=-1, high=1)


def wganp_loss(logits_real, logits_fake):
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    D_loss = tf.reduce_mean(logits_real) - tf.reduce_mean(logits_fake)
    G_loss = -tf.reduce_mean(logits_fake)

    return D_loss, G_loss, D_vars


def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.

    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image

    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    # TODO: compute D_loss and G_loss
    D_loss = None
    G_loss = None

    labels_real = tf.ones_like(logits_real)
    labels_fake = tf.zeros_like(logits_fake)
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))

    labels_fake = tf.ones_like(logits_fake)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))

    return D_loss, G_loss


def share_cat_loss(dis_cat_layer, gen_cat_layer):
    """Compute the gen hidden loss given the gen loss
    :param should_gen_layer: Tensor, shape [batch_size, ???], output of discriminator
    :param gen_layer: Tensor, shape[batch_size, ???]
    :return:
    """
    return tf.reduce_mean(tf.square(tf.reduce_mean(dis_cat_layer, axis=0) - tf.reduce_mean(gen_cat_layer, axis=0)))


def image_loss(true_image, false_image):
    true_image = tf.reshape(true_image, [-1, 32, 32, 3])
    false_image = tf.reshape(false_image, [-1, 32, 32, 3])
    return tf.reduce_sum(tf.square(true_image - false_image))


def get_optimizer(name, optimizer='adam', learning_rate=1, beta1=0.0, beta2=0.9):
    decay = 1.0
    lr = learning_rate * decay
    # tf.summary.scalar(name+'+learning_rate',lr)
    if optimizer == 'adam':
        return tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr, decay=0.99)
    elif optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    else:
        raise NotImplementedError


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.

    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)

    Returns:
    - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
    """
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    C_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    R_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    # R_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    return D_solver, G_solver, C_solver, R_solver


def encode_layer(input_image, layer_num):
    with tf.variable_scope("encode-" + str(layer_num), reuse=tf.AUTO_REUSE):
        a2 = tf.nn.avg_pool(input_image, ksize=[1, LAYERS-layer_num, LAYERS-layer_num, 1], strides=[1, 1, 1, 1],
                            padding='SAME')  # 8, 8, 32
        return a2


def decode_layer(input_image, layer_num):
    with tf.variable_scope("decode-" + str(layer_num), reuse=tf.AUTO_REUSE):
        w1_filter = tf.get_variable("w1_filter", shape=[4, 4, IMG_SIZE[3], 32])
        b1_filter = tf.get_variable("b1_filter", shape=[IMG_SIZE[3]])
        a1 = tf.layers.batch_normalization(
            tf.nn.conv2d_transpose(input_image, w1_filter, strides=[1, 4, 4, 1], padding='SAME',
                                   output_shape=[BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[2], IMG_SIZE[3]]) + b1_filter,
            training=True)
        a2 = (tf.nn.tanh(a1) + 1) / 2
        return a2


def encode(x):
    with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
        W1_conv = tf.get_variable('W1_conv', shape=[3, 3, 3, 32])
        b1_conv = tf.get_variable('b1_conv', shape=32)
        W2_conv = tf.get_variable('W2_conv', shape=[3, 3, 32, 64])
        b2_conv = tf.get_variable('b2_conv', shape=64)
        W3_conv = tf.get_variable('W3_conv', shape=[4, 4, 64, 64])
        b3_conv = tf.get_variable('b3_conv', shape=64)
        W4_conv = tf.get_variable('W4_conv', shape=[3, 3, 64, 64])
        b4_conv = tf.get_variable('b4_conv', shape=64)
        W5_conv = tf.get_variable('W5_conv', shape=[3, 3, 64, 64])
        b5_conv = tf.get_variable('b5_conv', shape=64)
        W6_conv = tf.get_variable('W6_conv', shape=[3, 3, 64, 64])
        b6_conv = tf.get_variable('b6_conv', shape=64)
        W1 = tf.get_variable('W1', shape=[8 * 8 * 64, CATGORY_INFO])
        b1 = tf.get_variable('b1', shape=CATGORY_INFO)
        W2 = tf.get_variable('W2', shape=[CATGORY_INFO, 1], trainable=False, initializer=tf.random_normal_initializer)
        b2 = tf.get_variable('b2', shape=1, trainable=False, initializer=tf.random_normal_initializer)
        X = tf.reshape(x, [-1, 32, 32, 3])
        a1 = leaky_relu(tf.nn.conv2d(X, W1_conv, strides=[1, 1, 1, 1], padding='SAME') + b1_conv)
        a2 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16, 16, 32
        print("a2", a2.shape)
        a3 = leaky_relu(tf.nn.conv2d(a2, W2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv)
        a4 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 8, 8, 64
        print("a4", a4.shape)
        a5 = leaky_relu(tf.nn.conv2d(a4, W3_conv, strides=[1, 1, 1, 1], padding='SAME') + b3_conv)  # -1,8,8,64
        a6 = leaky_relu(tf.nn.conv2d(a5, W4_conv, strides=[1, 1, 1, 1], padding='SAME') + b4_conv)
        # a6=leaky_relu(tf.nn.conv2d(a5,W4_conv,strides=[1,1,1,1],padding='SAME')+b4_conv)
        a7 = leaky_relu(tf.nn.conv2d(a6, W5_conv, strides=[1, 1, 1, 1], padding='SAME') + b5_conv)
        a8 = leaky_relu(tf.nn.conv2d(a7, W6_conv, strides=[1, 1, 1, 1], padding='SAME') + b6_conv)  # 8,8,64

        a5 = tf.reshape(a8, [-1, 4096])  # 8*8*64
        a6 = leaky_relu(tf.matmul(a5, W1) + b1)  # [-1,category_info]
        return a6


def discriminator(x, layer_num):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    # shared_category = tf.reshape(shared_category, [-1, CATGORY_INFO])
    with tf.variable_scope("discriminator-" + str(layer_num), reuse=tf.AUTO_REUSE):
        # TODO: implement architecture

        W1_conv = tf.get_variable('W1_conv', shape=[5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1_conv = tf.get_variable('b1_conv', shape=[32], initializer=tf.constant_initializer(0.0))
        W2_conv = tf.get_variable('W2_conv', shape=[5, 5, 32, 64],initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2_conv = tf.get_variable('b2_conv', shape=[64], initializer=tf.constant_initializer(0.0))
        W3_conv = tf.get_variable('W3_conv', shape=[5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3_conv = tf.get_variable('b3_conv', shape=[128],initializer=tf.constant_initializer(0.0))
        W4_conv = tf.get_variable('W4_conv', shape=[5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4_conv = tf.get_variable('b4_conv', shape=[256], initializer=tf.constant_initializer(0.0))
        W5_conv = tf.get_variable('W5_conv', shape=[5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))

        W1 = tf.get_variable('W1', shape=[2 * 2 * 256, 1])
        b1 = tf.get_variable('b1', shape=[1])
        W2 = tf.get_variable('W2', shape=[2048, 1], trainable=False, initializer=tf.random_normal_initializer)
        b2 = tf.get_variable('b2', shape=1, trainable=False, initializer=tf.random_normal_initializer)
        X = tf.reshape(x, [-1, 32, 32, 3])
        a1 = tf.nn.bias_add(tf.nn.conv2d(X, W1_conv, strides=[1, 2, 2, 1], padding='SAME'), b1_conv)
        a2 = leaky_relu(a1)
        a3 = tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.conv2d(a2, W2_conv, strides=[1, 2, 2, 1], padding='SAME'), b2_conv), training=True)
        a4 = leaky_relu(a3)
        a5 = tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.conv2d(a4, W3_conv, strides=[1, 2, 2, 1], padding='SAME'), b3_conv), training=True)  # -1,8,8,64
        a6 = leaky_relu(a5)
        a7 = tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.conv2d(a6, W4_conv, strides=[1, 2, 2, 1], padding='SAME'),  b4_conv), training=True)
        a7 = leaky_relu(a7)

        a5 = tf.reshape(a7, [BATCH_SIZE, 2*2*256])  # 8*8*64
        a6 = (tf.matmul(a5, W1))  # [-1,category_info]
        logits = a6

        return logits



#
# def decode(shared_category):
#     with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
#         # noise = tf.random_normal([tf.shape(shared_category)[0], 32*32*3], mean=0, stddev=0.1)
#         # mapping category to graph
#         W1 = tf.get_variable('W1', shape=[CATGORY_INFO, 8 * 8 * 128])
#         b1 = tf.get_variable('b1', shape=8 * 8 * 128)
#
#         #         s2 = tf.get_variable('s2', shape=[7*7*128]) # batch-norm scale parameter
#         #         o2 = tf.get_variable('o2', shape=[7*7*128]) # batch-norm offset parameter
#         W2_deconv = tf.get_variable('W2_deconv', shape=[4, 4, 64, 128])
#         b2_deconv = tf.get_variable('b2_deconv', shape=64)
#         W3_deconv = tf.get_variable('W3_deconv', shape=[4, 4, 64, 64])
#         b3_deconv = tf.get_variable('b3_deconv', shape=64)
#         #         s1_deconv = tf.get_variable('s1_deconv', shape=[14,14,64]) # batch-norm scale parameter
#         #         o1_deconv = tf.get_variable('o1_deconv', shape=[14,14,64]) # batch-norm offset parameter
#         W4_deconv = tf.get_variable('W4_deconv', shape=[4, 4, 64, 64])
#         b4_deconv = tf.get_variable('b4_deconv', shape=64)
#
#         W5_deconv = tf.get_variable('W5_deconv', shape=[4, 4, 64, 64])
#         b5_deconv = tf.get_variable('b5_deconv', shape=64)
#
#         W6_deconv = tf.get_variable('W6_deconv', shape=[4, 4, 64, 64])
#         b6_deconv = tf.get_variable('b6_deconv', shape=64)
#
#         W7_deconv = tf.get_variable('W7_deconv', shape=[4, 4, 3, 64])
#         b7_deconv = tf.get_variable('b7_deconv', shape=3)
#
#         #         m2, v2 = tf.nn.moments(a3, axes=[0], keep_dims=False) # mean and var for batch-norm
#         #         a4 = tf.nn.batch_normalization(a3, m2, v2, o2, s2, 1e-6)
#         a4 = tf.layers.batch_normalization(tf.matmul(shared_category, W1) + b1, training=True, name="share")
#
#         a5 = tf.reshape(a4, [-1, 8, 8, 128])
#         a6 = tf.layers.batch_normalization(tf.nn.conv2d_transpose(a5, W2_deconv, strides=[1, 2, 2, 1], padding='SAME',
#                                                output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b2_deconv, training=True)
#         #         mdc1, vdc1 = tf.nn.moments(a6, axes=[0], keep_dims=False) # mean and var for batch-norm
#         #         a7 = tf.nn.batch_normalization(a6, mdc1, vdc1, o1_deconv, s1_deconv, 1e-6)
#         a7 = leaky_relu(a6)
#         a8 = tf.layers.batch_normalization(tf.nn.conv2d_transpose(a7, W3_deconv, strides=[1, 1, 1, 1], padding='SAME',
#                                                output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b3_deconv, training=True)
#         #         mdc1, vdc1 = tf.nn.moments(a6, axes=[0], keep_dims=False) # mean and var for batch-norm
#         #         a7 = tf.nn.batch_normalization(a6, mdc1, vdc1, o1_deconv, s1_deconv, 1e-6)
#         a9 = leaky_relu(a8)
#
#         a10 = tf.layers.batch_normalization(tf.nn.conv2d_transpose(a9, W4_deconv, strides=[1, 1, 1, 1], padding='SAME',
#                                                 output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b4_deconv, training=True)
#         a11 = leaky_relu(a10)
#
#         a12 = tf.layers.batch_normalization(tf.nn.conv2d_transpose(a11, W5_deconv, strides=[1, 1, 1, 1], padding='SAME',
#                                                 output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b5_deconv, training=True)
#         a13 = leaky_relu(a12)
#
#         a14 = tf.layers.batch_normalization(tf.nn.conv2d_transpose(a13, W6_deconv, strides=[1, 1, 1, 1], padding='SAME',
#                                                 output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b6_deconv, training=True)
#         a15 = leaky_relu(a14, training=True)
#
#         a16 = tf.nn.tanh(tf.nn.conv2d_transpose(a15, W7_deconv, strides=[1, 2, 2, 1], padding='SAME',
#                                                 output_shape=[tf.shape(a7)[0], 32, 32, 3]) + b7_deconv)
#         a17 = tf.reshape(a16, [-1, 32 * 32 * 3])
#         # a10 = (tf.layers.batch_normalization(a9, training=True))
#         img = (a16 + 1) / 2
#         print('gen', shared_category.shape)
#         return img

def batch_mul(conf, x):
    old_shape = x.shape
    conf = tf.squeeze(conf)
    ret = tf.transpose(tf.reshape(x, [BATCH_SIZE, -1]))*conf
    return tf.reshape(tf.transpose(ret), old_shape)


def generate_final(img, mem, z, layer_num):
    with tf.variable_scope("generator-img-" + str(layer_num), reuse=tf.AUTO_REUSE):
        img_linear_size = IMG_SIZE[1] * IMG_SIZE[2] * IMG_SIZE[3]
    
        HIE = 2**3
        INNER_IMG_SIZE = IMG_SIZE[1]//8*IMG_SIZE[2]//8*INNER_DIM
        w_noise = tf.get_variable("W_noise", shape=[noise_dim, INNER_IMG_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_noise = tf.get_variable("b_noise", shape=INNER_IMG_SIZE,initializer=tf.truncated_normal_initializer(stddev=0.02))
        info_1 = tf.get_variable("w_info_1", shape=[5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        info_2 = tf.get_variable("w_info_2", shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        info = tf.get_variable("w_info", shape=[5, 5, 64, INNER_DIM], initializer=tf.truncated_normal_initializer(stddev=0.02))
        exp_filters = tf.get_variable("gener", shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        exp_f = tf.get_variable("bf", shape=[1], initializer=tf.truncated_normal_initializer(stddev=0.02))

        exp_filters_1 = tf.get_variable("gener_1", shape=[5, 5, 64, INNER_DIM],initializer=tf.truncated_normal_initializer(stddev=0.02))
        exp_f_1 = tf.get_variable("bf_1", shape=[64], initializer=tf.truncated_normal_initializer(stddev=0.02))

        exp_filters_2 = tf.get_variable("gener_2", shape=[5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        exp_f_2 = tf.get_variable("bf_2", shape=[32])

        refine = tf.get_variable("refine", shape=[5, 5, 128, INNER_DIM], initializer=tf.truncated_normal_initializer(stddev=0.02))
        refine_f = tf.get_variable("refine_f", shape=[128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        ref_filters = tf.get_variable("ref_fil", shape=[5, 5, 32, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        ref_f = tf.get_variable("ref_f", shape=[32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        ref_filters_2 = tf.get_variable("ref_fil_2", shape=[5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        ref_f_2 = tf.get_variable("ref_f_2", shape=[3], initializer=tf.truncated_normal_initializer(stddev=0.02))

        final_filter = tf.get_variable("final", shape=[5, 5, 3, 3], initializer=tf.truncated_normal_initializer(stddev=0.02))
        final_f_1 = tf.get_variable("final_f_2", shape=[3], initializer=tf.truncated_normal_initializer(stddev=0.02))

        final_filter_2 = tf.get_variable("final2", shape=[5, 5, 3, 3], initializer=tf.truncated_normal_initializer(stddev=0.02))
        final_f_2 = tf.get_variable("final_f_23", shape=[3], initializer=tf.truncated_normal_initializer(stddev=0.02))

        img_shape_size = [-1, IMG_SIZE[1], IMG_SIZE[2], IMG_SIZE[3]]
        x = tf.reshape(img, img_shape_size)

        noise_graph = tf.matmul(z, w_noise) + b_noise
        noise_graph = tf.reshape(noise_graph, [-1, IMG_SIZE[1]//8, IMG_SIZE[2]//8, INNER_DIM])

        # do the cnn
        query_1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(x, info_1, strides=[1, 2, 2, 1], padding='SAME'), training=True))  # 32 32 3
        query_2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(query_1, info_2, strides=[1, 2, 2, 1], padding='SAME'), training=True))  # 32 32 3
        query = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(query_2, info, strides=[1, 2, 2, 1], padding='SAME'), training=True))  # 32 32 3)
        print('query', mem.shape)
        conf = tf.layers.dense(tf.reshape(query, [BATCH_SIZE, -1]), units=1, activation=tf.nn.sigmoid)
        exp_1 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d_transpose(query, exp_filters_1, strides=[1, 2, 2, 1], padding='SAME',
                                   output_shape=[tf.shape(query)[0], IMG_SIZE[1]//4, IMG_SIZE[2]//4,
                                                 64], name="exp_convt_0"), training=True))
        exp_2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d_transpose(exp_1, exp_filters_2, strides=[1, 2, 2, 1], padding='SAME',
                                   output_shape=[tf.shape(query)[0], IMG_SIZE[1]//2, IMG_SIZE[2]//2,
                                                 32], name="exp_convt_1"), training=True ))
        expectation = tf.sigmoid(
            tf.nn.conv2d_transpose(exp_2, exp_filters, strides=[1, 2, 2, 1], padding='SAME',
                                   output_shape=[tf.shape(query)[0], IMG_SIZE[1], IMG_SIZE[2],
                                                 1], name="exp_convt_2"))
        # get mask
        if layer_num > 0:
            mask =expectation
            mask = tf.nn.max_pool(mask, [1,2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
        else:
            mask = 1

        # use noise to generate graph

       
        if layer_num == 0:
            half_noise_graph = noise_graph
        else:
            
            half_noise_graph =  noise_graph + query # batch_mul(conf, query) + batch_mul(1-conf, mem)

        # gen graph
        half_noise_graph = tf.nn.relu(tf.layers.batch_normalization(half_noise_graph, training=True))
        gen_info = tf.nn.relu(tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.conv2d_transpose(half_noise_graph, refine, strides=[1, 2, 2, 1], padding='SAME', output_shape=[tf.shape(noise_graph)[0], IMG_SIZE[1]//4, IMG_SIZE[2]//4,
                                                            128], name="conv2t_0"), refine_f), training=True)) # 32 32 3

        gen_pickture = tf.nn.relu(tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.conv2d_transpose(gen_info, ref_filters, strides=[1, 2, 2, 1], padding='SAME',
                                              output_shape=[tf.shape(gen_info)[0], IMG_SIZE[1]//2, IMG_SIZE[2]//2,
                                                            32], name="conv2dt_1"),ref_f), training=True))
        gen_pickture_2 = (tf.nn.bias_add(tf.nn.conv2d_transpose(gen_pickture, ref_filters_2, strides=[1, 2, 2, 1], padding='SAME',
                                              output_shape=[tf.shape(gen_info)[0], IMG_SIZE[1], IMG_SIZE[2],
                                                            IMG_SIZE[3]], name="conv2dt_2"), ref_f_2))


        reg_gen = (tf.nn.tanh(gen_pickture_2)+1)/2

        new_picture = (1 - mask) * img + mask * reg_gen

        if layer_num == 0:
            return new_picture, mem, mask

        # ref

        final_picture =  (1+ tf.tanh(tf.nn.bias_add(tf.nn.conv2d_transpose(new_picture, final_filter, strides=[1, 1, 1, 1], padding='SAME',
                                              output_shape=[tf.shape(gen_info)[0], IMG_SIZE[1], IMG_SIZE[2],
                                                            IMG_SIZE[3]], name="conv2dt_3"), final_f_1)))/2

        final_picture = (1 - mask) * img + mask * final_picture

        final_picture =  (1+ tf.tanh(tf.nn.bias_add(tf.nn.conv2d_transpose(final_picture, final_filter_2, strides=[1, 1, 1, 1], padding='SAME',
                                              output_shape=[tf.shape(gen_info)[0], IMG_SIZE[1], IMG_SIZE[2],
                                                            IMG_SIZE[3]], name="conv2dt_3"), final_f_2)))/2
        final_picture = (1 - mask) * img + mask * final_picture

        # update
        mem = batch_mul(conf, query) + batch_mul(1-conf, mem)
        return new_picture, mem, mask
#by miaomiao
def generate_pictures(sess):
    print('generating pics after training')
    init_assign_op,init_feed_dict=utils.restore_ckpt(train_dir,log)
    sess.run(init_assign_op,feed_dict=init_feed_dict)
    output_list=[]
    for i in range(NUM_GENERATED_BATCHES):
        pass




def run_a_gan(
        sess,
        refine_graph,
        G_sample,
        mems_in,
        # noisy_graph,
        # D_clip,
        D_train_step, D_loss,
        G_train_step, G_loss,
        # R_train_step, R_loss,
        # C_train_step, C_loss,
        # R_train_step, R_loss,
        # U_train_step, U_loss,
        show_every=500, print_every=50, batch_size=BATCH_SIZE, num_epoch=2):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    - L_loss: Learner loss mapping from discriminator to gen hidden layer
    - S_loss: inner hidden layer loss
    Returns:
        Nothing
    """

    # compute the number of iterations we need
    images, _, _ = load_training_data()
    np.random.shuffle(images)

    def next_batch(images):
        while True:
            for i in range(images.shape[0] // batch_size):
                yield images[i * batch_size: (i + 1) * batch_size]

    iter = next_batch(images)
    mems = []
    for _ in range(LAYERS):
        at = sample_noise(BATCH_SIZE, 4*4*32) 
        mems.append(np.reshape(at, [BATCH_SIZE, 4, 4, 32]))
    # show a batch
    TOTAL = 10000

    for it in range(100000):

        # every show often, show a sample result

        # run a batch of data through the network
        minibatch = next(iter)
        z_rand = sample_noise(BATCH_SIZE, noise_dim)

        # print('123', minibatch.shape)
        D_loss_cur, G_loss_cur = sess.run([D_loss, G_loss], feed_dict={x: minibatch, z: z_rand, ms: mems})
        _ = sess.run([D_train_step,G_train_step], feed_dict={x: minibatch, z: z_rand, ms: mems})
        
        
        # _, _, _, _ = sess.run([D_train_step, C_train_step, R_train_step, U_train_step], feed_dict={x: minibatch,z:z_rand})
        # _, D_loss_cur = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        # _, C_loss_cur = sess.run([C_train_step, C_loss], feed_dict={x: minibatch})
        # _, R_loss_curr = sess.run([R_train_step, R_loss], feed_dict={x: minibatch})
        # _, U_loss_curr = sess.run([U_train_step, U_loss], feed_dict={x: minibatch})
        # _, D_noise_loss_curr = sess.run([D_train_step2, D_noise_loss], feed_dict={x: minibatch})

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % print_every == 0:
            for num in range(LAYERS):
                print('Iter: {}, D:{:.4}, G:{:.4}'.format(it, D_loss_cur[num], G_loss_cur[num]))
        if it % show_every == 0:  # print('Final images')
            smp = next(iter)
            z_rand = sample_noise(BATCH_SIZE, noise_dim)
            # show_images(smp[:64])
            samples = sess.run(G_sample, feed_dict={z: z_rand, x: smp, ms: mems})
            # noise_image = sess.run(noisy_graph, feed_dict={x: smp, z:z_rand})
            refines_image = sess.run(refine_graph, feed_dict={x: smp, z: z_rand, ms: mems})
        
            for num in range(LAYERS):
                # show_images(noise_image[:64])
                #show_images(refines_image[num][:64])
                sx = (samples[num][:64])
                fig = show_images(sx)
                # viewpics.save_images(refines_image[num][:64], image_manifold_size(64),
                #                      '{}/refine{}_iteration_{}.png'.format(sample_dir, it, num))
                viewpics.save_images(sx, image_manifold_size(64),
                                     '{}/sample{}_iteration_{}.png'.format(sample_dir2, it, num))
        
            #plt.show()

        #modified by miao miao
        if it % CKPT_STEP ==0:
            print("checkpointing ..%s"%it)
            saver.save(sess,checkpoint_dir,global_step=it,write_meta_graph=False)

        mems = sess.run(mems_in, feed_dict={x: minibatch, z: z_rand, ms: mems})
        # print(np.sum(mems))
        
        


tf.reset_default_graph()

# our noise dimension
noise_dim = 1024
INNER_DIM = 32

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [BATCH_SIZE, 32, 32, 3])
z = tf.placeholder(tf.float32, [BATCH_SIZE, noise_dim])
ms = tf.placeholder(tf.float32, [LAYERS, BATCH_SIZE, 4,4, INNER_DIM])

def encode_pipeline(img, total_layer):
    outputs = []
    for num in range(total_layer):
        encoded = encode_layer(img, num)
        outputs.append(encoded)

    return outputs


def encode_decode_loss(decodes, real_image):
    losses = []
    for num in range(len(decodes)):
        loss = image_loss(decodes[num], real_image)
        losses.append(loss)
    return losses


def discriminator_pipeline(middle_real_image, real_image, middle_fake_image, fake_image):
    total_num = len(middle_real_image)
    real_logits, fake_logits = [], []
    for fake_img_num in range(total_num):
        rs = []
        fs = []
        for dis_num in range(total_num):
            r = discriminator(middle_real_image[fake_img_num], dis_num)
            f = discriminator(middle_fake_image[fake_img_num], dis_num)
            rs.append(r)
            fs.append(f)
        real_logits.append(rs)
        fake_logits.append(fs)
    # r, f = discriminator(real_image, fake_image)
    # real_logits.append(r)
    # fake_logits.append(f)
    return real_logits, fake_logits


def discriminator_generator_loss(logits_reals, logits_fakes, n_masks):
    c_losses = []
    g_losses = []
    for num in range(len(logits_fakes)):
        c_loss, g_loss = gan_loss(logits_reals[num][num], logits_fakes[num][num])
        c_losses.append(c_loss)
    # for num in range(len(logits_fake)):
    
    #     if num == 0:
    #         dis_layer_critics = logits_fake[num][num]
    #         labels_fake = tf.ones_like(dis_layer_critics)
    #         g_loss = tf.reduce_mean(
    #             tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=dis_layer_critics))
    #     else:
    #         dis_layer_critics =tf.nn.sigmoid(logits_fake[num][num]) - tf.nn.sigmoid(logits_fake[num - 1][num])
    #         g_loss = -tf.reduce_mean(dis_layer_critics)
        if num == 0:
            g_losses.append(g_loss)
        else:
            g_losses.append( g_loss+ 100* tf.reduce_mean(tf.square(n_masks[num]-n_masks[num-1])))
    return c_losses, g_losses


def generator_pipeline(z, ms, layer_num):
    fakes = []
    mems = []
    masks = []
    start_img = tf.ones(shape=[BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[2], IMG_SIZE[3]])
    print("start_img", start_img.shape)
    for num in range(layer_num):
        middle_result, mem, mask = generate_final(start_img, ms[num], z, num)
        fakes.append(middle_result)
        mems.append(mem)
        masks.append(mask)
        start_img = middle_result
    return fakes, mems, masks


def get_trainable_vars(name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)


def generator_train(gen_losses, solver):
    trains = []
    for num in range(len(gen_losses)):
        vars = []
        #for i in range(num+1):
        vars.extend(get_trainable_vars("generator-img-" + str(num)))
        train = solver.minimize(gen_losses[num], var_list=vars)
        trains.append(train)
    return trains


def discriminator_train(dis_losses, solver):
    trains = []

    for num in range(len(dis_losses)):
        train = solver.minimize(dis_losses[num], var_list=get_trainable_vars("discriminator-" + str(num)))
        trains.append(train)
    return trains


def refine_train(img_losses, solver):
    trains = []
    for num in range(len(img_losses)):
        train = solver.minimize(img_losses[num], var_list=get_trainable_vars("encode-" + str(num)) + get_trainable_vars(
            "decode-" + str(num)))
        trains.append(train)
    return trains




with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
    middle_images = encode_pipeline(x, LAYERS)
    # refine_image = generate_refinement(encoded_x)
    # scale images to be -1 to 1
    g_samples, n_mems, n_masks = generator_pipeline(z, ms, LAYERS)

    logits_real, logits_fake = discriminator_pipeline(middle_images, None, g_samples, None)
    ende_loss = encode_decode_loss(middle_images, x)
    D_losses, G_losses = discriminator_generator_loss(logits_real, logits_fake, g_samples)
    D_solver, U_solver, C_solver, R_solver = get_solvers()
    # ende_train_steps = refine_train(ende_loss, R_solver)
    dis_train_steps = discriminator_train(D_losses, D_solver)
    gen_trains_steps = generator_train(G_losses, U_solver)

    #added by chris
    saver=tf.train.Saver(max_to_keep=100,keep_checkpoint_every_n_hours=1)



with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(
        sess,
        middle_images,
        g_samples,
        n_mems,
        # D_clip,
        dis_train_steps, D_losses,
        gen_trains_steps, G_losses,
        # ende_train_steps, ende_loss,
        # D_train_step2, D_noise_loss,
        # C_train_step, C_loss,
        # R_train_step, R_loss,
        # U_train_step, U_loss
    )
