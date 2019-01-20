from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from data_process.cifar10_loader import load_training_data

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# shared category size
CATGORY_INFO = 5 * 5 * 64
BATCH_SIZE = 128


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
    return x # return 2 * x - 1.0


def damage_img(x):
    shp = 1
    noise = np.random.normal(0, 1, BATCH_SIZE* shp*shp*3)
    noise = noise.reshape([-1, shp, shp, 3])
    xstart = np.random.randint(0, 32-shp-1)
    ystart = np.random.randint(0, 32-shp-1)
    print(xstart, ystart)
    blank = np.zeros(BATCH_SIZE*32*32*3).reshape([-1, 32, 32, 3])
    part = blank[:, xstart:xstart+shp, ystart:ystart+shp, :]  
    print(blank.shape,  part.shape)
    part +=  noise 
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

def wganp_loss(logits_real,logits_fake):
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    D_loss=tf.reduce_mean(logits_real)-tf.reduce_mean(logits_fake)
    G_loss=-tf.reduce_mean(logits_fake)
    
    return D_loss,G_loss,D_clip


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

def get_optimizer(name,optimizer='adam',learning_rate=1,beta1=0.0,beta2=0.9):
    decay = 1.0
    lr=learning_rate*decay
    #tf.summary.scalar(name+'+learning_rate',lr)
    if optimizer=='adam':
        return tf.train.AdamOptimizer(lr,beta1=beta1,beta2=beta2)
    elif optimizer=='rmsprop':
        return tf.train.RMSPropOptimizer(lr,decay=0.99)
    elif optimizer=='sgd':
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


def discriminator(x):
    """Compute discriminator score for a batch of input images.

    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]

    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score
    for an image being real for each input image.
    """
    with tf.variable_scope("discriminator1", reuse=tf.AUTO_REUSE):
        # TODO: implement architecture

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
        W2 = tf.get_variable('W2', shape=[CATGORY_INFO, 1], trainable=True, initializer=tf.random_normal_initializer)
        b2 = tf.get_variable('b2', shape=1, trainable=True, initializer=tf.random_normal_initializer)
        X = tf.reshape(x, [-1, 32, 32, 3])
        a1 = leaky_relu(tf.nn.conv2d(X, W1_conv, strides=[1, 1, 1, 1], padding='SAME') + b1_conv)
        a2 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 16, 16, 32
        print("a2", a2.shape)
        a3 = leaky_relu(tf.nn.conv2d(a2, W2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv) 
        a4 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 8, 8, 64
        print("a4", a4.shape) 
        a5 = leaky_relu(tf.nn.conv2d(a4, W3_conv, strides=[1, 1, 1, 1], padding='SAME') + b3_conv)#-1,8,8,64
        #a4x = tf.nn.max_pool(a3x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME') # 5, 5, 64
        #a3xx = leaky_relu(tf.nn.conv2d(a4x, W4_conv, strides=[1, 1, 1, 1], padding='VALID') + b4_conv)
        #a4xx = tf.nn.max_pool(a3xx, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        a6=leaky_relu(tf.nn.conv2d(a5,W4_conv,strides=[1,1,1,1],padding='SAME')+b4_conv)
        a7 = leaky_relu(tf.nn.conv2d(a6, W5_conv, strides=[1, 1, 1, 1], padding='SAME') + b5_conv)
        a8 = leaky_relu(tf.nn.conv2d(a7, W6_conv, strides=[1, 1, 1, 1], padding='SAME') + b6_conv)

        a5 = tf.reshape(a8, [-1, CATGORY_INFO]) # 8*8*64
        a6 = leaky_relu(tf.matmul(a5, W1) + b1)
        a7 = tf.matmul(a6, W2) + b2
        logits = a7
        print('dis', a6.shape)
        return logits, a6


def generator(z):
    """Generate images from a random noise vector.

    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]

    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    shared = generate_category(z)
    cust = generate_customized(shared)
    return generate_refinement(cust)


def generate_category(z):
    with tf.variable_scope("generator1-cat", reuse=tf.AUTO_REUSE):
        # TODO: implement architecture
        W1 = tf.get_variable('W1', shape=[z.get_shape()[1], 1024])
        b1 = tf.get_variable('b1', shape=1024)
        #         s1 = tf.get_variable('s1', shape=[1024]) # batch-norm scale parameter
        #         o1 = tf.get_variable('o1', shape=[1024]) # batch-norm offset parameter
        W2 = tf.get_variable('W2', shape=[1024, CATGORY_INFO])
        b2 = tf.get_variable('b2', shape=CATGORY_INFO)

    a1 = tf.nn.relu(tf.matmul(z, W1) + b1)
        #         m1, v1 = tf.nn.moments(a1, axes=[0], keep_dims=False) # mean and var for batch-norm
        #         a2 = tf.nn.batch_normalization(a1, m1, v1, o1, s1, 1e-6)
    a2 = tf.layers.batch_normalization(a1, training=True)
    a3 = tf.nn.relu(tf.matmul(a2, W2) + b2)
    return a3


def generate_refinement(shared_category):
    with tf.variable_scope("generator1-ref", reuse=tf.AUTO_REUSE):
        noise =  tf.random_normal([tf.shape(shared_category)[0], 32*32*3], mean=0, stddev=0.1)
        # mapping category to graph
        W1 = tf.get_variable('W1', shape=[CATGORY_INFO, 8 * 8 * 128])
        b1 = tf.get_variable('b1', shape=8 * 8 * 128)

        #         s2 = tf.get_variable('s2', shape=[7*7*128]) # batch-norm scale parameter
        #         o2 = tf.get_variable('o2', shape=[7*7*128]) # batch-norm offset parameter
        W2_deconv = tf.get_variable('W2_deconv', shape=[4, 4, 64, 128])
        b2_deconv = tf.get_variable('b2_deconv', shape=64)
        W3_deconv = tf.get_variable('W3_deconv', shape=[4, 4, 64, 64])
        b3_deconv = tf.get_variable('b3_deconv', shape=64)
        #         s1_deconv = tf.get_variable('s1_deconv', shape=[14,14,64]) # batch-norm scale parameter
        #         o1_deconv = tf.get_variable('o1_deconv', shape=[14,14,64]) # batch-norm offset parameter
        W4_deconv = tf.get_variable('W4_deconv', shape=[4, 4, 64, 64])
        b4_deconv = tf.get_variable('b4_deconv', shape=64)

        W5_deconv = tf.get_variable('W5_deconv', shape=[4, 4, 64, 64])
        b5_deconv = tf.get_variable('b5_deconv', shape=64)

        W6_deconv = tf.get_variable('W6_deconv', shape=[4, 4, 64, 64])
        b6_deconv = tf.get_variable('b6_deconv', shape=64)

        W7_deconv = tf.get_variable('W7_deconv', shape=[4, 4, 3, 64])
        b7_deconv = tf.get_variable('b7_deconv', shape=3)


        #         m2, v2 = tf.nn.moments(a3, axes=[0], keep_dims=False) # mean and var for batch-norm
        #         a4 = tf.nn.batch_normalization(a3, m2, v2, o2, s2, 1e-6)
        a4 = tf.layers.batch_normalization(tf.matmul(shared_category, W1) + b1, training=True, name="share")

        a5 = tf.reshape(a4, [-1, 8, 8, 128])
        a6 = tf.nn.relu(tf.nn.conv2d_transpose(a5, W2_deconv, strides=[1, 2, 2, 1], padding='SAME',
                                               output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b2_deconv)
        #         mdc1, vdc1 = tf.nn.moments(a6, axes=[0], keep_dims=False) # mean and var for batch-norm
        #         a7 = tf.nn.batch_normalization(a6, mdc1, vdc1, o1_deconv, s1_deconv, 1e-6)
        a7 = tf.layers.batch_normalization(a6, training=True)
        a8 = tf.nn.relu(tf.nn.conv2d_transpose(a7, W3_deconv, strides=[1, 1, 1, 1], padding='SAME',
                                               output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b3_deconv)
        #         mdc1, vdc1 = tf.nn.moments(a6, axes=[0], keep_dims=False) # mean and var for batch-norm
        #         a7 = tf.nn.batch_normalization(a6, mdc1, vdc1, o1_deconv, s1_deconv, 1e-6)
        a9 = tf.layers.batch_normalization(a8, training=True)

        a10 = tf.nn.relu(tf.nn.conv2d_transpose(a9, W4_deconv, strides=[1, 1, 1, 1], padding='SAME',
                                               output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b4_deconv)
        a11 = tf.layers.batch_normalization(a10, training=True)

        a12 = tf.nn.relu(tf.nn.conv2d_transpose(a11, W5_deconv, strides=[1, 1, 1, 1], padding='SAME',
                                                output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b5_deconv)
        a13 = tf.layers.batch_normalization(a12, training=True)

        a14 = tf.nn.relu(tf.nn.conv2d_transpose(a13, W6_deconv, strides=[1, 1, 1, 1], padding='SAME',
                                                output_shape=[tf.shape(a5)[0], 16, 16, 64]) + b6_deconv)
        a15 = tf.layers.batch_normalization(a14, training=True)


        a16 = tf.nn.tanh(tf.nn.conv2d_transpose(a15, W7_deconv, strides=[1, 2, 2, 1], padding='SAME',
                                               output_shape=[tf.shape(a7)[0], 32, 32, 3]) + b7_deconv)
        a17 = tf.reshape(a16, [-1, 32 * 32 * 3])
        # a10 = (tf.layers.batch_normalization(a9, training=True))
        img = a17
        print('gen', shared_category.shape)
        return img + noise


def generate_customized(shared_category):
    with tf.variable_scope("generator1-cust", reuse=tf.AUTO_REUSE):
        W1 = tf.get_variable('W1', shape=[CATGORY_INFO, CATGORY_INFO])
        b1 = tf.get_variable('b1', shape=CATGORY_INFO)
        W2 = tf.get_variable('W2', shape=[CATGORY_INFO, CATGORY_INFO])
        b2 = tf.get_variable('b2', shape=CATGORY_INFO)

        h1 = tf.nn.leaky_relu(tf.matmul(shared_category, W1) + b1)
        return tf.nn.leaky_relu(tf.matmul(h1, W2) + b2)


def run_a_gan(
        sess,
        refine_graph,
        noisy_graph,
        D_clip,
        D_train_step, D_loss,
        D_train_step2, D_noise_loss,
        C_train_step, C_loss,
        R_train_step, R_loss,
        U_train_step, U_loss,
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

        # show a batch
        
        for it in range(10000):

            # every show often, show a sample result

            # run a batch of data through the network
            minibatch = next(iter)
            z_rand = sample_noise(BATCH_SIZE, noise_dim)
            
            # print('123', minibatch.shape)
            D_loss_cur, C_loss_cur, R_loss_cur, U_loss_cur, D_noise_loss_cur = sess.run([D_loss, C_loss, R_loss, U_loss, D_noise_loss], feed_dict={x: minibatch, z:z_rand}) 
            _ = sess.run([D_clip, D_train_step, D_train_step2], feed_dict={x: minibatch,z:z_rand}) 
            #_, _, _, _ = sess.run([D_train_step, C_train_step, R_train_step, U_train_step], feed_dict={x: minibatch,z:z_rand})
            #_, D_loss_cur = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
            #_, C_loss_cur = sess.run([C_train_step, C_loss], feed_dict={x: minibatch})
            #_, R_loss_curr = sess.run([R_train_step, R_loss], feed_dict={x: minibatch})
            #_, U_loss_curr = sess.run([U_train_step, U_loss], feed_dict={x: minibatch})
            #_, D_noise_loss_curr = sess.run([D_train_step2, D_noise_loss], feed_dict={x: minibatch})


            # print loss every so often.
            # We want to make sure D_loss doesn't go to 0
            if it % print_every == 0:
                print('Iter: {}, D:{:.4}, G:{:.4}, C:{:.4} R:{:.4}'.format(it,
                                                                                    D_loss_cur,
                                                                                    #D_noise_loss_cur,
                                                                                    U_loss_cur,
                                                                                    C_loss_cur,
                                                                                    R_loss_cur))
            if it % show_every == 0:    #print('Final images')
                smp = next(iter)
                z_rand = sample_noise(BATCH_SIZE, noise_dim)
                #show_images(smp)
                samples = sess.run(G_sample, feed_dict={z: z_rand})
                noise_image = sess.run(noisy_graph, feed_dict={x: smp, z:z_rand})
                refines_image = sess.run(refine_graph, feed_dict={x: smp, z: z_rand})
                show_images(noise_image[:64])
                show_images(refines_image[:64])
                #fig = show_images(samples[:64])
                plt.show()


tf.reset_default_graph()


# our noise dimension
noise_dim = 1280

# placeholders for images from the training dataset
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
z = tf.placeholder(tf.float32, [None, noise_dim])

with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
    # scale images to be -1 to 1
    logits_real, dis_cat_layer_real = discriminator(preprocess_img(x))

    # use noise not to let the dis downgrade
    noisy_graph = damage_img(preprocess_img(x))
    

    # generate category
    category_fake = generate_category(z)

    # generate refinement
    refine_image = generate_refinement(dis_cat_layer_real)

    # generated images
    G_sample = generator(z)

    # Re-use discriminator weights on new inputs
    logits_fake_little, _ = discriminator(noisy_graph) 
    #scope.reuse_variables()
    logits_refine_fake, _ = discriminator(refine_image)
    #scope.reuse_variables()
    logits_fake, _ = discriminator(G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator1')
C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator1-cat')
U_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator1-cust')
R_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator1-ref')
D_solver, U_solver, C_solver, R_solver = get_solvers()
D_clip=[p.assign(tf.clip_by_value(p,-0.01,0.01)) for p in D_vars]

# U_loss is customized loss
_, U_loss = gan_loss(logits_real, logits_fake)
D_loss, R_loss = gan_loss(logits_real, logits_refine_fake)
D_noise_loss, _ = gan_loss(logits_fake_little, logits_refine_fake)

# category loss measures distribution difference
C_loss = share_cat_loss(dis_cat_layer_real, category_fake)
#_, R_loss = gan_loss(logits_real, logits_refine_fake)

# dis variables
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)

D_train_step2 = D_solver.minimize(D_noise_loss, var_list=D_vars)

# customized layer is responsible for total loss
U_train_step = U_solver.minimize(U_loss, var_list=U_vars)

# category layer
C_train_step = C_solver.minimize(C_loss, var_list=C_vars)

# refinement layer
R_train_step = R_solver.minimize(R_loss, var_list=R_vars)

with get_session() as sess:
    sess.run(tf.global_variables_initializer())
    run_a_gan(
        sess,
        refine_image,
        noisy_graph,
        D_clip,
        D_train_step, D_loss,
        D_train_step2, D_noise_loss,
        C_train_step, C_loss,
        R_train_step, R_loss,
        U_train_step, U_loss
    )
