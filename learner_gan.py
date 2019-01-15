from config_gan import args,get_logging_config
import sys
sys.path.append("/Users/chris/PycharmProjects/ShareGan")
from paths import DATASETS,CKPT_ROOT
import logging.config
import datetime
import time
import os
import tensorflow as tf
import utils.general_utils as utils
import numpy as np
import logging
from easydict import EasyDict as edict
from data_process import matcher
from model import origin_mlp_gan
import data_config as data_cfg
from evaluation import inception_score
logging.config.dictConfig(get_logging_config(args.model_name))
log=logging.getLogger("gan")
data_config=data_cfg.get_config(args.dataset)
def sample_noise():
    return tf.random_uniform([args.batch_size, args.noise_dim], minval=-1, maxval=1)

def get_optimizer(name,optimizer=args.optimizer):
    if args.lr_decay:
        global_step=tf.train.get_global_step()
        decay= tf.maximum(0., 1.-(tf.maximum(0., tf.cast(global_step, tf.float32) -
                                              args.linear_decay_start))/args.max_iterations)
    else:
        decay = 1.0
    lr=args.learning_rate*decay
    #tf.summary.scalar(name+'+learning_rate',lr)
    if optimizer=='adam':
        return tf.train.AdamOptimizer(lr,beta1=args.adam_beta1,beta2=args.adam_beta2)
    elif optimizer=='rmsprop':
        return tf.train.RMSPropOptimizer(lr,decay=0.99)
    elif optimizer=='sgd':
        return tf.train.GradientDescentOptimizer(lr)
    else:
        raise NotImplementedError

def get_model():
    model=None
    if args.model_name == 'mlp':
        model = origin_mlp_gan
    elif args.model_name == 'sgan':
        pass
    return model

def construct_model():
    model=get_model()
    real_images,iter_fn=matcher.load_dataset('train',args.batch_size,args.dataset,32)
    datacfg=data_cfg.get_config(args.dataset)
    model.construct_variables(args.noise_dim,datacfg)
    input_noise=sample_noise()
    fake_img=model.get_generator(input_noise,datacfg)

    logits_real=model.get_discriminator(real_images,datacfg)
    logits_fake=model.get_discriminator(fake_img,datacfg)

    optimizer=get_optimizer(name="",optimizer=args.optimizer)
    D_loss,G_Loss=model.get_loss(logits_real,logits_fake)
    Discriminator_train_op=optimizer.minimize(D_loss)
    Generator_train_op=optimizer.minimize(G_Loss)
    return Discriminator_train_op,Generator_train_op,iter_fn,D_loss,G_Loss

def train(train_dir):
    generator_train_steps=1
    discriminator_train_step=1
    D_train_op,G_train_op,iter_fun,D_loss,G_loss=construct_model()
    global_step=tf.train.get_or_create_global_step()
    init_assign_op,init_feed_dict=utils.restore_ckpt(train_dir,log)
    summary_op=tf.summary.merge_all()
    clean_init_op=tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    increment_global_step_op= tf.assign_add(global_step, 1)
    saver=tf.train.Saver(max_to_keep=100,keep_checkpoint_every_n_hours=1)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        summary_writer=tf.summary.FileWriter(train_dir)
        sess.run(clean_init_op)
        sess.run(init_assign_op,feed_dict=init_feed_dict)
        iter_fun(sess)

        starting_step=sess.run(global_step)
        starting_time=time.time()
        log.info("Starting training from step %i..."%starting_step)
        for step in range(starting_step,args.max_iterations+1):
            start_time=time.time()
            try:
                gen_loss=0
                for _ in range(generator_train_steps):
                    _,cur_gen_loss=sess.run([G_train_op,G_loss])
                    gen_loss+=cur_gen_loss

                dis_loss=0
                for _ in range(discriminator_train_step):
                    _,cur_dis_loss=sess.run([D_train_op,D_loss])
                    dis_loss+=cur_dis_loss
                sess.run(increment_global_step_op)
            except (tf.errors.OutOfRangeError,tf.errors.CancelledError):
                break
            except KeyboardInterrupt:
                log.info("Killed by C")
                break

            if step % args.print_step ==0:
                duration=float(time.time()-start_time)
                examples_per_batch=args.batch_size / duration
                log.info("step %i: gen_loss %f, dis_loss %f (%.1f examples/sec; %.3f sec/batch)"
                         % (step,gen_loss,dis_loss,examples_per_batch,duration))
                avg_speed=(time.time()-starting_time)/(step-starting_step+1)
                time_to_finish=avg_speed*(args.max_iterations-step)
                end_date=datetime.datetime.now()+datetime.timedelta(seconds=time_to_finish)
                log.info("%i iterations left,expected to finsh at %s (avg speed: %.3f sec/batch)"
                         %(args.max_iterations-step,end_date.strftime("Y-%m-%d %H:%M:%S"),avg_speed))

            if step % args.summary_step ==0:
                summary=tf.Summary()
                pass

            if step % args.ckpt_step ==0 and step>=0:
                log.debug("Saving checkpoint ...")
                checkpoint_path=os.path.join(train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step,write_meta_graph=False)



def gengerate(train_dir,suffix=''):

    datacfg = data_cfg.get_config(args.dataset)
    model=get_model()
    log.info("Generating %i batches using suffix %s"%(args.num_generated_batches,suffix))
    init_assign_op,init_feed_dict=utils.restore_ckpt(train_dir,log)
    noise=sample_noise()
    model.construct_variables(noise_dim=args.noise_dim,data_cfg=datacfg)
    fake_images=model.get_generator(noise,datacfg)
    clean_init_op = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
    #saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        sess.run(clean_init_op)
        #saver.restore(sess,train_dir)
        sess.run(init_assign_op,feed_dict=init_feed_dict)
        output_list=[]
        for i in range(args.num_generated_batches):
            output_list.append(sess.run(fake_images))

    np.save(os.path.join(DATASETS,args.dataset,"X_gan_%s.npy"%(args.model_name+suffix)),output_list)


def get_inception_score():
    datacfg=data_cfg.get_config(args.dataset)
    generated_pics=np.load(os.path.join(DATASETS,args.dataset,'X_gan_%s.npy'%args.model_name))
    generated_pics=np.reshape(generated_pics,[-1,datacfg.dataset.image_size,datacfg.dataset.image_size,datacfg.dataset.channels])
    mean,var=inception_score.get_inception_score(generated_pics)
    log.info('the inception score is %s,with d=standard deviation %s'%(mean,var))

def get_fid_score():
    pass

if __name__ =='__main__':
    train_dir=CKPT_ROOT+args.model_name
    log.info('start action')
    for action in args.actions.split(','):
        tf.reset_default_graph()
        if action == 'train_gan':
            print("starting training at %s"%train_dir)
            train(train_dir)

        elif action == 'generate':

            gengerate(train_dir)
        elif action == "inception_score":
            get_inception_score()

        elif action =='fid distance':
            get_fid_score()
        else:
            print('Action is not known')
            quit(1)

# <<<<<<< HEAD:gan.py
# =======
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import random
# >>>>>>> 44691f811e3cab1beffd9cebeef740f76ba85262:learner_gan.py
#
#
#
#
#
# <<<<<<< HEAD:gan.py
# =======
#     for i, img in enumerate(images):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(img.reshape([sqrtimg,sqrtimg]))
#     return
#
# def preprocess_img(x):
#     return 2 * x - 1.0
#
# def deprocess_img(x):
#     return (x + 1.0) / 2.0
#
# def rel_error(x,y):
#     return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#
# def count_params():
#     """Count the number of parameters in the current TensorFlow graph """
#     param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
#     return param_count
#
#
# def get_session():
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.Session(config=config)
#     return session
#
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./data/', one_hot=False)
#
# # show a batch
# show_images(mnist.train.next_batch(16)[0])
#
#
# def leaky_relu(x, alpha=0.01):
#     """Compute the leaky ReLU activation function.
#
#     Inputs:
#     - x: TensorFlow Tensor with arbitrary shape
#     - alpha: leak parameter for leaky ReLU
#
#     Returns:
#     TensorFlow Tensor with the same shape as x
#     """
#     # TODO: implement leaky ReLU
#     return tf.maximum(x, alpha * x)
#
#
# def sample_noise(batch_size, dim):
#     """Generate random uniform noise from -1 to 1.
#
#     Inputs:
#     - batch_size: integer giving the batch size of noise to generate
#     - dim: integer giving the dimension of the the noise to generate
#
#     Returns:
#     TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
#     """
#     # TODO: sample and return noise
#     return tf.random_uniform([batch_size, dim], minval=-1, maxval=1)
#
#
# def gan_loss(logits_real, logits_fake):
#     """Compute the GAN loss.
#
#     Inputs:
#     - logits_real: Tensor, shape [batch_size, 1], output of discriminator
#         Log probability that the image is real for each real image
#     - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
#         Log probability that the image is real for each fake image
#
#     Returns:
#     - D_loss: discriminator loss scalar
#     - G_loss: generator loss scalar
#     """
#     # TODO: compute D_loss and G_loss
#     D_loss = None
#     G_loss = None
#
#     labels_real = tf.ones_like(logits_real)
#     labels_fake = tf.zeros_like(logits_fake)
#     D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real))
#     D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))
#
#     labels_fake = tf.ones_like(logits_fake)
#     G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))
#
#     return D_loss, G_loss
#
# def share_gen_loss(should_gen_layer, gen_layer):
#     """Compute the gen hidden loss given the gen loss
#     :param should_gen_layer: Tensor, shape [batch_size, ???], output of discriminator
#     :param gen_layer: Tensor, shape[batch_size, ???]
#     :return:
#     """
#     return tf.reduce_sum(tf.square(should_gen_layer - gen_layer))
#
#
# def learner_loss(should_output, real_output):
#     # should output is the mapping dis to gen
#     # real output should be gen real output
#     # have the shape of (None, 28, 28, 2)
#     return tf.reduce_sum(tf.square(should_output - real_output))
#
#
# def get_solvers(learning_rate=1e-3, beta1=0.5):
#     """Create solvers for GAN training.
#
#     Inputs:
#     - learning_rate: learning rate to use for both solvers
#     - beta1: beta1 parameter for both solvers (first moment decay)
#
#     Returns:
#     - D_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
#     - G_solver: instance of tf.train.AdamOptimizer with correct learning_rate and beta1
#     """
#     D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
#     G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
#     L_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
#     S_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
#     return D_solver, G_solver, L_solver, S_solver
#
#
# def discriminator(x):
#     """Compute discriminator score for a batch of input images.
#
#     Inputs:
#     - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
#
#     Returns:
#     TensorFlow Tensor with shape [batch_size, 1], containing the score
#     for an image being real for each input image.
#     """
#     with tf.variable_scope("discriminator1", reuse=tf.AUTO_REUSE):
#         # TODO: implement architecture
#
#         W1_conv = tf.get_variable('W1_conv', shape=[5, 5, 1, 32])
#         b1_conv = tf.get_variable('b1_conv', shape=32)
#         W2_conv = tf.get_variable('W2_conv', shape=[5, 5, 32, 64])
#         b2_conv = tf.get_variable('b2_conv', shape=64)
#         W1 = tf.get_variable('W1', shape=[4 * 4 * 64, 4 * 4 * 64])
#         b1 = tf.get_variable('b1', shape=4 * 4 * 64)
#         W2 = tf.get_variable('W2', shape=[4 * 4 * 64, 1])
#         b2 = tf.get_variable('b2', shape=1)
#         X = tf.reshape(x, [-1, 28, 28, 1])
#         a1 = leaky_relu(tf.nn.conv2d(X, W1_conv, strides=[1, 1, 1, 1], padding='VALID') + b1_conv)
#         a2 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="share")
#
#         a3 = leaky_relu(tf.nn.conv2d(a2, W2_conv, strides=[1, 1, 1, 1], padding='VALID') + b2_conv)
#         a4 = tf.nn.max_pool(a3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#         a5 = tf.reshape(a4, [-1, 4 * 4 * 64])
#         a6 = leaky_relu(tf.matmul(a5, W1) + b1)
#         a7 = tf.matmul(a6, W2) + b2
#         logits = a7
#
#         return logits, a3
#
#
# def generator(z):
#     """Generate images from a random noise vector.
#
#     Inputs:
#     - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
#
#     Returns:
#     TensorFlow Tensor of generated images, with shape [batch_size, 784].
#     """
#     with tf.variable_scope("generator1"):
#         # TODO: implement architecture
#
#         W1 = tf.get_variable('W1', shape=[z.get_shape()[1], 1024])
#         b1 = tf.get_variable('b1', shape=1024)
#         #         s1 = tf.get_variable('s1', shape=[1024]) # batch-norm scale parameter
#         #         o1 = tf.get_variable('o1', shape=[1024]) # batch-norm offset parameter
#         W2 = tf.get_variable('W2', shape=[1024, 7 * 7 * 128])
#         b2 = tf.get_variable('b2', shape=7 * 7 * 128)
#         #         s2 = tf.get_variable('s2', shape=[7*7*128]) # batch-norm scale parameter
#         #         o2 = tf.get_variable('o2', shape=[7*7*128]) # batch-norm offset parameter
#         W1_deconv = tf.get_variable('W1_deconv', shape=[4, 4, 64, 128])
#         b1_deconv = tf.get_variable('b1_deconv', shape=64)
#         #         s1_deconv = tf.get_variable('s1_deconv', shape=[14,14,64]) # batch-norm scale parameter
#         #         o1_deconv = tf.get_variable('o1_deconv', shape=[14,14,64]) # batch-norm offset parameter
#         W2_deconv = tf.get_variable('W2_deconv', shape=[4, 4, 1, 64])
#         b2_deconv = tf.get_variable('b2_deconv', shape=1)
#
#         a1 = tf.nn.relu(tf.matmul(z, W1) + b1)
#         #         m1, v1 = tf.nn.moments(a1, axes=[0], keep_dims=False) # mean and var for batch-norm
#         #         a2 = tf.nn.batch_normalization(a1, m1, v1, o1, s1, 1e-6)
#         a2 = tf.layers.batch_normalization(a1, training=True)
#         a3 = tf.nn.relu(tf.matmul(a2, W2) + b2)
#         #         m2, v2 = tf.nn.moments(a3, axes=[0], keep_dims=False) # mean and var for batch-norm
#         #         a4 = tf.nn.batch_normalization(a3, m2, v2, o2, s2, 1e-6)
#         a4 = tf.layers.batch_normalization(a3, training=True, name="share")
#         a5 = tf.reshape(a4, [-1, 7, 7, 128])
#         a6 = tf.nn.relu(tf.nn.conv2d_transpose(a5, W1_deconv, strides=[1, 2, 2, 1], padding='SAME',
#                                                output_shape=[tf.shape(a5)[0], 14, 14, 64]) + b1_deconv)
#         #         mdc1, vdc1 = tf.nn.moments(a6, axes=[0], keep_dims=False) # mean and var for batch-norm
#         #         a7 = tf.nn.batch_normalization(a6, mdc1, vdc1, o1_deconv, s1_deconv, 1e-6)
#         a7 = tf.layers.batch_normalization(a6, training=True)
#         a8 = tf.nn.tanh(tf.nn.conv2d_transpose(a7, W2_deconv, strides=[1, 2, 2, 1], padding='SAME',
#                                                output_shape=[tf.shape(a7)[0], 28, 28, 1]) + b2_deconv)
#         a9 = tf.reshape(a8, [-1, 28 * 28])
#         img = a9
#         return img, a4
#
#
# def learner(hidden):
#     # hidden layer of dis
#     # mapping from None, 8, 8, 64 -> 6272
#     with tf.variable_scope("learner1", reuse=tf.AUTO_REUSE):
#         reg = tf.reshape(hidden, [-1, 4096])
#         h1 = tf.layers.dense(reg, units=128, activation=tf.sigmoid)
#         return tf.layers.dense(h1, units=6272, activation=tf.nn.leaky_relu)
#
# def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss,  L_loss, S_loss, L_train_step, S_train_step,
#               show_every=250, print_every=50, batch_size=128, num_epoch=2):
#     """Train a GAN for a certain number of epochs.
#
#     Inputs:
#     - sess: A tf.Session that we want to use to run our data
#     - G_train_step: A training step for the Generator
#     - G_loss: Generator loss
#     - D_train_step: A training step for the Generator
#     - D_loss: Discriminator loss
#     - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
#     - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
#     - L_loss: Learner loss mapping from discriminator to gen hidden layer
#     - S_loss: inner hidden layer loss
#     Returns:
#         Nothing
#     """
#     # compute the number of iterations we need
#     max_iter = int(mnist.train.num_examples * num_epoch / batch_size)
#     for it in range(5000):
#         # every show often, show a sample result
#
#         # run a batch of data through the network
#         minibatch, minbatch_y = mnist.train.next_batch(batch_size)
#         _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
#         _, G_loss_curr = sess.run([G_train_step, G_loss])
#
#
#
#         # generator have smaller loss
#         L_loss_cur, S_loss_cur = -1.0, -1.0
#
#         # if gan loss is small we tried to learn the mapping
#         if (random.random()*5 > G_loss_curr ):
#             _, L_loss_cur = sess.run([L_train_step, L_loss], feed_dict={x: minibatch})
#
#         else:
#             # other wise let learner tell us how to generate
#             _, S_loss_cur = sess.run([S_train_step, S_loss], feed_dict={x: minibatch})
#
#         # print loss every so often.
#         # We want to make sure D_loss doesn't go to 0
#         if it % print_every == 0:
#             print('Iter: {}, D: {:.4}, G:{:.4}, L:{:.4}, S:{:.4}'.format(it, D_loss_curr, G_loss_curr, L_loss_cur, S_loss_cur))
#     print('Final images')
#     samples = sess.run(G_sample)
#
#     fig = show_images(samples[:64])
#     plt.show()
#
#
# tf.reset_default_graph()
#
# batch_size = 128
# # our noise dimension
# noise_dim = 96
#
# # placeholders for images from the training dataset
# x = tf.placeholder(tf.float32, [None, 784])
# z = sample_noise(batch_size, noise_dim)
# # generated images
# G_sample, real_gen_hidden_layer = generator(z)
#
# with tf.variable_scope("") as scope:
#     # scale images to be -1 to 1
#     logits_real, dis_hidden_layer_real = discriminator(preprocess_img(x))
#     # Re-use discriminator weights on new inputs
#     scope.reuse_variables()
#     logits_fake, dis_hidden_layer_fake = discriminator(G_sample)
#
#     # get output of the learner
#     approx_gen_hidden = learner(dis_hidden_layer_fake)
#
#
# # Get the list of variables for the discriminator and generator
# D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator1')
# G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator1')
# L_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "learner1")
# D_solver, G_solver, L_solver, S_solver = get_solvers()
# D_loss, G_loss = gan_loss(logits_real, logits_fake)
# L_loss = learner_loss(approx_gen_hidden, real_gen_hidden_layer)
# S_loss = share_gen_loss(approx_gen_hidden, real_gen_hidden_layer)
# D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
# G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
# L_train_step = L_solver.minimize(L_loss, var_list=L_vars)
# S_train_step = S_solver.minimize(S_loss, var_list=G_vars)
#
#
# with get_session() as sess:
#     sess.run(tf.global_variables_initializer())
#     run_a_gan(sess,G_train_step, G_loss, D_train_step, D_loss, L_loss, S_loss, L_train_step, S_train_step)
# >>>>>>> 44691f811e3cab1beffd9cebeef740f76ba85262:learner_gan.py
