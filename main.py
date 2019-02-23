from config_gan import args,get_logging_config
import sys
from paths import DATASETS,CKPT_ROOT,SAMPLE
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
from model import origin_dcgan
import data_config as data_cfg
from evaluation import inception_score
from evaluation import frachet_inception_distance
import viewpics
from viewpics import image_manifold_size
from model import deep_dcgan
from model import dcgan_standard

sample_dir=os.path.join('./samples',args.model_name,args.dataset)
logging.config.dictConfig(get_logging_config(args.model_name))
log=logging.getLogger("gan")
data_config=data_cfg.get_config(args.dataset)
def sample_noise():
    return tf.random_uniform([args.batch_size, args.noise_dim], minval=-1, maxval=1)

def sample_generate_noise():
  return tf.random_uniform([100,args.noise_dim],minval=-1, maxval=1)

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
    if args.model_name == 'dcgan':
        model = dcgan_standard
    elif args.model_name == 'ddcgan':
        model = deep_dcgan
    return model

def construct_model():
    model=get_model()
    real_images,iter_fn=matcher.load_dataset('train',args.batch_size,args.dataset,32)
    datacfg=data_cfg.get_config(args.dataset)
    input_noise=sample_noise()
    fake_img=model.generator(input_noise, reuse=False)

    logits_real=model.discriminator(real_images,reuse=False)
    logits_fake=model.discriminator(fake_img,reuse=True)

    optimizer=get_optimizer(name="",optimizer=args.optimizer)
    D_loss,G_Loss=model.get_loss(logits_real,logits_fake)
    D_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
    G_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')
    Discriminator_train_op=optimizer.minimize(D_loss,var_list=D_vars)
    Generator_train_op=optimizer.minimize(G_Loss,var_list=G_vars)
    return Discriminator_train_op,Generator_train_op,iter_fn,D_loss,G_Loss,fake_img

def train(train_dir):
    generator_train_steps=1
    discriminator_train_step=1
    D_train_op,G_train_op,iter_fun,D_loss,G_loss,fake_img=construct_model()
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
                    _,cur_gen_loss,generated_img=sess.run([G_train_op,G_loss,fake_img])
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
            if step %args.view_pic_step ==0:
                #viewpics.view_pics(args,False,generated_img,step)
                viewpics.save_images(generated_img,
                                     image_manifold_size(generated_img.shape[0]),
                                     '{}/train_{:04d}.png'.format(sample_dir,step))

            if step % args.summary_step ==0:
                summary=tf.Summary()
                pass

            if step % args.ckpt_step ==0 and step>=0:
                log.debug("Saving checkpoint ...")
                checkpoint_path=os.path.join(train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step,write_meta_graph=False)

def denormalize(images):
  images= (images+1)*127.5
  images=images.astype(np.uint8)
  return images

def gengerate(train_dir,suffix=''):
    datacfg = data_cfg.get_config(args.dataset)
    model=get_model()
    log.info("Generating %i batches using suffix %s"%(args.num_generated_batches,suffix))
    noise=sample_generate_noise()
    fake_images=model.sampler(noise,False)
    init_assign_op,init_feed_dict=utils.restore_ckpt(train_dir,log)
    # clean_init_op = tf.group(tf.global_variables_initializer(),
    #                          tf.local_variables_initializer())
    #saver = tf.train.Saver()
    tf.get_default_graph().finalize()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
       # sess.run(clean_init_op)
        #saver.restore(sess,train_dir)
        sess.run(init_assign_op,feed_dict=init_feed_dict)
        output_list=[]
        for i in range(args.num_generated_batches):
            output_list.append(sess.run(fake_images)) 
        output=np.concatenate(output_list,axis=0)
    output=denormalize(output)
    np.save(os.path.join(SAMPLE,args.model_name,args.dataset,"X_gan_%s.npy"%(args.model_name+suffix)),output)


def get_inception_score():
    datacfg=data_cfg.get_config(args.dataset)
    generated_pics=np.load(os.path.join(SAMPLE,args.model_name,args.dataset,'X_gan_%s.npy'%args.model_name))
    #generated_pics=np.reshape(generated_pics,[-1,datacfg.dataset.image_size,datacfg.dataset.image_size,datacfg.dataset.channels])
    mean,var=inception_score.get_inception_score(generated_pics)
    log.info('the inception score is %s,with d=standard deviation %s'%(mean,var))

def get_fid_score():
   generated_pics=np.load(os.path.join(SAMPLE,args.model_name,args.dataset,'X_gan_%s.npy'%args.model_name))
   generated_pics=generated_pics.transpose(0,3,1,2)
   real_data_pics=np.load(os.path.join(DATASETS,args.dataset,'X_train.npy'))
   real_data_pics=real_data_pics.transpose(0,3,1,2)
   fid=frachet_inception_distance.get_fid(generated_pics,real_data_pics)
   log.info('the frachet inception distance  is %s'%fid)

    

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

        elif action =='fid_distance':
            get_fid_score()
        else:
            print('Action is not known')
            quit(1)
