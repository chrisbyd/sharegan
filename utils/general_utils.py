import tensorflow as tf
slim=tf.contrib.slim
def restore_ckpt(ckpt_dir, log, ckpt_number=0, global_step=None):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if ckpt_number == 0:
            ckpt_to_restore = ckpt.model_checkpoint_path
        else:
            ckpt_to_restore = ckpt_dir+'/model.ckpt-%i' % ckpt_number
        variables_to_restore=tf.global_variables()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            ckpt_to_restore, variables_to_restore)
        log.info("Restore weights from %s", ckpt_to_restore)
    else:
        init_assign_op = tf.no_op()
        init_feed_dict = None
        log.info("This network is trained from scratch")

    return init_assign_op, init_feed_dict
