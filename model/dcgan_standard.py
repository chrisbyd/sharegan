import tensorflow as tf
from model.ops import *
import tensorflow as tf

batch_size=64
gf_dim=32
df_dim=32
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def generator(z,reuse=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = 32, 32
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        z_, h0_w, h0_b = linear(
            z, gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        h0 = tf.reshape(
            z_, [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(tf.layers.batch_normalization(h0,training=True))

        h1, h1_w, h1_b = deconv2d(
            h0, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(tf.layers.batch_normalization(h1,training=True))

        h2, h2_w, h2_b = deconv2d(
            h1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training=True))

        h3, h3_w, h3_b = deconv2d(
            h2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(tf.layers.batch_normalization(h3,training=True))

        h4, h4_w, h4_b = deconv2d(
            h3, [batch_size, s_h, s_w, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

def sampler(z,reuse=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = 32, 32
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        z_, h0_w, h0_b = linear(
            z, gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        h0 = tf.reshape(
            z_, [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(tf.layers.batch_normalization(h0,training=True))

        h1, h1_w, h1_b = deconv2d(
            h0, [100, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(tf.layers.batch_normalization(h1,training=True))

        h2, h2_w, h2_b = deconv2d(
            h1, [100, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(tf.layers.batch_normalization(h2,training=True))

        h3, h3_w, h3_b = deconv2d(
            h2, [100, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(tf.layers.batch_normalization(h3,training=True))

        h4, h4_w, h4_b = deconv2d(
            h3, [100, s_h, s_w, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
     

def discriminator(image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
        h1 = lrelu(tf.layers.batch_normalization(conv2d(h0, df_dim*2, name='d_h1_conv'),training=True))
        h2 = lrelu(tf.layers.batch_normalization(conv2d(h1, df_dim*4, name='d_h2_conv'),training=True))
        h3 = lrelu(tf.layers.batch_normalization(conv2d(h2, df_dim*8, name='d_h3_conv'),training=True))
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4

def get_loss(logits_real,logits_fake):
    labels_real = tf.ones_like(logits_real)
    labels_fake = tf.zeros_like(logits_fake)
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real))
    D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))
    labels_fake = tf.ones_like(logits_fake)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_fake, logits=logits_fake))
    return D_loss,G_loss

      