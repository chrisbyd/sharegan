import tensorflow as tf
a=tf.constant([1])
logits=tf.constant([3.0])
loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=[1.0],logits=[3.0])
b=tf.nn.sigmoid(logits)

with tf.Session() as sess:
    los,b_=sess.run([loss,b])
    print(los)
    print(b_)