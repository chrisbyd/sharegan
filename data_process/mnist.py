import tensorflow as tf
import os
from paths import DATASETS

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        "pixels": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Parse the string into an array of pixels corresponding to the image
    images = tf.decode_raw(parsed["image_raw"],tf.uint8)
    images = tf.reshape(images,[28,28,1])
    images=tf.cast(images,tf.float32)
    labels = tf.cast(parsed['label'], tf.int32)
    labels = tf.one_hot(labels,10)
    pixels = tf.cast(parsed['pixels'], tf.int32)
    return images, labels

                                                                                       
def load_dataset(split, batch_size, normalize=True, dataset='mnist',
                 augmentation=False, shuffle=True, prefetch_batches=10,
                 dequantize=True, classes=None):
    
    dataset_folder = os.path.join(DATASETS, dataset,'output.tfrecords')
    # ugly hack to allow support of multiple splits
    # this pattern though looks very generic
    dataset=tf.data.TFRecordDataset(dataset_folder)
    dataset=dataset.map(parser)
    dataset=dataset.shuffle(buffer_size=10000)
    dataset=dataset.batch(batch_size)
    dataset=dataset.repeat(5)
    iterator=dataset.make_one_shot_iterator()
    imgs,labels=iterator.get_next()

    def iterator_fn(sess):
        pass 

    return imgs, labels, iterator_fn
