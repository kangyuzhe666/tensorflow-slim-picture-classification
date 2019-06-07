from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from nets import inception_v3
from nets import nets_factory
from preprocessing import preprocessing_factory
import cv2
import numpy as np
import os

test_image_size = 299
batch_size = 1

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'model.ckpt-44432',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'test_list', '', 'Test image list.')

tf.app.flags.DEFINE_string(
    'test_dir', '.', 'Test image directory.')

FLAGS = tf.app.flags.FLAGS


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        network_fn = nets_factory.get_network_fn(
            "inception_v3",
            num_classes=101,
            is_training=False)

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            "inception_v3",
            is_training=False)


        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
        logits, _ = network_fn(tensor_input)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        L = []
        for root, dirs, files in os.walk('/home/grid/tensorflow/20190407/test'):
            for file in files:
                # if os.path.splitext(file)[1] == '.jpeg':
                L.append(os.path.join(file))

        print(L)
        print(len(L))
        test_ids = L
        print(test_ids)
        tot = len(L)
        results = list()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            time_start = time.time()
            for idx in range(0, tot, batch_size):
                images = list()
                idx_end = min(tot, idx + batch_size)
                for i in range(idx, idx_end):
                    image_id = test_ids[i]
                    test_path = os.path.join('/home/grid/tensorflow/20190407/test', image_id)
                    image = open(test_path, 'rb').read()
                    print(test_path)
                    image = tf.image.decode_jpeg(image, channels=3)
                    processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
                    processed_image = sess.run(processed_image)
                    images.append(processed_image)
                images = np.array(images)
                predictions = sess.run(logits, feed_dict={tensor_input: images})
                max_index = np.argmax(predictions)
                print(max_index)


                for i in range(idx, idx_end):
                    print('{} {}'.format(image_id, predictions[i - idx].tolist()))
            time_total = time.time() - time_start
            print('total time: {}, total images: {}, average time: {}'.format(
                time_total, len(test_ids), time_total / len(test_ids)))


if __name__ == '__main__':
    tf.app.run()