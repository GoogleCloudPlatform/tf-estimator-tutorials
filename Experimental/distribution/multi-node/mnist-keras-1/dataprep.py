from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string(
    "output_dir", default=None,
    help="The output directory to use for storing input data for TPU "
    "training job.")

FLAGS = tf.flags.FLAGS

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_example(image, label):
  feature={
      "image": _bytes_feature(image.tobytes()),
      "label": _bytes_feature(label.tobytes())}
  return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecord(images, labels, output_file):
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for image, label in zip(images, labels):
      example = create_example(image, label)
      record_writer.write(example.SerializeToString())

def main(argv):
  non_test, test = tf.keras.datasets.mnist.load_data()
  X_train, y_train = non_test[0][:-5000], non_test[1][:-5000]
  X_eval, y_eval = non_test[0][-5000:], non_test[1][:-5000]
  X_test, y_test = test[0], test[1]

  convert_to_tfrecord(X_train, y_train,
                      "{}/train.tfrecord".format(FLAGS.output_dir))
  convert_to_tfrecord(X_eval, y_eval,
                      "{}/eval.tfrecord".format(FLAGS.output_dir))
  convert_to_tfrecord(X_test, y_test,
                      "{}/test.tfrecord".format(FLAGS.output_dir))

if __name__ == "__main__":
  tf.app.run()

