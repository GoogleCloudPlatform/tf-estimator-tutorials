from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

# Model specific parameters
tf.flags.DEFINE_string(
    "data_dir", default="gs://your-bucket-name/keras-mnist/data",
    help="Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string(
    "model_dir", default="gs://your-bucket-name/keras-mnist/keras-model",
    help="Estimator model_dir")
tf.flags.DEFINE_integer(
    "batch_size", default=100,
    help="Mini-batch size for the training. Note that this is the global batch "
    "size and not the per-shard batch.")
tf.flags.DEFINE_float(
    "learning_rate", default=0.005,
    help="Learning rate used to optimize your model.")
tf.flags.DEFINE_integer(
    "train_steps", default=1000,
    help="Total number of training steps.")
tf.flags.DEFINE_string(
    "export_dir", default="gs://your-bucket-name/keras-mnist/keras-model/export",
    help="The directory where the exported SavedModel will be stored.")

FLAGS = tf.flags.FLAGS


def get_estimator():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=[28*28]))
  model.add(tf.keras.layers.Dense(300, activation='relu'))
  model.add(tf.keras.layers.Dense(100, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.SGD(lr=0.005),
                metrics=['accuracy'])
  estimator = tf.keras.estimator.model_to_estimator(
    model, model_dir=FLAGS.model_dir)

  input_signature = model.input.name.split(':')[0]
  return estimator, input_signature

def get_serving_input_fn(input_signature):
  def preprocess(x):
    return tf.reshape(x, [-1, 28*28]) / 255.0

  def serving_input_fn():
    receiver_tensor = {'X': tf.placeholder(tf.float32, shape=[None, 28, 28])}
    features = {input_signature: tf.map_fn(preprocess, receiver_tensor['X'])}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
  return serving_input_fn

def generate_input_fn(file_pattern, mode, batch_size, count=None):
  def parse_record(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),            
        })
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [28*28]) / 255.0
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [])
    label = tf.one_hot(label, 10, dtype=tf.int32)
    return image, label

  def input_fn():
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.cache()
      dataset = dataset.shuffle(10000)
      dataset = dataset.repeat(count=count)
      
    dataset = dataset.map(parse_record)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels
  
  return input_fn

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  estimator, input_signature = get_estimator()
  train_input_fn = generate_input_fn(
    file_pattern='{}/train.tfrecord'.format(FLAGS.data_dir),
    mode=tf.estimator.ModeKeys.TRAIN,
    batch_size=FLAGS.batch_size, count=None)

  eval_input_fn = generate_input_fn(
    file_pattern='{}/eval.tfrecord'.format(FLAGS.data_dir),
    mode=tf.estimator.ModeKeys.EVAL,
    batch_size=FLAGS.batch_size, count=None)

  test_input_fn = generate_input_fn(
    file_pattern='{}/test.tfrecord'.format(FLAGS.data_dir),
    mode=tf.estimator.ModeKeys.PREDICT,
    batch_size=FLAGS.batch_size, count=None)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

  exporter = tf.estimator.LatestExporter(
    name='export',
    serving_input_receiver_fn=get_serving_input_fn(input_signature))

  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn,
    steps=None,
    start_delay_secs=60,
    throttle_secs=60,
    exporters=exporter)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  tf.app.run()
