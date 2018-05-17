#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10


def parse_record(serialized_example):
  """Parsing CIFAR-10 dataset that is saved in TFRecord format."""
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    })

  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([IMAGE_DEPTH * IMAGE_HEIGHT * IMAGE_WIDTH])
  image = tf.reshape(image, [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
  image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

  label = tf.cast(features['label'], tf.int32)
  label = tf.one_hot(label, NUM_CLASSES)

  return image, label


def preprocess_image(image, is_training=False):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, IMAGE_HEIGHT + 8, IMAGE_WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def generate_input_fn(filenames, mode=tf.estimator.ModeKeys.EVAL, batch_size=1):
  """Input function for Estimator API."""
  def _input_fn():
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if is_training:
      buffer_size = batch_size * 2 + 1
      dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training), label))

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)

    images, labels = dataset.make_one_shot_iterator().get_next()

    features = {'images': images}
    return features, labels

  return _input_fn


def get_feature_columns():
  """Define feature columns."""
  feature_columns = {
    'images': tf.feature_column.numeric_column(
        'images', (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)),
  }
  return feature_columns


def serving_input_fn():
  """Define serving function."""
  receiver_tensor = {
      'images': tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
  }
  features = {
      'images': tf.map_fn(preprocess_image, receiver_tensor['images'])
  }
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


def inference(images):
  # 1st Convolutional Layer
  conv1 = tf.layers.conv2d(
      inputs=images, filters=64, kernel_size=[5, 5], padding='same',
      activation=tf.nn.relu, name='conv1')
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1, pool_size=[3, 3], strides=2, name='pool1')
  norm1 = tf.nn.lrn(
      pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

  # 2nd Convolutional Layer
  conv2 = tf.layers.conv2d(
      inputs=norm1, filters=64, kernel_size=[5, 5], padding='same',
      activation=tf.nn.relu, name='conv2')
  norm2 = tf.nn.lrn(
      conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  pool2 = tf.layers.max_pooling2d(
      inputs=norm2, pool_size=[3, 3], strides=2, name='pool2')

  # Flatten Layer
  shape = pool2.get_shape()
  pool2_ = tf.reshape(pool2, [-1, shape[1]*shape[2]*shape[3]])

  # 1st Fully Connected Layer
  dense1 = tf.layers.dense(
      inputs=pool2_, units=384, activation=tf.nn.relu, name='dense1')

  # 2nd Fully Connected Layer
  dense2 = tf.layers.dense(
      inputs=dense1, units=192, activation=tf.nn.relu, name='dense2')

  # 3rd Fully Connected Layer (Logits)
  logits = tf.layers.dense(
      inputs=dense2, units=NUM_CLASSES, activation=tf.nn.relu, name='logits')

  return logits


def model_fn(features, labels, mode, params):
  # Create the input layers from the features
  feature_columns = list(get_feature_columns().values())

  images = tf.feature_column.input_layer(
    features=features, feature_columns=feature_columns)

  images = tf.reshape(
    images, shape=(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))

  # Calculate logits through CNN
  logits = inference(images)

  if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')

  if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
    global_step = tf.train.get_or_create_global_step()
    label_indices = tf.argmax(input=labels, axis=1)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)
    tf.summary.scalar('cross_entropy', loss)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': predicted_indices,
        'probabilities': probabilities
    }
    export_outputs = {
        'predictions': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, export_outputs=export_outputs)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)
