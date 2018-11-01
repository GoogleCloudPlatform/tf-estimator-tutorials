# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Extract from notebook for Serving Optimization on Keras """

from __future__ import print_function

from datetime import datetime
import os
import sh
import sys
import tensorflow as tf
from tensorflow import data
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph

from inference_test import inference_test, load_mnist_keras
from optimize_graph import (run_experiment, get_graph_def_from_saved_model,
    describe_graph, get_size, get_metagraph, get_graph_def_from_file,
    convert_graph_def_to_saved_model, freeze_model, optimize_graph, TRANSFORMS)

NUM_CLASSES = 10
MODELS_LOCATION = 'models/mnist'
MODEL_NAME = 'keras_classifier'


def keras_model_fn(params):

  inputs = tf.keras.layers.Input(shape=(28, 28), name='input_image')
  input_layer = tf.keras.layers.Reshape(target_shape=(28, 28, 1), name='reshape')(inputs)

  # convolutional layers
  conv_inputs = input_layer
  for i in range(params.num_conv_layers):
    filters = params.init_filters * (2**i)
    conv = tf.keras.layers.Conv2D(kernel_size=3, filters=filters, strides=1, padding='SAME', activation='relu')(conv_inputs)
    max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')(conv)
    batch_norm = tf.keras.layers.BatchNormalization()(max_pool)
    conv_inputs = batch_norm

  flatten = tf.keras.layers.Flatten(name='flatten')(conv_inputs)

  # fully-connected layers
  dense_inputs = flatten
  for i in range(len(params.hidden_units)):
    dense = tf.keras.layers.Dense(units=params.hidden_units[i], activation='relu')(dense_inputs)
    dropout = tf.keras.layers.Dropout(params.dropout)(dense)
    dense_inputs = dropout

  # softmax classifier
  logits = tf.keras.layers.Dense(units=NUM_CLASSES, name='logits')(dense_inputs)
  softmax = tf.keras.layers.Activation('softmax', name='softmax')(logits)

  # keras model
  model = tf.keras.models.Model(inputs, softmax)
  return model


def create_estimator_keras(params, run_config):

  keras_model = keras_model_fn(params)
  print(keras_model.summary())

  optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate)
  keras_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  mnist_classifier = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model,
      config=run_config
  )

  return mnist_classifier


#### Train and Export Model

def train_and_export_model(train_data, train_labels):
  model_dir = os.path.join(MODELS_LOCATION, MODEL_NAME)

  hparams  = tf.contrib.training.HParams(
      batch_size=100,
      hidden_units=[512, 512],
      num_conv_layers=3,
      init_filters=64,
      dropout=0.2,
      max_training_steps=50,
      eval_throttle_secs=10,
      learning_rate=1e-3,
      debug=True
  )

  run_config = tf.estimator.RunConfig(
      tf_random_seed=19830610,
      save_checkpoints_steps=1000,
      keep_checkpoint_max=3,
      model_dir=model_dir
  )

  if tf.gfile.Exists(model_dir):
      print('Removing previous artifacts...')
      tf.gfile.DeleteRecursively(model_dir)

  os.makedirs(model_dir)

  estimator = run_experiment(hparams, train_data, train_labels, run_config, create_estimator_keras)

  def make_serving_input_receiver_fn():
      inputs = {'input_image': tf.placeholder(
          shape=[None,28,28], dtype=tf.float32, name='serving_input_image')}
      return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)

  export_dir = os.path.join(model_dir, 'export')

  if tf.gfile.Exists(export_dir):
      tf.gfile.DeleteRecursively(export_dir)

  estimator.export_savedmodel(
      export_dir_base=export_dir,
      serving_input_receiver_fn=make_serving_input_receiver_fn()
  )

  return export_dir


def setup_model():
  train_data, train_labels, eval_data, eval_labels = load_mnist_keras()
  export_dir = train_and_export_model(train_data, train_labels)
  return export_dir, eval_data


NUM_TRIALS = 10

def main(args):
  if len(args) > 1 and args[1] == '--inference':
    export_dir = args[2]
    _, _, eval_data, _ = load_mnist_keras()

    total_load_time = 0.0
    total_serve_time = 0.0
    saved_model_dir = os.path.join(
        export_dir, [f for f in os.listdir(export_dir) if f.isdigit()][0])
    for i in range(0, NUM_TRIALS):
      load_time, serving_time = inference_test(saved_model_dir, eval_data, repeat=10000)
      total_load_time += load_time
      total_serve_time += serving_time

    print("****************************************")
    print("*** Load time on original model: {:.2f}".format(total_load_time / NUM_TRIALS))
    print("*** Serve time on original model: {:.2f}".format(total_serve_time / NUM_TRIALS))
    print("****************************************")

    total_load_time = 0.0
    total_serve_time = 0.0
    optimized_export_dir = os.path.join(export_dir, 'optimized')
    for i in range(0, NUM_TRIALS):
      load_time, serving_time = inference_test(optimized_export_dir, eval_data,
                                               signature='serving_default',
                                               repeat=10000)
      total_load_time += load_time
      total_serve_time += serving_time
    print("****************************************")
    print("*** Load time on optimized model: {:.2f}".format(total_load_time / NUM_TRIALS))
    print("*** Serve time on optimized model: {:.2f}".format(total_serve_time / NUM_TRIALS))
    print("****************************************")

  else:
    # generate and output original model
    export_dir, eval_data = setup_model()
    saved_model_dir = os.path.join(export_dir, os.listdir(export_dir)[-1])
    describe_graph(get_graph_def_from_saved_model(saved_model_dir))
    get_size(saved_model_dir, 'saved_model.pb')
    get_metagraph(saved_model_dir)

    # freeze model and describe it
    freeze_model(saved_model_dir, 'softmax/Softmax', 'frozen_model.pb')
    frozen_filepath = os.path.join(saved_model_dir, 'frozen_model.pb')
    describe_graph(get_graph_def_from_file(frozen_filepath))
    get_size(saved_model_dir, 'frozen_model.pb', include_vars=False)

    # optimize model and describe it
    optimize_graph(saved_model_dir, 'frozen_model.pb', TRANSFORMS, 'softmax/Softmax')
    optimized_filepath = os.path.join(saved_model_dir, 'optimized_model.pb')
    describe_graph(get_graph_def_from_file(optimized_filepath))
    get_size(saved_model_dir, 'optimized_model.pb', include_vars=False)

    # convert to saved model and output metagraph again
    optimized_export_dir = os.path.join(export_dir, 'optimized')
    convert_graph_def_to_saved_model(optimized_export_dir, optimized_filepath,
                                     'softmax', 'softmax/Softmax:0')
    get_size(optimized_export_dir, 'saved_model.pb')
    get_metagraph(optimized_export_dir)


if __name__ == '__main__':
  main(sys.argv)
