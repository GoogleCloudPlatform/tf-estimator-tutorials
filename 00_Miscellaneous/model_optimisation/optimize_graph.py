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

""" Extract from notebook for Serving Optimization """

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

from inference_test import inference_test, load_mnist_data

NUM_CLASSES = 10
MODELS_LOCATION = 'models/mnist'
MODEL_NAME = 'cnn_classifier'


def model_fn(features, labels, mode, params):
  # conv layers
  def _cnn_layers(input_layer, num_conv_layers, init_filters, mode):
    inputs = input_layer
    for i in range(num_conv_layers):
      current_filters = init_filters * (2**i)
      conv = tf.layers.conv2d(inputs=inputs, kernel_size=3,
                              filters=current_filters, strides=1,
                              padding='SAME', name='conv{}'.format(i+1))
      pool = tf.layers.max_pooling2d(
          inputs=conv, pool_size=2, strides=2,
          padding='SAME', name='pool{}'.format(i+1))
      batch_norm = tf.layers.batch_normalization(
          pool, name='batch_norm{}'.format(i+1))
      if mode==tf.estimator.ModeKeys.TRAIN:
          batch_norm = tf.nn.dropout(batch_norm, params.drop_out)
      inputs = batch_norm
    outputs = batch_norm
    return outputs

  # model body
  def _body(features, mode, params):
    input_layer = tf.reshape(features['input_image'],
                             [-1, 28, 28, 1], name='input_image')
    conv_outputs = _cnn_layers(input_layer,
                               params.num_conv_layers,
                               params.init_filters, mode)
    flatten = tf.layers.flatten(inputs=conv_outputs, name='flatten')
    fully_connected = tf.contrib.layers.stack(
        inputs=flatten,
        layer=tf.contrib.layers.fully_connected,
        stack_args=params.hidden_units,
        activation_fn=tf.nn.relu)
    # unused_layer
    unused_layers = tf.layers.dense(flatten, units=100, name='unused',
                                    activation=tf.nn.relu)
    logits = tf.layers.dense(fully_connected, units=NUM_CLASSES,
                             name='logits', activation=None)
    return logits

  # model head
  head = tf.contrib.estimator.multi_class_head(n_classes=NUM_CLASSES)

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      logits=_body(features, mode, params),
      labels=labels,
      optimizer=tf.train.AdamOptimizer(params.learning_rate)
  )


def create_estimator(params, run_config):

  def _metric_fn(labels, predictions):
    metrics = {}
    pred_class = predictions['class_ids']
    metrics['micro_accuracy'] = tf.metrics.mean_per_class_accuracy(
        labels=labels, predictions=pred_class, num_classes=NUM_CLASSES
    )
    return metrics

  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn, params=params, config=run_config)

  mnist_classifier = tf.contrib.estimator.add_metrics(
      estimator=mnist_classifier, metric_fn=_metric_fn)

  return mnist_classifier


#### Run Experiment

def run_experiment(hparams, train_data, train_labels, run_config, create_estimator_fn=create_estimator):
  train_spec = tf.estimator.TrainSpec(
      input_fn = tf.estimator.inputs.numpy_input_fn(
          x={'input_image': train_data},
          y=train_labels,
          batch_size=hparams.batch_size,
          num_epochs=None,
          shuffle=True),
      max_steps=hparams.max_training_steps
  )
  eval_spec = tf.estimator.EvalSpec(
      input_fn = tf.estimator.inputs.numpy_input_fn(
          x={'input_image': train_data},
          y=train_labels,
          batch_size=hparams.batch_size,
          num_epochs=1,
          shuffle=False),
      steps=None,
      throttle_secs=hparams.eval_throttle_secs
  )

  tf.logging.set_verbosity(tf.logging.INFO)

  time_start = datetime.utcnow()
  print('Experiment started at {}'.format(time_start.strftime('%H:%M:%S')))
  print('.......................................')

  estimator = create_estimator_fn(hparams, run_config)

  tf.estimator.train_and_evaluate(
      estimator=estimator,
      train_spec=train_spec,
      eval_spec=eval_spec
  )

  time_end = datetime.utcnow()
  print('.......................................')
  print('Experiment finished at {}'.format(time_end.strftime('%H:%M:%S')))
  print('')
  time_elapsed = time_end - time_start
  print('Experiment elapsed time: {} seconds'.format(time_elapsed.total_seconds()))

  return estimator


#### Train and Export Model

def train_and_export_model(train_data, train_labels):
  model_dir = os.path.join(MODELS_LOCATION, MODEL_NAME)

  hparams  = tf.contrib.training.HParams(
      batch_size=100,
      hidden_units=[1024],
      num_conv_layers=2,
      init_filters=64,
      drop_out=0.85,
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

  estimator = run_experiment(hparams, train_data, train_labels, run_config)

  def make_serving_input_receiver_fn():
      inputs = {'input_image': tf.placeholder(
          shape=[None,784], dtype=tf.float32, name='input_image')}
      return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)

  export_dir = os.path.join(model_dir, 'export')

  if tf.gfile.Exists(export_dir):
      tf.gfile.DeleteRecursively(export_dir)

  estimator.export_savedmodel(
      export_dir_base=export_dir,
      serving_input_receiver_fn=make_serving_input_receiver_fn()
  )

  return export_dir


#### Load GraphDef from a SavedModel Directory

def get_graph_def_from_saved_model(saved_model_dir):
  with tf.Session() as session:
      meta_graph_def = tf.saved_model.loader.load(
          session,
          tags=[tag_constants.SERVING],
          export_dir=saved_model_dir
      )
  return meta_graph_def.graph_def


#### Describe GraphDef

def describe_graph(graph_def, show_nodes=False):
  print('Input Feature Nodes: {}'.format([node.name for node in graph_def.node if node.op=='Placeholder']))
  print('')
  print('Unused Nodes: {}'.format([node.name for node in graph_def.node if 'unused'  in node.name]))
  print('')
  print('Output Nodes: {}'.format(
      [node.name for node in graph_def.node if ('predictions' in node.name or 'softmax' in node.name)]))
  print('')
  print('Quantization Nodes: {}'.format( [node.name for node in graph_def.node if 'quant' in node.name]))
  print('')
  print('Constant Count: {}'.format( len([node for node in graph_def.node if node.op=='Const'])))
  print('')
  print('Variable Count: {}'.format( len([node for node in graph_def.node if 'Variable' in node.op])))
  print('')
  print('Identity Count: {}'.format( len([node for node in graph_def.node if node.op=='Identity'])))
  print('')
  print('Total nodes: {}'.format( len(graph_def.node)))
  print('')

  if show_nodes==True:
    for node in graph_def.node:
      print('Op:{} - Name: {}'.format(node.op, node.name))


#### Get model size and metagraph

def get_size(model_dir, model_file='saved_model.pb', include_vars=True):
  model_file_path = os.path.join(model_dir, model_file)
  print(model_file_path, '')
  pb_size = os.path.getsize(model_file_path)
  variables_size = 0
  if include_vars and os.path.exists(os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
    variables_size = os.path.getsize(os.path.join(
        model_dir,'variables/variables.data-00000-of-00001'))
    variables_size += os.path.getsize(os.path.join(model_dir,'variables/variables.index'))
  print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
  if include_vars:
    print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
    print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))


def get_metagraph(model_dir):
  output = sh.saved_model_cli('show', '--dir=%s' % model_dir, '--all')
  print(output)


#### Get graph def from MetaGraphDef

def get_graph_def_from_file(graph_filepath):
  with ops.Graph().as_default():
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def


def optimize_graph(model_dir, graph_filename, transforms, output_node):
  input_names = []
  output_names = [output_node]
  if graph_filename is None:
    graph_def = get_graph_def_from_saved_model(model_dir)
  else:
    graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
  optimized_graph_def = TransformGraph(
      graph_def,
      input_names,
      output_names,
      transforms)
  tf.train.write_graph(optimized_graph_def,
                      logdir=model_dir,
                      as_text=False,
                      name='optimized_model.pb')
  print('****************************************')
  print('Graph optimized!')
  print('****************************************')


def freeze_model(saved_model_dir, output_node_names, output_filename):
  output_graph_filename = os.path.join(saved_model_dir, output_filename)
  initializer_nodes = ''
  freeze_graph.freeze_graph(
      input_saved_model_dir=saved_model_dir,
      output_graph=output_graph_filename,
      saved_model_tags = tag_constants.SERVING,
      output_node_names=output_node_names,
      initializer_nodes=initializer_nodes,
      input_graph=None,
      input_saver=False,
      input_binary=False,
      input_checkpoint=None,
      restore_op_name=None,
      filename_tensor_name=None,
      clear_devices=False,
      input_meta_graph=False,
  )
  print('****************************************')
  print('graph freezed!')
  print('****************************************')


def convert_graph_def_to_saved_model(export_dir, graph_filepath, output_key, output_node_name):
  if tf.gfile.Exists(export_dir):
    tf.gfile.DeleteRecursively(export_dir)
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    tf.saved_model.simple_save(
        session,
        export_dir,
        inputs={
            node.name: session.graph.get_tensor_by_name(
                '{}:0'.format(node.name))
            for node in graph_def.node if node.op=='Placeholder'},
        outputs={output_key: session.graph.get_tensor_by_name(
            output_node_name)}
    )
    print('****************************************')
    print('Optimized graph converted to SavedModel!')
    print('****************************************')


def setup_model():
  train_data, train_labels, eval_data, eval_labels = load_mnist_data()
  export_dir = train_and_export_model(train_data, train_labels)
  return export_dir, eval_data


TRANSFORMS = [
    'remove_nodes(op=Identity)',
    'fold_constants(ignore_errors=true)',
    'merge_duplicate_nodes',
    'strip_unused_nodes',
    'fold_batch_norms'
]

QUANTIZE_TRANSFORMS = [
    'quantize_weights',
    'quantize_nodes'
]

NUM_TRIALS = 10

def main(args):
  if len(args) > 1 and args[1] == '--inference':
    export_dir = args[2]
    _, _, eval_data, _ = load_mnist_data()

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
    freeze_model(saved_model_dir, 'head/predictions/class_ids', 'frozen_model.pb')
    frozen_filepath = os.path.join(saved_model_dir, 'frozen_model.pb')
    describe_graph(get_graph_def_from_file(frozen_filepath))
    get_size(saved_model_dir, 'frozen_model.pb', include_vars=False)

    # optimize model and describe it
    optimize_graph(saved_model_dir, 'frozen_model.pb', TRANSFORMS, 'head/predictions/class_ids')
    optimized_filepath = os.path.join(saved_model_dir, 'optimized_model.pb')
    describe_graph(get_graph_def_from_file(optimized_filepath))
    get_size(saved_model_dir, 'optimized_model.pb', include_vars=False)

    # convert to saved model and output metagraph again
    optimized_export_dir = os.path.join(export_dir, 'optimized')
    convert_graph_def_to_saved_model(optimized_export_dir, optimized_filepath, 'class_ids',
                                     'head/predictions/class_ids:0')
    get_size(optimized_export_dir, 'saved_model.pb')
    get_metagraph(optimized_export_dir)



if __name__ == '__main__':
  main(sys.argv)
