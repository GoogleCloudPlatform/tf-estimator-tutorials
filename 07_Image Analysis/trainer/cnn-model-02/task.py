# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Example implementation of code to run on the Cloud ML service.
"""

import fnmatch
import os
import tensorflow as tf

from gcloud import storage

import sample_model as sm


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'bucket_name', '', 'bucket_name.')
tf.app.flags.DEFINE_string(
    'output_dir', '', 'GCS location to root directory for checkpoints and exported models.')
tf.app.flags.DEFINE_string(
    'model_name', 'cnn-model-02', 'model name.')
tf.app.flags.DEFINE_string(
    'export_dir', 'Servo', 'model export directory.')
tf.app.flags.DEFINE_integer(
    'batch_size', 200, 'batch size for training.')
tf.app.flags.DEFINE_integer(
    'max_steps', 1000, 'max_step for training.')
tf.app.flags.DEFINE_integer(
    'eval_steps', 50, 'The number of steps that are used in evaluation phase.')
tf.app.flags.DEFINE_bool(
    'use_checkpoint', True, 'True if use checkpoints.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_steps', 500, '')
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 19851211, '')
tf.app.flags.DEFINE_string(
    'train_data_pattern', 'cifar-10/train*.tfrecord', 'path to train dataset on GCS.')
tf.app.flags.DEFINE_string(
    'eval_data_pattern', 'cifar-10/valid*.tfrecord', 'path to eval dataset on GCS.')


tf.logging.set_verbosity(tf.logging.INFO)


def get_filenames(pattern):
  storage_client = storage.Client()
  bucket = storage_client.lookup_bucket(FLAGS.bucket_name)

  prefix = '/'.join(pattern.split('/')[:-1])
  filenames = []

  for blob in bucket.list_blobs(prefix=prefix):
    if fnmatch.fnmatch(blob.name, pattern):
      filenames.append('gs://' + os.path.join(blob.bucket.name, blob.name))

  tf.logging.info(filenames)
  return filenames


def train_and_evaluate():
  model_dir = os.path.join(FLAGS.output_dir, FLAGS.model_name)

  # Create estimator from Keras model.
  estimator = tf.estimator.Estimator(
      model_fn=sm.model_fn,
      model_dir=model_dir,
      config=tf.estimator.RunConfig(
          # save_checkpoints_steps=FLAGS.save_checkpoints_steps,
          keep_checkpoint_max=5,
          tf_random_seed=FLAGS.tf_random_seed))

  # Profile Hook.
  profile_hook = tf.train.ProfilerHook(
      save_steps=FLAGS.save_checkpoints_steps,
      output_dir=model_dir,
      show_dataflow=True,
      show_memory=True)

  # Specify training data paths, batch size and max steps.
  train_spec = tf.estimator.TrainSpec(
      input_fn=sm.generate_input_fn(filenames=get_filenames(FLAGS.train_data_pattern),
                                    mode=tf.estimator.ModeKeys.TRAIN,
                                    batch_size=FLAGS.batch_size),
      max_steps=FLAGS.max_steps,
      hooks=[profile_hook]
  )

  # Currently (2017.12.14) the latest tf only support exporter with keras models.
  exporter = tf.estimator.LatestExporter(
      name=FLAGS.export_dir, serving_input_receiver_fn=sm.serving_input_fn,
      assets_extra=None, as_text=False, exports_to_keep=5)

  # Specify validation data paths, steps for evaluation and exporter specs
  eval_spec = tf.estimator.EvalSpec(
      input_fn=sm.generate_input_fn(filenames=get_filenames(FLAGS.eval_data_pattern),
                                    mode=tf.estimator.ModeKeys.EVAL,
                                    batch_size=FLAGS.batch_size),
      steps=FLAGS.eval_steps,
      name=None,
      hooks=None,
      exporters=exporter # Iterable of Exporters, or single one or None.
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main(unused_argv=None):
  tf.logging.info(tf.__version__)
  train_and_evaluate()

if __name__ == '__main__':
  tf.app.run()
