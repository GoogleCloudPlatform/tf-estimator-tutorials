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

import os
import tensorflow as tf

import trainer.sample_model as sm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    'max_steps', 1000, 'max_step for training.')
tf.app.flags.DEFINE_string(
    'output_dir', '', 'GCS location to root directory for checkpoints and exported models.')
tf.app.flags.DEFINE_string(
    'model_name', 'sample_model', 'model name.')
tf.app.flags.DEFINE_integer(
    'train_batch_size', 200, 'batch size for training.')
tf.app.flags.DEFINE_integer(
    'eval_batch_size', 200, 'batch size for evaluation.')
tf.app.flags.DEFINE_integer(
    'eval_steps', 50, 'The number of steps that are used in evaluation phase.')
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 19851211, '')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_steps', 500, '')
tf.app.flags.DEFINE_string(
    'train_data_pattern', 'cifar-10/train*.tfrecord', 'path to train dataset on GCS.')
tf.app.flags.DEFINE_string(
    'eval_data_pattern', 'cifar-10/valid*.tfrecord', 'path to eval dataset on GCS.')
tf.app.flags.DEFINE_float(
    'learning_rate', 1e-3, 'learning rate.')
tf.app.flags.DEFINE_integer(
    'num_gpus', 1, 'num of gpus in single-node-multi-GPUs setting.')
tf.app.flags.DEFINE_integer(
    'num_gpus_per_worker', 0, 'num of gpus for each node.')
tf.app.flags.DEFINE_bool(
    'auto_shard_dataset', False,
    'whether to auto-shard the dataset when there are multiple workers.')
tf.app.flags.DEFINE_float(
    'drop_out_rate', 1e-2, 'drop out rate')
tf.app.flags.DEFINE_integer(
    'dense_units', 1024, 'units in dense layer.')

tf.logging.set_verbosity(tf.logging.INFO)

def parse_tfrecord(example):
    feature={'label': tf.FixedLenFeature((), tf.int64),
             'image': tf.FixedLenFeature((), tf.string, default_value="")}
    parsed = tf.parse_single_example(example, feature)
    image = tf.decode_raw(parsed['image'],tf.float64)
    image = tf.cast(image,tf.float32)
    image = tf.reshape(image,[32,32,3])
    return image, parsed['label']


def image_scaling(x):
    return tf.image.per_image_standardization(x)

def distort(x):
    x = tf.image.resize_image_with_crop_or_pad(x, 40, 40)
    x = tf.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x

def dataset_input_fn(params):
    dataset = tf.data.TFRecordDataset(params['filenames'],
                                      num_parallel_reads=params['threads'])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=params['threads'])
    dataset = dataset.map(
        lambda x,y: (image_scaling(x),y), num_parallel_calls=params['threads'])
    if params['mode']==tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.map(
            lambda x,y: (distort(x),y), num_parallel_calls=params['threads'])
        dataset = dataset.shuffle(buffer_size=params['shuffle_buff'])
    dataset = dataset.repeat()
    dataset = dataset.batch(params['batch'])
    dataset = dataset.prefetch(8*params['batch'])
    return dataset


def train_dataset_input_fn(pattern):
    files = tf.gfile.Glob(pattern)
    params = {'filenames': files, 'mode': tf.estimator.ModeKeys.TRAIN,
              'threads': 16, 'shuffle_buff': 100000, 'batch': FLAGS.train_batch_size}
    return dataset_input_fn(params)


def eval_dataset_input_fn(pattern):
    files = tf.gfile.Glob(pattern)
    params = {'filenames': tf.gfile.Glob(pattern), 'mode': tf.estimator.ModeKeys.EVAL,
              'threads': 16, 'batch': FLAGS.eval_batch_size}
    return dataset_input_fn(params)


def serving_input_fn():
    receiver_tensor = {'images': tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)}
    features = tf.map_fn(image_scaling, receiver_tensor['images'])
    return tf.estimator.export.TensorServingInputReceiver(features, receiver_tensor)


def train_and_evaluate():
    model_dir = os.path.join(FLAGS.output_dir, FLAGS.model_name)

    # MirroredStrategy
    if FLAGS.num_gpus_per_worker > 0:
        distribution = tf.contrib.distribute.MirroredStrategy(
            num_gpus_per_worker=FLAGS.num_gpus_per_worker,
            auto_shard_dataset=FLAGS.auto_shard_dataset)
    elif FLAGS.num_gpus > 0:
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)
    else:
        distribution = None

    # Configuration for Estimator
    config = tf.estimator.RunConfig(
        save_checkpoints_secs=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=5,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        train_distribute=distribution,
        tf_random_seed=FLAGS.tf_random_seed)

    model_params = {
        'drop_out': FLAGS.drop_out_rate,
        'dense_units': FLAGS.dense_units,
        'learning_rate': FLAGS.learning_rate,
        'log': True}

    # Create Estimator.
    estimator = tf.estimator.Estimator(
        model_fn=sm.model_fn,
        model_dir=model_dir,
        params=model_params,
        config=config)

    # Specify training data paths, batch size and max steps.
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_dataset_input_fn(FLAGS.train_data_pattern),
        max_steps=FLAGS.max_steps)
    
    # Configuration for model exportation
    exporter = tf.estimator.LatestExporter(
        name='export',
        serving_input_receiver_fn=serving_input_fn,
        assets_extra=None, as_text=False, exports_to_keep=5)

    # Specify validation data paths, steps for evaluation and exporter specs
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_dataset_input_fn(FLAGS.eval_data_pattern),
        steps=FLAGS.eval_steps, exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main(unused_argv=None):
    tf.logging.info(tf.__version__)
    train_and_evaluate()
    
if __name__ == '__main__':
    tf.app.run()
