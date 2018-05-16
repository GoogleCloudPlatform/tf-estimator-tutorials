import os
import logging

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

SAVED_MODEL_DIR = 'model'
PROJECT = 'yaboo-sandbox'

export_dir = None
predictor_fn = None

def init_predictor(saved_model_dir):
  global export_dir
  global predictor_fn

  if predictor_fn is None:
    print('initializing predictor...')
    if saved_model_dir == None:
      dir_path = os.path.dirname(os.path.realpath(__file__))
      export_dir = os.path.join(dir_path, SAVED_MODEL_DIR)
    else:
      export_dir = saved_model_dir

    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=export_dir, signature_def_key='predict')

  return export_dir, predictor_fn

def estimate_local(instance, saved_model_dir=None):
  export_dir, predictor_fn = init_predictor(saved_model_dir)
  instance = dict((k, [v]) for k, v in instance.items())
  value = predictor_fn(instance)['predictions'][0][0]
  return export_dir, value
