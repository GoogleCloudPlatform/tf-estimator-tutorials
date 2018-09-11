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

import tensorflow as tf
from datetime import datetime

BATCH_SIZE = 10

def inference_test(saved_model_dir,
                   eval_data,
                   signature='predict',
                   repeat=10):
  tf.logging.set_verbosity(tf.logging.ERROR)

  # load model
  time_start = datetime.utcnow()
  for i in range(repeat/10):
    predictor = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key=signature
    )
  time_end = datetime.utcnow()

  loading_time = (time_end - time_start).total_seconds()

  print('', 'Model loading time: {} seconds'.format(
      loading_time), '')

  # serve a batch of
  time_start = datetime.utcnow()
  output = None
  for i in range(repeat):
    output = predictor(
        {'input_image': eval_data[:BATCH_SIZE]}
    )
  time_end = datetime.utcnow()

  print('Prediction produced for {} instances repeated {} times'.format(
      BATCH_SIZE, repeat), '')

  serving_time = (time_end - time_start).total_seconds()
  print('Inference elapsed time: {} seconds'.format(
      serving_time), '')

  print('Prediction output for the last instance:')
  for key in output.keys():
    print('{}: {}'.format(key,output[key][0]))

  return loading_time, serving_time
