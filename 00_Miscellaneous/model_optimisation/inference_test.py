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

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import numpy as np
import tensorflow as tf
from datetime import datetime
import requests
import sys
import json

BATCH_SIZE = 100

DISCOVERY_URL = 'https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json'

PROJECT = 'lramsey-goog-com-csa-ml'
MODEL_NAME = 'mnist_classifier'

credentials = GoogleCredentials.get_application_default()
api = discovery.build(
    'ml', 'v1',
    credentials=credentials,
    discoveryServiceUrl=DISCOVERY_URL
)


def load_mnist_data():
  mnist = tf.contrib.learn.datasets.load_dataset('mnist')
  train_data = mnist.train.images
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  return train_data, train_labels, eval_data, eval_labels


def load_mnist_keras():
  (train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()
  return train_data, train_labels, eval_data, eval_labels


def inference_tfserving(eval_data, batch=BATCH_SIZE, repeat=10, signature='predict'):
  url = 'http://localhost:8501/v1/models/mnist_classifier:predict'

  instances = [[float(i) for i in list(eval_data[img])] for img in range(batch)]

  request_data = {'signature_name': signature,
                  'instances': instances}

  time_start = datetime.utcnow()
  response = requests.post(url, data=json.dumps(request_data))
  if response.status_code != 200:
    raise Exception("Bad response status from TF Serving instance: %d" % response.status_code)
  for i in range(repeat-1):
    response = requests.post(url, data=json.dumps(request_data))
  time_end = datetime.utcnow()
  time_elapsed_sec = (time_end - time_start).total_seconds()

  print('Total elapsed time: {} seconds'.format(time_elapsed_sec))
  print('Time for batch size {} repeated {} times'.format(BATCH_SIZE, repeat))
  print('Average latency per batch: {} seconds'.format(time_elapsed_sec/repeat))


def predict(version, instances):
  request_data = {'instances': instances}
  model_url = 'projects/{}/models/{}/versions/{}'.format(
      PROJECT, MODEL_NAME, version)
  response = api.projects().predict(
      body=request_data, name=model_url).execute()
  class_ids = None
  try:
    class_ids = [item["class_ids"] for item in response["predictions"]]
  except:
    print(response)
  return class_ids


def inference_cmle(version, eval_data, batch=BATCH_SIZE, repeat=10):
  instances = [
    {'input_image': [float(i) for i in list(eval_data[img])]}
    for img in range(batch)
  ]

  # warmup request
  predict(version, instances[0])
  print('Warm up request performed!')
  print('Timer started...', '')

  time_start = datetime.utcnow()
  output = None
  for i in range(repeat):
    output = predict(version, instances)
  time_end = datetime.utcnow()
  time_elapsed_sec = (time_end - time_start).total_seconds()

  print('Total elapsed time: {} seconds'.format(time_elapsed_sec), '')
  print('Time for batch size {} repeated {} times'.format(BATCH_SIZE, repeat))
  print('Average latency per batch: {} seconds'.format(time_elapsed_sec/repeat))
  print('Prediction output for the last instance: {}'.format(output[0]))


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

if __name__ == '__main__':
  _, _, eval_data, _ = load_mnist_data()
  engine = sys.argv[1]
  if engine == 'predictor':
    inference_test(sys.argv[2], eval_data)
  elif engine == 'cmle':
    inference_cmle(sys.argv[2], eval_data)
  elif engine == 'tfserving':
    if len(sys.argv) > 2:
      print('Calling inference_tfserving with signature "{}"'.format(sys.argv[2]))
      inference_tfserving(eval_data, repeat=1000, signature=sys.argv[2])
    else:
      print('Calling inference_tfserving with signature "predict"')
      inference_tfserving(eval_data, repeat=1000)
