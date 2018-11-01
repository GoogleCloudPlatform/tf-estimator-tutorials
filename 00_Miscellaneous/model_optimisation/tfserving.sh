#!/bin/bash

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

rm -rf /tmp/tfserving
mkdir -p /tmp/tfserving

saved_models_base=models/mnist/keras_classifier/export

if [[ $# == 0 ]]; then
  saved_model_dir=${saved_models_base}/$(ls ${saved_models_base} | head -n 1)
else
  saved_model_dir=${saved_models_base}/$1
fi

cp -r $saved_model_dir /tmp/tfserving/0000000001

docker run -p 8501:8501 --mount type=bind,source=/tmp/tfserving,target=/models/mnist_classifier \
-e MODEL_NAME=mnist_classifier -t tensorflow/serving
