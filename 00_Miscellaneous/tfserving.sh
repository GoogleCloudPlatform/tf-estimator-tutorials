#!/bin/bash

rm -rf /tmp/tfserving
mkdir -p /tmp/tfserving

saved_models_base=models/mnist/cnn_classifier/export/
saved_model_dir=${saved_models_base}$(ls ${saved_models_base} | tail -n 1)
cp -r $saved_model_dir /tmp/tfserving

docker run -p 8501:8501 --mount type=bind,source=/tmp/tfserving,target=/models/mnist_classifier \
-e MODEL_NAME=mnist_classifier -t tensorflow/serving
