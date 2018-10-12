#!/bin/bash -eu
#
# Copyright 2018 Google LLC
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

# Configurable Parameters
PROJECT="YOUR-PROJECT-NAME"
BUCKET="gs://YOUR-BUCKET-NAME"
REGION="us-central1"
ZONE="us-central1-c"
DATASET="reuters"
TABLE="embeddings"
FILE_PATTERN="$BUCKET/data/*.sgm"

# Working directories for dataflow
ROOT_DIR="$BUCKET/dataflow"
STAGING_LOCATION="$ROOT_DIR/staging"
TEMP_LOCATION="$ROOT_DIR/temp"

# Working directories for tf.transform
TRANSFORM_ROOT_DIR="$ROOT_DIR/transform"
TRANSFORM_TEMP_DIR="$TRANSFORM_ROOT_DIR/temp"
TRANSFORM_EXPORT_DIR="$TRANSFORM_ROOT_DIR/export"

# Working directories for TFRecords and Debug log
TFRECORD_EXPORT_DIR="$ROOT_DIR/tfrecords"
DEBUG_OUTPUT_PREFIX="$ROOT_DIR/debug/log"

# Running Config for Dataflow
RUNNER=DataflowRunner
JOB_NAME=text-analysis
MACHINE_TYPE=n1-highmem-2

# Remove Root directory before running dataflow job.
gsutil rm -r $ROOT_DIR

# Command to invoke dataflow job.
python run_pipeline.py \
  --file_pattern=$FILE_PATTERN \
  --bq_project=$PROJECT \
  --bq_dataset=$DATASET \
  --bq_table=$TABLE \
  --transform_temp_dir=$TRANSFORM_TEMP_DIR \
  --transform_export_dir=$TRANSFORM_EXPORT_DIR \
  --enable_tfrecord \
  --tfrecord_export_dir $TFRECORD_EXPORT_DIR \
  --enable_debug \
  --debug_output_prefix=$DEBUG_OUTPUT_PREFIX \
  --project=$PROJECT \
  --runner=$RUNNER \
  --region=$REGION \
  --staging_location=$STAGING_LOCATION \
  --temp_location=$TEMP_LOCATION \
  --setup_file=$(pwd)/setup.py \
  --job_name=$JOB_NAME \
  --worker_machine_type=$MACHINE_TYPE
