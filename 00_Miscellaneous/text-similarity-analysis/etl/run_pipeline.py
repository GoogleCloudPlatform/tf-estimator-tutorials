#!/usr/bin/python
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import datetime
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions

from modules import pipeline


def add_input_options(parser):
  input_group = parser.add_argument_group('input options')
  input_group.add_argument('--file_pattern', dest='file_pattern',
                           default='./data/*.sgm',
                           help='A directory location of input data')

def add_transform_options(parser):
  transform_group = parser.add_argument_group('transform options')
  transform_group.add_argument('--transform_temp_dir',
                               dest='transform_temp_dir', default='tft_temp',
                               help='A temp directory used by tf.transform.')
  transform_group.add_argument('--transform_export_dir',
                               dest='transform_export_dir', default='tft_out',
                               help='A directory where tft function is saved')

def add_bigquery_options(parser):
  bigquery_group = parser.add_argument_group('bigquery options')
  bigquery_group.add_argument('--bq_project',
                              dest='bq_project', required=True,
                              help='Project name used in BigQuery.')
  bigquery_group.add_argument('--bq_dataset',
                              dest='bq_dataset', required=True,
                              help='Dataset name used in BigQuery.')
  bigquery_group.add_argument('--bq_table',
                              dest='bq_table', required=True,
                              help='Table name used in BigQuery.')

def add_tfrecord_options(parser):
  tfrecord_group = parser.add_argument_group('tfrecord options')
  tfrecord_group.set_defaults(enable_tfrecord=False)
  tfrecord_group.add_argument('--enable_tfrecord',
                              dest='enable_tfrecord', action='store_true',
                              help='Export transformed data in tfrecords format.')
  tfrecord_group.add_argument('--tfrecord_export_dir',
                              dest='tfrecord_export_dir', default='./export',
                              help='A directory where you save transform function')

def add_debug_options(parser):
  debug_group = parser.add_argument_group('debug options')
  debug_group.set_defaults(enable_debug=False)
  debug_group.add_argument('--enable_debug',
                           dest='enable_debug', action='store_true',
                           help='Enable debug options.')
  debug_group.add_argument('--debug_output_prefix',
                           dest='debug_output_prefix', default='debug-output',
                           help='Specify prefix of debug output.')
def main(argv=None):
  parser = argparse.ArgumentParser()
  add_input_options(parser)
  add_transform_options(parser)
  add_bigquery_options(parser)
  add_tfrecord_options(parser)
  add_debug_options(parser)
  known_args, pipeline_args = parser.parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
  setup_options = pipeline_options.view_as(SetupOptions)
  setup_options.save_main_session = True
  pipeline.run(pipeline_options, known_args)

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.ERROR)
  main()
