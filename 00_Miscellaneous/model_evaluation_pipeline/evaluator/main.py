from __future__ import absolute_import

import argparse
import datetime
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from process import pipeline


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--year_from',
                      dest='year_from',
                      default=1969,
                      help='A condition to extract test data from BigQuery.')
  parser.add_argument('--year_to',
                      dest='year_to',
                      default=1973,
                      help='A condition to extract test data from BigQuery.')
  parser.add_argument('--datasize',
                      dest='datasize',
                      default=1000000,
                      help='Maximum number of rows from BigQuery.')
  parser.add_argument('--saved_model_dir',
                      dest='saved_model_dir',
                      default='gs://yaboo-cloud-training-demos-ml/babyweight/trained_model/1969-1973/export/exporter/1525182730',
                      help='A path to a saved model directory on GCS.')
  parser.add_argument('--output_table',
                      dest='output_table',
                      default='yaboo-sandbox:testing.prediction3',
                      help='A name of a BQ table where you save eval results.')

  known_args, pipeline_args = parser.parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
  setup_options = pipeline_options.view_as(SetupOptions)
  setup_options.save_main_session = True

  pipeline.run(pipeline_options,
               known_args.saved_model_dir,
               known_args.year_from,
               known_args.year_to,
               known_args.datasize,
               known_args.output_table)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  main()
