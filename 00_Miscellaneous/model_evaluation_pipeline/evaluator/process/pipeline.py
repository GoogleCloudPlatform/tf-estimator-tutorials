from __future__ import absolute_import

import logging
import time

import apache_beam as beam
import tensorflow as tf

predictor_fn = None
table_schema = None
tf.logging.set_verbosity(tf.logging.ERROR)


def get_bigquery_query(year_from, year_to, datasize):
  query = """
  SELECT
    is_male, mother_age, plurality, gestation_weeks, weight_pounds
  FROM
    publicdata.samples.natality
  WHERE
    year >= {0} AND year <= {1}
    AND weight_pounds > 0
    AND mother_age > 0
    AND plurality > 0
    AND gestation_weeks > 0
    AND month > 0
  LIMIT {2}
  """.format(year_from, year_to, datasize)
  return query


def process_bigquery_row(bq_row):
  # modify opaque numeric race code into human-readable data.
  plurality = dict(zip([1, 2, 3, 4, 5, 6],
                       ['Single(1)', 'Twins(2)', 'Triplets(3)',
                        'Quadruplets(4)', 'Quintuplets(5)', 'Multiple(2+)']))
  instance = dict()
  instance['is_male'] = str(bq_row['is_male'])
  instance['mother_age'] = bq_row['mother_age']
  instance['plurality'] = plurality[bq_row['plurality']]
  instance['gestation_weeks'] = bq_row['gestation_weeks']
  instance['weight_true'] = float(bq_row['weight_pounds'])
  return instance


def process_instance(instance, saved_model_dir, year_from, year_to):
  # pop weight_true since it shouldn't be included in an instance.
  weight_true = instance.pop('weight_true')
  weight_predicted, time_inference = predict_weight(instance, saved_model_dir)

  # convert value to JSON serializable format.
  instance['weight_true'] = float(weight_true)
  instance['weight_predicted'] = float(weight_predicted)
  instance['weight_residual'] = (
      instance['weight_true'] - instance['weight_predicted'])

  # put test conditions to each instance.
  instance['model'] = saved_model_dir
  instance['testdata'] = '{0}-{1}'.format(year_from, year_to)
  instance['time_inference'] = 1000 * time_inference # sec to msec
  return instance


def predict_weight(instance, saved_model_dir):
  predictor_fn = init_predictor(saved_model_dir)
  instance = dict((k, [v]) for k, v in instance.items())

  tic = time.time()
  weight_predicted = predictor_fn(instance)['predictions'][0][0]
  toc = time.time()
  time_inference = toc - tic

  return weight_predicted, time_inference


def init_predictor(saved_model_dir):
  import tensorflow as tf

  global predictor_fn
  # Re-use predictor_fn once it's initialized.
  if predictor_fn is None:
    logging.info('initializing predictor...')
    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir, signature_def_key='predict')
  return predictor_fn


def get_output_table_schema():
  global table_schema
  if table_schema == None:
    column_names = ['is_male', 'mother_age', 'plurality', 'gestation_weeks',
                    'weight_true', 'weight_predicted', 'weight_residual',
                    'model', 'testdata', 'time_inference']
    column_types = {
        'model': 'STRING',
        'testdata': 'STRING',
        'is_male': 'STRING',
        'mother_age': 'INTEGER',
        'plurality': 'STRING',
        'gestation_weeks': 'INTEGER',
        'weight_true': 'FLOAT',
        'weight_predicted': 'FLOAT',
        'weight_residual': 'FLOAT',
        'time_inference': 'FLOAT',
    }
    table_schema = ','.join(
        ['{0}:{1}'.format(k, column_types[k]) for k in column_names])
  return table_schema

def run(pipeline_options, saved_model_dir, year_from, year_to, datasize,
        output_table):
  pipeline = beam.Pipeline(options=pipeline_options)

  (
      pipeline
      | 'Extract data' >> beam.io.Read(beam.io.BigQuerySource(
          query=get_bigquery_query(year_from, year_to, datasize),
          use_standard_sql=True))
      | 'Process rows' >> beam.Map(process_bigquery_row)
      | 'Predict weight' >> beam.Map(lambda instance: process_instance(
          instance, saved_model_dir, year_from, year_to))
      | 'Write to table' >> beam.io.Write(
          beam.io.BigQuerySink(
              output_table,
              schema=get_output_table_schema(),
              create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
              write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))
  )

  job = pipeline.run()

  if pipeline_options.get_all_options()['runner'] == 'DirectRunner':
    job.wait_until_finish()
