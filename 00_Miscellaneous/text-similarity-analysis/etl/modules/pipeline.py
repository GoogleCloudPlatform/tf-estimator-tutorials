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

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.coders as tft_coders

from tensorflow_transform.beam import impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata


###########################################################
# Preprocess Reuter Dataset
###########################################################

def get_paths(file_pattern, test_mode=False):
  """
  glob_pattern = './data/*.sgm'
  """
  import tensorflow as tf

  paths = tf.gfile.Glob(file_pattern)
  if test_mode:
    paths = paths[:1]

  return paths

def get_articles(file_path):
  """
  file_path = './data/reut2-000.sgm'
  """
  import bs4
  import tensorflow as tf

  data = tf.gfile.GFile(file_path).read()
  soup = bs4.BeautifulSoup(data, "html.parser")
  articles = []
  for raw_article in soup.find_all('reuters'):
    article = {
        'title': get_title(raw_article),
        'content': get_content(raw_article),
        'topics': get_topics(raw_article),
    }
    if None not in article.values():
      if [] not in article.values():
        articles.append(article)
  return articles

def get_title(article):
  title = article.find('text').title
  if title != None:
    title = title.text.encode('ascii', 'ignore')
  return title

def get_content(article):
  import nltk
  content = article.find('text').body
  if content != None:
    content = content.text.encode('ascii', 'ignore')
    content = content.replace('\n Reuter\n\x03', '')
    content = content.replace('\n', ' ')
    try:
      content = '\n'.join(nltk.sent_tokenize(content))
    except LookupError:
      nltk.download('punkt')
      content = '\n'.join(nltk.sent_tokenize(content))
  return content

def get_topics(article):
  topics = []
  for topic in article.topics.children:
    topics.append(topic.text.encode('ascii', 'ignore'))
  if len(topics) > 0:
    return ','.join(topics)
  else:
    return ''

###########################################################
# TensorFlow Transform
###########################################################

def get_metadata():
  from tensorflow_transform.tf_metadata import dataset_schema
  from tensorflow_transform.tf_metadata import dataset_metadata

  metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
    'title': dataset_schema.ColumnSchema(
        tf.string, [], dataset_schema.FixedColumnRepresentation()),
    'content': dataset_schema.ColumnSchema(
        tf.string, [], dataset_schema.FixedColumnRepresentation()),
    'topics': dataset_schema.ColumnSchema(
        tf.string, [], dataset_schema.FixedColumnRepresentation()),
  }))
  return metadata

def preprocess_fn(input_features):
  import tensorflow_transform as tft

  title_embed = tft.apply_function(get_embed_content, input_features['content'])
  content_embed = tft.apply_function(get_embed_title, input_features['title'])
  output_features = {
      'topics': input_features['topics'],
      'title': input_features['title'],
      'content': input_features['content'],
      'title_embed': title_embed,
      'content_embed': content_embed,
  }
  return output_features

def get_embed_title(
    title,
    module_url='https://tfhub.dev/google/universal-sentence-encoder/1'):
  import tensorflow as tf
  import tensorflow_hub as hub

  module = hub.Module(module_url)
  embed = module(title)
  return embed

def get_embed_content(
    content, delimiter='\n',
    module_url='https://tfhub.dev/google/universal-sentence-encoder/1'):
  import tensorflow as tf
  import tensorflow_hub as hub

  module = hub.Module(module_url)

  def _map_fn(t):
    t = tf.cast(t, tf.string)
    t = tf.string_split([t], delimiter).values
    e = module(t)
    e = tf.reduce_mean(e, axis=0)
    return tf.squeeze(e)

  embed = tf.map_fn(_map_fn, content, dtype=tf.float32)
  return embed

###########################################################
# Write data to files or bq table
###########################################################

def to_bq_row(entry):
  # might not need to round...
  entry['title_embed'] = [round(float(e), 3) for e in entry['title_embed']]
  entry['content_embed'] = [round(float(e), 3) for e in entry['content_embed']]
  return entry

def get_bigquery_schema():
  """
  Returns a bigquery schema.
  """
  from apache_beam.io.gcp.internal.clients import bigquery

  table_schema = bigquery.TableSchema()
  columns = (('topics', 'string', 'nullable'),
             ('title', 'string', 'nullable'),
             ('content', 'string', 'nullable'),
             ('title_embed', 'float', 'repeated'),
             ('content_embed', 'float', 'repeated'))

  for column in columns:
    column_schema = bigquery.TableFieldSchema()
    column_schema.name = column[0]
    column_schema.type = column[1]
    column_schema.mode = column[2]
    table_schema.fields.append(column_schema)

  return table_schema

###########################################################
# Dataflow Pipeline
###########################################################

def run(pipeline_options, known_args):
  pipeline = beam.Pipeline(options=pipeline_options)

  with impl.Context(known_args.transform_temp_dir):
    articles = (
        pipeline
        | 'Get Paths' >> beam.Create(get_paths(known_args.file_pattern))
        | 'Get Articles' >> beam.Map(get_articles)
        | 'Get Article' >> beam.FlatMap(lambda x: x)
    )

    dataset = (articles, get_metadata())

    transform_fn = (
        dataset
        | 'Analyse dataset' >> impl.AnalyzeDataset(preprocess_fn)
    )

    transformed_data_with_meta = (
        (dataset, transform_fn)
        | 'Transform dataset' >> impl.TransformDataset()
    )

    transformed_data, transformed_metadata = transformed_data_with_meta

    transform_fn | 'Export Transform Fn' >> transform_fn_io.WriteTransformFn(
        known_args.transform_export_dir)

    (
        transformed_data
        | 'Convert to Insertable data' >> beam.Map(to_bq_row)
        | 'Write to BigQuery table' >> beam.io.WriteToBigQuery(
            project=known_args.bq_project,
            dataset=known_args.bq_dataset,
            table=known_args.bq_table,
            schema=get_bigquery_schema(),
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
    )

    if known_args.enable_tfrecord:
      transformed_data | 'Write TFRecords' >> beam.io.tfrecordio.WriteToTFRecord(
          file_path_prefix='{0}/{1}'.format(known_args.tfrecord_export_dir, 'reuter'),
          file_name_suffix='.tfrecords',
          coder=tft_coders.example_proto_coder.ExampleProtoCoder(transformed_metadata.schema))

    if known_args.enable_debug:
      transformed_data | 'Debug Output' >> beam.io.textio.WriteToText(
          file_path_prefix=known_args.debug_output_prefix, file_name_suffix='.txt')


  job = pipeline.run()

  if pipeline_options.get_all_options()['runner'] == 'DirectRunner':
    job.wait_until_finish()
