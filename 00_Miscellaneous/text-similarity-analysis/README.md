# Text Semantic Similarity Analysis Pipeline

This is a Dataflow pipeline that extracts feature embeddings from
article documents, located in Google Cloud Storage, and store them to
BigQuery. After successfully running this pipeline, you can start
searching similar documents from BigQuery using a cosine distance
between feature embeddings as a similarity.

Figure 1 shows the overall architecture of the pipeline. It uses
reuters-21578, which is a collection of publicly available articles,
as input. The pipeline is implemented using Apache Beam and
tf.transform, and runs at scale on Cloud Dataflow.

(figure should be added here)

In the pipeline, documents are processed to extract each article’s
title, topics, and content. The processing pipeline uses the
“Universal Sentence Encoder” module in tf.hub to extract text
embeddings for both the title and the content of each article read
from the source documents. The title, topics, and content of each
article, along with the extracted embeddings are stored in
BigQuery. Having the articles, along with their embeddings, stored in
BigQuery allow us to explore similar articles, using cosine similarity
metric between embeddings of tiles and/or contents.

# How to run the pipeline

## Requirements

You need to have your [GCP Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects). You can use [Cloud Shell](https://cloud.google.com/shell/docs/quickstart) or [gcloud CLI](https://cloud.google.com/sdk/) to run all the commands in this guideline.

## Setup a project

Follow the [instruction](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and create a GCP project. 
Once created, enable the Dataflow API, BigQuery API in this [page](https://console.developers.google.com/apis/enabled). You can also find more details about enabling the [billing](https://cloud.google.com/billing/docs/how-to/modify-project?#enable-billing)

We recommend to use CloudShell from the GCP console to run the below commands. CloudShell starts with an environment already logged in to your account and set to the currently selected project. The following commands are required only in a workstation shell environment, they are not needed in the CloudShell. 

```bash
gcloud auth login
gcloud config set project [your-project-id]
gcloud config set compute/zone us-central1-a
```

## Prepare a input data

You need to download reuters-21578 dataset from from https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection. After downloading reuters.tar.gz from the site, you need to type the following commands to store reuter dataset to Google Cloud Storage.

```bash
tar -zxvf reuters.tar.gz
gsutil -m cp -R reuters gs://${YOUR-BUCKET-NAME}
```

## Run the pipeline

You can find a shell script (link) that lets you easily run the pipeline.

```bash
PROJECT="YOUR-PROJECT-NAME"
BUCKET="gs://YOUR-BUCKET-NAME"
REGION="us-central1"
ZONE="us-central1-c"
```

```bash
DATASET="reuters"
TABLE="embeddings"
FILE_PATTERN="$BUCKET/data/*.sgm"
```

```bash
# Working directories for dataflow
ROOT_DIR="$BUCKET/dataflow"
STAGING_LOCATION="$ROOT_DIR/staging"
TEMP_LOCATION="$ROOT_DIR/temp"
```

```bash
# Working directories for tf.transform
TRANSFORM_ROOT_DIR="$ROOT_DIR/transform"
TRANSFORM_TEMP_DIR="$TRANSFORM_ROOT_DIR/temp"
TRANSFORM_EXPORT_DIR="$TRANSFORM_ROOT_DIR/export"
```

```bash
# Working directories for TFRecords and Debug log
TFRECORD_EXPORT_DIR="$ROOT_DIR/tfrecords"
DEBUG_OUTPUT_PREFIX="$ROOT_DIR/debug/log"
```

```bash
# Running Config for Dataflow
RUNNER=DataflowRunner
JOB_NAME=text-analysis
MACHINE_TYPE=n1-highmem-2
```

```bash
# Remove Root directory before running dataflow job.
gsutil rm -r $ROOT_DIR
```


```bash
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
```

![alt text](http://url/to/img.png)
