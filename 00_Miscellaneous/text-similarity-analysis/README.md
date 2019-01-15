# Text Semantic Similarity Analysis Pipeline

This is a Dataflow pipeline that reads article documents in Google Cloud Storage, extracts feature embeddings from the documents, and store those embeddings to BigQuery. After running the pipeline, you can easily search contextually similar documents based on a cosine distance between feature embeddings.

In the pipeline, documents are processed to extract each article’s title, topics, and content. The processing pipeline uses the “Universal Sentence Encoder” module in tf.hub to extract text embeddings for both the title and the content of each article read from the source documents. The title, topics, and content of each article, along with the extracted embeddings are stored in BigQuery. Having the articles, along with their embeddings, stored in BigQuery allow us to explore similar articles, using cosine similarity metric between embeddings of tiles and/or contents.

# How to run the pipeline

## Requirements

You need to have your [GCP Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects). You can use [Cloud Shell](https://cloud.google.com/shell/docs/quickstart) or [gcloud CLI](https://cloud.google.com/sdk/) to run all the commands in this guideline.

## Setup a project

Follow the [instruction](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and create a GCP project. 
Once created, enable the Dataflow API, BigQuery API in this [page](https://console.developers.google.com/apis/enabled). You can also find more details about enabling the [billing](https://cloud.google.com/billing/docs/how-to/modify-project?#enable-billing)

We recommend to use CloudShell from the GCP console to run the below commands. CloudShell starts with an environment already logged in to your account and set to the currently selected project. The following commands are required only in a workstation shell environment, they are not needed in the CloudShell. 

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project [your-project-id]
gcloud config set compute/zone us-central1-a
```

## Prepare a input data

You need to download reuters-21578 dataset from from [here](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection). After downloading reuters.tar.gz from the site, you need to type the following commands to store reuter dataset to Google Cloud Storage.

```bash
export BUCKET=gs://[your-bucket-name]

mkdir temp reuters
tar -zxvf reuters21578.tar.gz -C temp/
mv temp/*.sgm reuters/ && rm -rf temp
gsutil mb $BUCKET
gsutil -m cp -R reuters $BUCKET
```

## Setup python environment and sample code

Follow commands below to install required python packages and download a dataflow pipeline code.

```bash
git clone [this-repo]
cd [this-repo]/00_Miscellaneous/text-similarity-analysis

# Make sure you have python 2.7 environement
pip install -r requirements.txt
```

## Run the pipeline

Set running configurations for your Dataflow job. You will need GCE instances with high memory in Dataflow job because tf.hub module uses lots of memory that doesn't fit memory of default GCE instance.

```bash
# Running configurations for Dataflow
export PROJECT=[your-project-name]
export JOB_NAME=[your-dataflow-job-name]
export REGION=[your-preferred-region]
export RUNNER=DataflowRunner
export MACHINE_TYPE=n1-highmem-2
```

If you've followed the instruction in previous section, you should have reuter dataset in GCS. Set a file pattern of reuter dataset to FILE_PATTERN variable.

```bash
# A file pattern of reuter dataset located in GCS.
export FILE_PATTERN=$BUCKET/reuters/*.sgm
```

Note that you have to create BigQuery dataset before running Dataflow job. You should also set name of BigQuery dataset and table so Dataflow pipeline can output the feature embeddings to the right place in BigQuery.

```bash
# Information about output table in BigQuery.
export BQ_PROJECT=$PROJECT

# You must prepare dataset before running the pipeline, otherwise it will fail.
export BQ_DATASET=[your-bigquery-dataset-name]

# An output of the pipeline will be exported to this table.
export BQ_TABLE=[your-bigquery-table-name]
```

Next, you have to just run below commands. TF_EXPORT directory is where SavedModel file will be output by tf.transform. You can re-use it to get feature embeddings from documents later.

```bash
# A root directory.
export ROOT="$BUCKET/$JOB_NAME"

# Working directories for Dataflow jobs.
export DF_ROOT="$ROOT/dataflow"
export DF_STAGING="$DF_ROOT/staging"
export DF_TEMP="$DF_ROOT/temp"

# Working directories for tf.transform.
export TF_ROOT="$ROOT/transform"
export TF_TEMP="$TF_ROOT/temp"
export TF_EXPORT="$TF_ROOT/export"

# A directory where tfrecords data will be output.
export TFRECORD_OUTPUT_DIR="$ROOT/tfrecords"
export PIPELINE_LOG_PREFIX="$ROOT/log/output"
```

Before running the pipeline, you can remove previous working directory
with below command if you want.

```bash
gsutil rm -r $ROOT
```

Finally, you can run the pipeline with this command.

```bash
python etl/run_pipeline.py \
  --project=$PROJECT \
  --region=$REGION \
  --setup_file=$(pwd)/etl/setup.py \
  --job_name=$JOB_NAME \
  --runner=$RUNNER \
  --worker_machine_type=$MACHINE_TYPE \
  --file_pattern=$FILE_PATTERN \
  --bq_project=$BQ_PROJECT \
  --bq_dataset=$BQ_DATASET \
  --bq_table=$BQ_TABLE \
  --staging_location=$DF_STAGING \
  --temp_location=$DF_TEMP \
  --transform_temp_dir=$TF_TEMP \
  --transform_export_dir=$TF_EXPORT \
  --enable_tfrecord \
  --tfrecord_export_dir $TFRECORD_OUTPUT_DIR \
  --enable_debug \
  --debug_output_prefix=$PIPELINE_LOG_PREFIX
```
