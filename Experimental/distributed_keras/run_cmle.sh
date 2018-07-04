%bash

REGION=us-central1
JOBDIR=gs://your-bucket-name/keras-mnist/keras-model-cmle/job
JOBNAME=sm_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME

gcloud ml-engine jobs submit training $JOBNAME \
   --region=$REGION \
   --module-name=trainer.main \
   --package-path="$(pwd)/trainer" \
   --job-dir=$JOBDIR \
   --staging-bucket=gs://your-bucket-name \
   --config=config.yaml \
   --runtime-version=1.7 \
   -- \
   --data_dir=gs://your-bucket-name/keras-mnist/data \
   --model_dir=gs://your-bucket-name/keras-mnist/keras-model-cmle \
   --export_dir=gs://your-bucket-name/keras-mnist/keras-model-cmle/export \
   --train_steps=100000
