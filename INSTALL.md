# Setup

## On GCP

Follow these steps to setup a VM instance running jupyter on GCP, with TensorFlow and other dependencies installed.


Create environment variables:

```
export HOST_NAME=`whoami`-0
export PROJECT=[your GCP project id]
export IMAGE_PROJECT=deeplearning-platform-release
export ZONE=[your preferred zone, e.g. europe-west4-a]
```

### (optional) Request quota for P100s on your project

https://cloud.google.com/compute/quotas

### If you have P100 quota and wish to use GPUs

```
export IMAGE_NAME="tf-latest-cu92"
```

```
gcloud beta compute instances create ${HOST_NAME} \
 --project=${PROJECT} \
 --zone=${ZONE} \
 --machine-type=n1-standard-4 \
 --maintenance-policy=TERMINATE \
 --accelerator=type=nvidia-tesla-p100,count=1 \
 --metadata='install-nvidia-driver=True' \
 --image-family=${IMAGE_NAME} \
 --image-project=${IMAGE_PROJECT} \
 --boot-disk-size=200GB \
 --boot-disk-type=pd-standard \
 --boot-disk-device-name=${HOST_NAME} \
 --scopes=https://www.googleapis.com/auth/cloud-platform
```

### If you don't have quota, or don't want to use GPUs

```
export IMAGE_NAME="tf-latest-cpu"
```

```
gcloud beta compute instances create ${HOST_NAME} \
 --project=${PROJECT} \
 --zone=${ZONE} \
 --machine-type=n1-standard-4 \
 --maintenance-policy=TERMINATE \
 --image-family=${IMAGE_NAME} \
 --image-project=${IMAGE_PROJECT} \
 --boot-disk-size=200GB \
 --boot-disk-type=pd-standard \
 --boot-disk-device-name=${HOST_NAME} \
 --scopes=https://www.googleapis.com/auth/cloud-platform
```


### SSH into your instance, creating a local port for Jupyter

```
gcloud compute ssh $HOST_NAME --project=$PROJECT --zone=$ZONE -- -L 8080:localhost:8080
```

### Access JupyterLab running locally

Go to http://localhost:8080

Open a Terminal via the Launcher




