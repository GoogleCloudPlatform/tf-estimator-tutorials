# TensorFlow Estimator APIs Tutorials - TensorFlow v1.4

<img src="images/exp-api2.png" width="1400" hight="400">

## The tutorials use the TF estimator APIs to cover:

* Canned Estimators (Regression, Classification, Clustering, Time Series, Autoencoding, etc.)
*  Metadata-driven approach to build the model features (work with numeric and categorical input attributes)
*  Wide & deep Models - (handling dense and sparse feature columns)
*  Scaling input feautures using the normalizer_fn in numeric_column()
*  Feature Engineering (crossing, clipping, embedding, and bucketization, as well as custom logic during data input)
*  Experiment APIs (tf.contrib.learn.experiment) and tf.estimator.train_and_evaluate (trainSpec & evalSpec)
*  Data input (tf.estimator.inputs.pandas_input_fn, tf.train.string_input_producer, and tf.data.Dataset APIs)
*  Work with .csv and .tfrecords (tf.example) data files
*  Early Stopping (SessionRunHooks)
*  Serving (export_savedmodel)
*  Custom Estimators (model_fn & EstimatorSpec)
