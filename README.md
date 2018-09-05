# TensorFlow Estimator APIs Tutorials

## Setup
Please follow the directions in INSTALL if you need help setting up your environment.

## Theses tutorials use the TF estimator APIs to cover:

* Various ML tasks, currently covering:
  * Classification
  * Regression
  * Clustering (k-means)
  * Time-series Analysis (AR Models)
  * Dimensionality Reduction (Autoencoding)
  * Sequence Models (RNN and LSTMs)
  * Image Analysis (CNN for Image Classification)
  * Text Analysis (Text Classification with embeddings, CNN, and RNN)
*  How to use **canned estimators**  to train ML models.

* How to use **tf.Transform** for preprocessing and feature engineering (TF v1.7)

* Implement **TensorFlow Model Analysis (TFMA)** to assess the quality of the mode (TF v1.7)

* How to use **tf.Hub** text feature column embeddings (TF v1.7)

* How to implement **custom estimators** (model_fn & EstimatorSpec).

* A standard **metadata-driven** approach to build the model **feature_column**(s) including:
  * **numerical** features
  * **categorical** features with **vocabulary**,
  * **categorical** features **hash bucket**, and
  * **categorical** features with **identity**

* Data **input pipelines** (input_fn) using:
  * tf.estimator.inputs.**pandas_input_fn**,
  * tf.train.**string_input_producer**, and
  * tf.data.**Dataset** APIs to read both **.csv** and **.tfrecords** (tf.example) data files
  * tf.contrib.timeseries.**RandomWindowInputFn** and **WholeDatasetInputFn** for time-series data
  * Feature **preprocessing** and **creation** as part of reading data (input_fn), for example, sin, sqrt, polynomial expansion, fourier transform, log, boolean comparisons, euclidean distance, custom formulas, etc.

* A standard approach to prepare **wide** (sparse) and **deep** (dense) feature_column(s) for Wide and Deep **DNN Liner Combined Models**

* The use of **normalizer_fn** in numeric_column() to **scale** the numeric features using pre-computed statistics (for Min-Max or Standard scaling)

* The use of **weight_column** in the canned estimators, as well as in **loss function** in custom estimators.

* Implicit **Feature Engineering** as part of defining feature_colum(s), including:
  * crossing
  * embedding
  * indicators (encoding categorical features), and
  * bucketization

*  How to use the  tf.contrib.learn.**experiment** APIs to train, evaluate, and export models

* Howe to use the tf.estimator.**train_and_evaluate** function (along with trainSpec & evalSpec) train, evaluate, and export models

* How to use **tf.train.exponential_decay** function as a learning rate scheduler

* How to **serve** exported model (export_savedmodel) using **csv** and **json** inputs

## Coming Soon:
* Early-stopping implementation
* DynamicRnnEstimator and the use of variable-length sequences
* Collaborative Filtering for Recommendation Models
* Text Analysis (Topic Models, etc.)
* Keras examples



