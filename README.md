# TensorFlow Estimator APIs Tutorials - TensorFlow v1.4

<img src="images/exp-api2.png" width="1400" hight="400">

## The tutorials use the TF estimator APIs to cover:

* Various ML tasks, currently covering:
  * Classification
  * Regression
  * Clustering (k-means)
  * Time-series Analysis (AR Models)
  * Dimensionality Reduction (Autoencoding)
  * Sequence Models (RNN and LSTMs)
  * Image Analysis (CNN for Image Classification)
  *  How to use **canned estimators**  to train ML models.

*  How to implement **custom estimators** (model_fn & EstimatorSpec).

*  A standard **metadata-driven** approach to build the model **feature_column**(s) (including numeric features as well as categorical features with 1) vocabulary, 2) hash bucket, and 3) identity.

*  Data **input pipelines** (input_fn) using: 
 * tf.estimator.inputs.**pandas_input_fn**, 
 * tf.train.**string_input_producer**, and 
 * tf.data.**Dataset** APIs to read both **.csv** and **.tfrecords** (tf.example) data files
 * tf.contrib.timeseries.**RandomWindowInputFn** and **WholeDatasetInputFn** for time-series data
 * Feature **preprocessing** and **creation** as part of reading data (input_fn), for example, sin, sqrt, square, log, boolean comparisons, euclidean distance, etc.

*  A standard approach to prepare **wide** (sparse) and **deep** (dense) feature_column(s) for Wide and Deep **DNN Liner Combined Models**

*  The use of **normalizer_fn** in numeric_column() to **scale** the numeric features using pre-computed statistics (for Min-Max or Standard scaling)

* The use of **weight_column** in the canned estimators

* Implicit **Feature Engineering** as part of defining feature_colum(s), including:
  * crossing, 
  * clipping,
  * embedding,
  * indicators (categorical features), and
  * bucketization
  *  How to use the  tf.contrib.learn.**experiment** APIs to train, evaluate, and export models

* Howe to use the tf.estimator.**train_and_evaluate** function (along with trainSpec & evalSpec) train, evaluate, and export models

* How to **serve** exported model (export_savedmodel) using **csv** and **json** inputs

## Coming Soon:
* Early-stopping implementation
* DynamicRnnEstimator and the use of variable-length sequences
* Collaborative Filtering for Recommendation Models
* Text Analysis (Text Classification, Topic Models, Word/Doc embedding, etc.)
* tf.Transform to preprocessing and feature engineering
* keras examples



