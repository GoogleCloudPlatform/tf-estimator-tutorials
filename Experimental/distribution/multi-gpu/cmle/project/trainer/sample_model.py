# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


def _conv(x,kernel, name, log=False):
    with tf.variable_scope(name):
        W = tf.get_variable(initializer=tf.truncated_normal(shape=kernel, stddev=0.01), name='W')
        b = tf.get_variable(initializer=tf.constant(0.0, shape=[kernel[3]]), name='b')
        conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
        activation = tf.nn.relu(tf.add(conv,b))
        pool = tf.nn.max_pool(activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        if log == True:
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", activation)
        return pool


def _dense(x,size_in,size_out,name,relu=False,log=False):
    with tf.variable_scope(name):
        flat = tf.reshape(x, [-1, size_in])
        W = tf.get_variable(initializer=tf.truncated_normal([size_in,size_out], stddev=0.1), name='W')
        b = tf.get_variable(initializer=tf.constant(0.0, shape=[size_out]), name='b')
        activation = tf.add(tf.matmul(flat, W), b)
        if relu==True:
            activation = tf.nn.relu(activation)
        if log==True:
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", activation)
        return activation
    

def _model(features, mode, params):
    input_layer = tf.reshape(features, [-1, 32, 32, 3])
    conv1 = _conv(input_layer, kernel=[5,5,3,128], name='conv1', log=params['log'])
    conv2 = _conv(conv1, kernel=[5,5,128,128], name='conv2', log=params['log'])
    conv3 = _conv(conv2, kernel=[3,3,128,256], name='conv3', log=params['log'])
    conv4 = _conv(conv3, kernel=[3,3,256,512], name='conv4', log=params['log'])
    dense = _dense(conv4, size_in=2*2*512, size_out=params['dense_units'],
                   name='Dense', relu=True, log=params['log'])
    
    if mode==tf.estimator.ModeKeys.TRAIN:
        dense = tf.nn.dropout(dense, params['drop_out'])
        
    logits = _dense(dense, size_in=params['dense_units'],
                    size_out=10, name='Output', relu=False, log=params['log'])
    return logits


def model_fn(features, labels, mode, params):
    logits = _model(features, mode, params)
    predictions = {"logits": logits,
                   "classes": tf.argmax(input=logits,axis=1),
                   "probabilities": tf.nn.softmax(logits,name='softmax')}
    export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
    
    if (mode==tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(params['learning_rate'],
                                                   tf.train.get_global_step(),
                                                   decay_steps=100000,
                                                   decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        tf.summary.scalar('learning_rate', learning_rate)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))
        metrics = {'accuracy':accuracy}
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss, eval_metric_ops=metrics)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs)
