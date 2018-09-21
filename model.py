#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 09:52:06 2018

@author: abouillet
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utilities import resize_img, get_next_batch, get_dataset_test
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("./data/train.csv").iloc[:]
le = preprocessing.LabelEncoder()
dataset["label"] = le.fit_transform(dataset["label"])
X_train, X_test, y_train, y_test = train_test_split(dataset["fname"], dataset["label"], test_size=0.33)
del dataset["manually_verified"]
data = np.array(pd.concat([X_train, y_train], axis=1))
data_test = np.array(pd.concat([X_test, y_test], axis=1))


num_exemples = dataset.shape[0]
height = 28
width = 28
channels = 3
dim = (width, height)
n_inputs = (height * width) * channels

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5

n_outputs = 41


with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)
    

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

batch_size = 64
n_epochs = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(num_exemples // batch_size):
            X_batch, y_batch = get_next_batch(data, batch_size, dim)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        X_batch_test, y_batch_test = get_next_batch(data_test,batch_size, dim)
        acc_test = accuracy.eval(feed_dict={X: X_batch_test, y: y_batch_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path = saver.save(sess, "./sound_model")
        
