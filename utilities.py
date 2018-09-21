# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import multiprocessing as mp
from functools import partial
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf


extension = ".png"

def resize_img(img_file, dim, path):
    spectrogram = cv2.imread(path+ os.path.splitext(img_file[0])[0]+extension, cv2.IMREAD_COLOR)
    norm_image = cv2.normalize(spectrogram, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # resize image
    image = cv2.resize(norm_image, dim)
    return image

def get_next_batch(dataset, batch_size, dim):
    sounds = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False), :]
    path = "./data/audio_train/"
    pool = mp.Pool()
    prod_x=partial(resize_img, dim=dim, path=path)
    X_batch = np.array(pool.map(prod_x, sounds))
    y_batch = np.array(sounds[:, [1]], dtype=np.int32)
    pool.close()
    
    return X_batch.reshape(batch_size,-batch_size), y_batch.reshape(-1)

def get_data_test(dataset, dim):
    path = "./data/audio_train/"
    pool = mp.Pool()
    prod_x=partial(resize_img, dim=dim, path=path)
    X_batch = np.array(pool.map(prod_x, dataset))
    y_batch = np.array(dataset[:, [1]], dtype=np.int32)
    pool.close()
    
    return X_batch.reshape(dataset.shape[0],-dataset.shape[0]), y_batch.reshape(-1)

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

