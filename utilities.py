# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import multiprocessing as mp
from functools import partial
import pandas as pd
from sklearn import preprocessing

path = "./data/audio_train/"
extension = ".png"

def resize_img(img_file, dim):
    spectrogram = cv2.imread(path+ os.path.splitext(img_file[0])[0]+extension, cv2.IMREAD_COLOR)
    norm_image = cv2.normalize(spectrogram, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # resize image
    image = cv2.resize(norm_image, dim)
    return image

def get_next_batch(dataset, batch_size, dim):
    sounds = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False), :]
    pool = mp.Pool()
    prod_x=partial(resize_img, dim=dim)
    X_batch = np.array(pool.map(prod_x, sounds))
    y_batch = np.array(sounds[:, [1]], dtype=np.int32)
    pool.close()
    
    return X_batch.reshape(batch_size,-batch_size), y_batch.reshape(-1)

def get_dataset_test(dataset, batch_size, dim):
    dataset = pd.read_csv("./all/test_post_competition.csv").iloc[:, [0,1]]
    dataset = dataset[dataset.label != 'None']
    le = preprocessing.LabelEncoder()
    label_encoded = le.fit_transform(dataset["label"].reshape(-1))
    dataset["label"] = label_encoded
    pool = mp.Pool()
    prod_x=partial(resize_img, dim=dim)
    X_batch = np.array(pool.map(prod_x, dataset.fname))
    pool.close()
    y_batch = np.array(dataset.label, dtype=np.int32)
    
    return X_batch.reshape(dataset.shape[0],-dataset.shape[0]), y_batch.reshape(-1)