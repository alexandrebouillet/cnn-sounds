# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

def resize_img(img_file, dim):
    spectrogram = cv2.imread(img_file)
    # resize image
    return cv2.resize(spectrogram, dim)

def get_next_batch(dataset, batch_size):
    sounds = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False), :]
    images = [cv2.imread("./data/"+ os.path.splitext(file[0])[0]+".png") for file in sounds]
    X_batch, y_batch = images, sounds[:, [1]]
    print(type(sounds))
    
    return X_batch, y_batch