# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

path = "./data/"
extension = ".png"

def resize_img(img_file, dim):
    spectrogram = cv2.imread(img_file)
    # resize image
    return cv2.resize(spectrogram, dim)

def get_next_batch(dataset, batch_size, dim):
    sounds = dataset[np.random.choice(dataset.shape[0], batch_size, replace=False), :]
    images = [resize_img(path+ os.path.splitext(file[0])[0]+extension, dim ) for file in sounds]
    X_batch, y_batch = np.asarray(images), np.array(sounds[:, [1]])
    
    return X_batch.reshape(batch_size,-batch_size), y_batch.reshape(-1)