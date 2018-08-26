# -*- coding: utf-8 -*-
import cv2
import random
import os

def resize_img(img_file, dim):
    spectrogram = cv2.imread(img_file)

    # resize image
    return cv2.resize(spectrogram, dim)

def get_next_batch(dataset, batch_size):
    sounds = random.sample(dataset, batch_size)
    images = [cv2.imread(os.path.splitext("./data/"+file+".png")[0]) for file in sounds]
    print(images)
    X_batch, y_batch = []
    
    return X_batch, y_batch