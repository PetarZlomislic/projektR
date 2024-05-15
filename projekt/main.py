import os, timeit
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import models

from cvnn import layers

SEED = 20
BATCH_SIZE = 64
PATH = "C:/Users/zlomi/OneDrive/Radna površina/projektR/data/complex_npy_segments"

def read_files(path):
    Xs = []
    ys = []
    for file in os.listdir(path):
        Xs.append(np.load(path+"/"+file))
        ys.append(file.split("_")[1][0])
    return Xs, ys

labels = ["AR", "AS", "GR", "GS", "PR", "PS"]
labels_material = ["A", "G", "P"]

X, y = read_files(PATH)
X = tf.convert_to_tensor(X)
X = tf.expand_dims(X, -1)
y = tf.convert_to_tensor(list(map((lambda x: labels_material.index(x)), y)))

#tf.random.set_seed(SEED)

model = models.Sequential()
model.add(layers.ComplexInput((20, 1024, 1)))
model.add(layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
model.add(layers.ComplexMaxPooling2D((2, 2)))
model.add(layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
model.add(layers.ComplexMaxPooling2D((2, 2)))
model.add(layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
model.add(layers.ComplexFlatten())
model.add(layers.ComplexDense(64, activation='cart_relu'))
model.add(layers.ComplexDense(3, activation='softmax_real_with_abs'))

'''
model = tf.keras.models.Sequential([
        layers.ComplexInput(input_shape=(20, 1024, 1)),
        layers.ComplexFlatten(),
        layers.ComplexDense(1024, activation='cart_relu'),
        layers.ComplexDense(1024, activation='cart_relu'),
        layers.ComplexDense(1024, activation='cart_relu'),
        layers.ComplexDense(3, activation='softmax_real_with_abs')
    ])
'''
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

start = timeit.default_timer()
model.fit(
    x=X,
    y=y,
    epochs=50,
    verbose=True, 
    shuffle=False,
    validation_split=0.2
)
stop = timeit.default_timer()