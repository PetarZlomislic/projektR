import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import models

from cvnn import layers

# CONVNET
# acc: ~90%
CONVNET = models.Sequential()
CONVNET.add(layers.ComplexInput((20, 1024, 1)))
CONVNET.add(layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
CONVNET.add(layers.ComplexMaxPooling2D((2, 2)))
CONVNET.add(layers.ComplexConv2D(32, (3, 3), activation='cart_relu'))
CONVNET.add(layers.ComplexMaxPooling2D((2, 2)))
CONVNET.add(layers.ComplexConv2D(64, (3, 3), activation='cart_relu'))
CONVNET.add(layers.ComplexFlatten())
CONVNET.add(layers.ComplexDense(64, activation='cart_relu'))
CONVNET.add(layers.ComplexDense(3, activation='softmax_real_with_abs'))

# COMMON LINEAR NEURAL NETWORK
# acc: ~20% overfit
CLNN = tf.keras.models.Sequential([
        layers.ComplexInput(input_shape=(20, 1024, 1)),
        layers.ComplexFlatten(),
        layers.ComplexDense(1024, activation='cart_relu'),
        layers.ComplexDense(1024, activation='cart_relu'),
        layers.ComplexDense(1024, activation='cart_relu'),
        layers.ComplexDense(3, activation='softmax_real_with_abs')
    ])