import numpy as np
import cvnn.layers as complex_layers
import tensorflow as tf

input_shape = (4, 28, 28, 3)
x = tf.cast(tf.random.normal(input_shape), tf.complex64)

model = tf.keras.models.Sequential()
model.add(complex_layers.ComplexInput(input_shape=input_shape[1:]))
model.add(complex_layers.ComplexFlatten())
model.add(complex_layers.ComplexDense(units=64, activation='cart_relu'))
model.add(complex_layers.ComplexDense(units=10, activation='linear'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y = model(x)