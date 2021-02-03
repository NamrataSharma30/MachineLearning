# -*- coding: utf-8 -*-
"""FashionMNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cq9ZNtY83j7eNOaS59gIkYfsYG4nCUK7
"""

import tensorflow as tf
import numpy as np
import os
import distutils
from random import randrange

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# add empty color dimension and scale to [0,1]
x_train = np.expand_dims(x_train, -1)/255
x_test = np.expand_dims(x_test, -1)/255

n_train = 1000
np.random.seed(2020)
ind = np.random.permutation(x_train.shape[0])
x_unlabeled =  x_train[ind[n_train:]]


x_train =  x_train[ind[:n_train]]
y_train = y_train[ind[:n_train]]

xs = np.zeros_like(x_train)

x_train1 = np.roll(x_train,-1)
x_train2 = np.roll(x_train1,-1)
x_train3 = np.roll(x_train2,-1)
x_train4 = np.roll(x_train3,-1)
x_train5 = np.roll(x_train4,-1)
x_train6 = np.roll(x_train5,-1)
x_train7 = np.roll(x_train6,-1)
x_train8 = np.roll(x_train7,-1)
x_train_new = np.concatenate((x_train,x_train1,x_train2,
                              x_train3,x_train4,x_train5,
                              x_train6,x_train7,x_train8))


# x_test1 = np.roll(x_test,-1)
# x_test2 = np.roll(x_test1,-1)
# x_test3 = np.roll(x_test2,-1)
# x_test4 = np.roll(x_test3,-1)
# x_test5 = np.roll(x_test4,-1)
# x_test6 = np.roll(x_test5,-1)
# x_test7 = np.roll(x_test6,-1)
# x_test8 = np.roll(x_test7,-1)
# x_test_new = np.concatenate((x_test,x_test1,x_test2,
#                               x_test3,x_test4,x_test5,
#                               x_test6,x_test7,x_test8))


# def synthetic_classification(x1,x2):
#   # x1 and x2 are assumed to belong to the same class
#   w = randrange(10) # w is a random number between 0 and 1 
#   return x1*w + x2*(1-w)

# temp = []
# for i in range(x_train.shape[0]):
#   for j in range(x_train.shape[0]):
#     if y_train[i] == y_train[j] :
#       xs = synthetic_classification(x_train[i], x_train[j])
#       temp.append(xs)
#       #np.append(x_train[i], xs, axis=0)

# temp_array = np.array(temp)
# x_train_new = np.concatenate((x_train, temp_array[:1000], temp_array[1001:2000], temp_array[2001:3000], 
#                               temp_array[3001:4000], 
#                               temp_array[4001:5000], temp_array[5001:6000], temp_array[6001:7000], 
#                               temp_array[7001:8000], temp_array[8001:9000], temp_array[9000:10008]))

print(x_test_new.shape)



# Convert y to one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)



# y_test = np.concatenate((y_test, y_test, y_test, y_test, y_test, y_test,
#                               y_test, y_test, y_test))

import tensorflow as tf
import os
tf.keras.backend.clear_session()

resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.TPUStrategy(resolver)

import tensorflow as tf
from tensorflow.keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
import os
import distutils
from keras.layers import Dropout

def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu'))
  model.add(BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
  model.add(BatchNormalization())
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(64,activation ='relu'))
  model.add(Dropout(0.5))
  model.add(tf.keras.layers.Dense(10,activation = 'softmax'))
  return model

model = create_model()
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

temp = []
temp_x = []
for i in range(20):
  model.fit(x_train, y_train)
  y = model.predict(x_unlabeled)
  c = model.predict_proba(x_unlabeled)
  highest = np.argmax(c, axis=0)
  # temp.append(y[highest])
  # temp_x.append(x_test[highest])
  np.concatenate((y_train[i], y[highest][i]))
  ind_ = np.where(y[highest])
  x_train[ind_] = x_unlabeled[ind_]

# yh = np.array(temp)
# xh = np.array(temp_x)

#print(yh.shape, xh.shape)


history = model.fit(x_train, y_train, batch_size=98, epochs=20, validation_data=(x_test, y_test))