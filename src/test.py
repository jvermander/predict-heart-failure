# m , n
import util as ut
import neuralnets as nn

import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import numpy as np
import pandas as pd

import sys
import time

np.set_printoptions(edgeitems=5)
np.set_printoptions(linewidth=300)

def main( argv ):
  x_train, y_train, x_cv, y_cv, x_test, y_test = get_dataset()
  x_train, x_cv, x_test, mean, std = ut.normalize_dataset(x_train, x_cv, x_test)

  model = keras.Sequential(
    [
      keras.Input(shape=(12,)),
      layers.Dense(5, activation='sigmoid', name='layer1'),
      layers.Dense(5, activation='sigmoid', name='layer2'),
      layers.Dense(1, activation='sigmoid', name='output'),
    ]
  )
  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
  model.fit(x_train, y_train.reshape(-1, 1), batch_size=x_train.shape[0], epochs=5000)


  y = np.round(model(x_train)).reshape(-1)
  
  acc = np.mean(y == y_train)
  erridx = np.where(y != y_train)[0]
  err = erridx.shape[0] / y.shape[0]
  print(acc)
  print(erridx)
  print(err)

  print(y[erridx])
  print(y_train[erridx])

  y = np.round(model(x_cv)).reshape(-1)

  acc = np.mean(y == y_cv)
  erridx = np.where(y != y_cv)[0]
  err = erridx.shape[0] / y.shape[0]
  print(acc)
  print(erridx)
  print(err)

  print(y[erridx])
  print(y_cv[erridx])




  # inputs = keras.Input(shape=x_train.shape[1])
  # hidden = layers.Dense(5, activation='sigmoid')(inputs)
  # hidden = layers.Dense(5, activation='sigmoid')(hidden)
  # outputs = layers.Dense(1)(hidden)
  
  # model = keras.Model(inputs=inputs, outputs=outputs, name='mymodel')
  # model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3), loss='binary_crossentropy')
  # history = model.fit(x_train, y_train, batch_size=x_train.shape[0], epochs=2, validation_data=(x_cv, y_cv))
  # print(model(x_train))

def get_dataset():
  data = ut.read_csv('data/heart.csv')
  np.random.seed(1)
  data = np.random.permutation(data)

  examples = data[:, :-1]
  labels = data[:, -1]

  m = examples.shape[0]
  m_train = int(np.round(m * 0.60))
  m_cv = int(np.round(m * 0.20))

  x_train = examples[:m_train, :]
  y_train = labels[:m_train]

  x_cv = examples[m_train : m_train + m_cv, :]
  y_cv = labels[m_train : m_train + m_cv]

  x_test = examples[m_train + m_cv:, :]
  y_test = labels[m_train + m_cv:]
  
  return x_train, y_train, x_cv, y_cv, x_test, y_test

if(__name__ == "__main__"):
  main(sys.argv[1:])


