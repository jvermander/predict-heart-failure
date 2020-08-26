# m , n

import util as ut
import neuralnets as nn

import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import time

np.random.seed(1)
tf.random.set_seed(1)

np.set_printoptions(edgeitems=5)
np.set_printoptions(linewidth=300)

def main( argv ):
  x_train, y_train, x_cv, y_cv, x_test, y_test = get_dataset()
  x_train, x_cv, x_test, mean, std = ut.normalize_dataset(x_train, x_cv, x_test)

  f = open('out.txt', 'a')
  units = [4, 6, 8, 12, 16, 18, 24, 32, 36, 64, 128, 256]
  l = 0.0000
  alpha = 0.001
  for i in range(len(units)):
    for j in range(len(units)):
      np.random.seed(1)
      tf.random.set_seed(1)
      model = keras.Sequential([keras.Input(shape=(12,))])
      model.add(layers.Dense(units[i], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=l)))
      model.add(layers.Dense(units[j], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=l)))
      model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=l)))
      model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=alpha), 
                    metrics=['binary_accuracy',
                              keras.metrics.Precision(), 
                              keras.metrics.Recall(),
                              'AUC'])
      model.summary()
      history = model.fit(x_train, y_train.reshape(-1, 1), batch_size=x_train.shape[0], epochs=2048, validation_data=(x_cv, y_cv))
      print(history.history.keys())
      acc = max(history.history['val_binary_accuracy'])
      epoch = history.history['val_binary_accuracy'].index(acc)
      prec = history.history['val_precision'][epoch]
      recall = history.history['val_recall'][epoch]
      f1 = 2 * prec * recall / (prec + recall)
      auc = history.history['val_auc'][epoch]
      loss = history.history['val_loss'][epoch]
      tacc = history.history['binary_accuracy'][epoch]
      
      prec_all = np.array(history.history['val_precision'])
      recall_all = np.array(history.history['val_recall'])
      f1_all = 2 * (prec_all * recall_all) / (prec_all + recall_all)
      f1_epoch = np.nanargmax(f1_all)
      f1_acc = history.history['val_binary_accuracy'][f1_epoch]
      f1_max = f1_all[f1_epoch]

      for layer in model.layers:
        print(layer.output_shape[1], "|", end='', sep='', file=f, flush=True)
      print(" %.4f %d: %.3f\t%.3f %.3f %.3f\t%.3f (%.3f)\t%.3f %.3f %.3f\n" 
        % (l, epoch, acc, f1, prec, recall, f1_max, f1_acc, auc, loss, tacc), file=f, flush=True)
      
      # plt.plot(history.history['loss'], label="Training Loss", color='red')
      # plt.plot(history.history['binary_accuracy'], label="Training Accuracy", color='orange')
      # plt.plot(history.history['val_loss'], label="Validation Loss", color='blue')
      # plt.plot(history.history['val_binary_accuracy'], label="Validation Accuracy", color='purple')
      # # plt.ylabel("Loss")
      # plt.xlabel("Number of Epochs")
      # plt.legend(loc="upper left")
      # plt.show()

      model.pop()
      model.pop()
      keras.backend.clear_session()

  for i in range(len(units)):
    for j in range(len(units)):
      for k in range(len(units)):
        np.random.seed(1)
        tf.random.set_seed(1)
        model = keras.Sequential([keras.Input(shape=(12,))])
        model.add(layers.Dense(units[i], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=l)))
        model.add(layers.Dense(units[j], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=l)))
        model.add(layers.Dense(units[k], activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=l)))
        model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2=l)))
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=alpha), 
                      metrics=['binary_accuracy',
                                keras.metrics.Precision(), 
                                keras.metrics.Recall(),
                                'AUC'])
        model.summary()
        history = model.fit(x_train, y_train.reshape(-1, 1), batch_size=x_train.shape[0], epochs=2048, validation_data=(x_cv, y_cv))
        print(history.history.keys())
        acc = max(history.history['val_binary_accuracy'])
        epoch = history.history['val_binary_accuracy'].index(acc)
        prec = history.history['val_precision'][epoch]
        recall = history.history['val_recall'][epoch]
        f1 = 2 * prec * recall / (prec + recall)
        auc = history.history['val_auc'][epoch]
        loss = history.history['val_loss'][epoch]
        tacc = history.history['binary_accuracy'][epoch]
        
        prec_all = np.array(history.history['val_precision'])
        recall_all = np.array(history.history['val_recall'])
        f1_all = 2 * (prec_all * recall_all) / (prec_all + recall_all)
        f1_epoch = np.nanargmax(f1_all)
        f1_acc = history.history['val_binary_accuracy'][f1_epoch]
        f1_max = f1_all[f1_epoch]

        for layer in model.layers:
          print(layer.output_shape[1], "|", end='', sep='', file=f, flush=True)
        print(" %.4f %d: %.3f\t%.3f %.3f %.3f\t%.3f (%.3f)\t%.3f %.3f %.3f\n" 
          % (l, epoch, acc, f1, prec, recall, f1_max, f1_acc, auc, loss, tacc), file=f, flush=True)
        
        # plt.plot(history.history['loss'], label="Training Loss", color='red')
        # plt.plot(history.history['binary_accuracy'], label="Training Accuracy", color='orange')
        # plt.plot(history.history['val_loss'], label="Validation Loss", color='blue')
        # plt.plot(history.history['val_binary_accuracy'], label="Validation Accuracy", color='purple')
        # # plt.ylabel("Loss")
        # plt.xlabel("Number of Epochs")
        # plt.legend(loc="upper left")
        # plt.show()

        model.pop()
        model.pop()
        keras.backend.clear_session()

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


