import numpy as np
import pandas as pd

def read_csv( path ):
    data = pd.read_csv(path).values
    return data

def normalize( X ):
  mu = np.zeros((X.shape[1],))
  std = np.zeros((X.shape[1],))
  X_norm = np.zeros(X.shape)

  mu = np.mean(X, axis=0)
  std = np.std(X, axis=0)
  X_norm = (X - mu) / std
  
  return X_norm, mu, std

def normalize_dataset( X, X_cv, X_test ):
    X_norm, mu, std = normalize(X)
    X_cv_norm = (X_cv - mu) / std
    X_test_norm = (X_test - mu) / std

    return X_norm, X_cv_norm, X_test_norm, mu, std