import numpy as np

def relu(x):
  if x < 0:
    return 0
  else:
    return x

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

def loss(y_pred):
  return -np.log(y_pred)
