import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  if x < 0:
    return 0
  else:
    return x

def d_relu(x):
  if x <= 0:
    return 0
  else:
    return 1

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

def loss(y_preds):
  losses = -np.log(y_preds)
  return losses.min(), losses.mean(), losses.max()
