import mnist
import numpy as np
from cnn import CNN
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(edgeitems=100, linewidth=200000)

train_images = mnist.train_images()[:1]
train_labels = mnist.train_labels()[:1]
n = CNN(6, 12)
O = n.feedforward(train_images[0])
print(O)
