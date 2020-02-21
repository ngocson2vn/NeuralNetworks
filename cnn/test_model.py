import mnist
import numpy as np
from cnn import CNN
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(edgeitems=100, linewidth=200000)

test_images = (mnist.test_images() / 255) - 0.5
test_labels = mnist.test_labels()

with open("model.bin", "rb") as model:
  cnn = pickle.load(model)
  for img, label in zip(train_images[1000:2001], train_labels[1000:2001]):
    O = cnn.feedforward(img)
    prediction = np.argmax(O)
    print("Label: {}, prediction: {}, probability: {}".format(label, prediction, 100 * O[prediction]))
