import mnist
import numpy as np
from cnn import CNN
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(edgeitems=100, linewidth=200000)

cnn = CNN(6, 12)

# print("INITIAL HYPERPARAMETERS:")
# print("cnn.k1:")
# print(cnn.k1)
# print("cnn.b1:")
# print(cnn.b1)
# print("cnn.k2:")
# print(cnn.k2)
# print("cnn.b2:")
# print(cnn.b2)
# print()

train_images = (mnist.train_images() / 255) - 0.5
train_labels = mnist.train_labels()

test_images = (mnist.train_images() / 255) - 0.5
test_labels = mnist.train_labels()

stats = cnn.train(train_images, train_labels, test_images, test_labels, 10, 0.005)
epochs = stats[0]
min_losses = stats[1]
avg_losses = stats[2]
max_losses = stats[3]
accuracies = stats[4]

with open("model_full.bin", "wb") as f:
  pickle.dump(cnn, f)

plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(epochs, min_losses, label="Min loss")
plt.plot(epochs, avg_losses, label="Avg loss")
plt.plot(epochs, max_losses, label="Max loss")
plt.plot(epochs, accuracies, label="Accuracy")
plt.legend(loc="center")
plt.show()
