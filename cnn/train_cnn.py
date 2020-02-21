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

test_images = (mnist.test_images() / 255) - 0.5
test_labels = mnist.test_labels()

stats = cnn.train(train_images[:1000], train_labels[:1000], test_images[:100], test_labels[:100], 100, 0.005)
epochs = stats[0]
min_losses = stats[1]
avg_losses = stats[2]
max_losses = stats[3]
accuracies = stats[4]

with open("model.bin", "wb") as f:
  pickle.dump(cnn, f)

plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(epochs, min_losses, label="Min loss")
plt.plot(epochs, avg_losses, label="Avg loss")
plt.plot(epochs, max_losses, label="Max loss")
plt.plot(epochs, accuracies, label="Accuracy")
plt.legend(loc="center")
plt.show()
