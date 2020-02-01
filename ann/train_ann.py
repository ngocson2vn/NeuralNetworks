from ann import NeuralNetwork
import matplotlib.pyplot as plt
import pickle

ground_truth_dataset = [
  [15, 3, 1],
  [10, 5, 1],
  [20, 1, 1],
  [1,  5, 0],
  [5,  0, 0],
  [30, 0, 1],
  [2,  1, 0],
  [5,  5, 1],
  [7, 10, 0],
  [25, 6, 1]
]

n = NeuralNetwork()

stats = n.train(ground_truth_dataset, 100000, 0.001)
epochs = stats[0]
min_losses = stats[1]
avg_losses = stats[2]
max_losses = stats[3]

with open("model.bin", "wb") as f:
  pickle.dump(n, f)

plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(epochs, min_losses, label="Min loss")
plt.plot(epochs, avg_losses, label="Avg loss")
plt.plot(epochs, max_losses, label="Max loss")
plt.legend(loc="center")
plt.show()
