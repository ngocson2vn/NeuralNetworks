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

# Preset weights and biases
n.w1 = -0.06958319062984783
n.w2 = -0.13036519749456643
n.w3 = -1.0571412909409637
n.w4 = 0.135581608179992
n.w5 = 0.5626741222756912
n.w6 = -0.010929699015683102
n.b1 = -0.83347780239002
n.b2 = 0.27420776870407937
n.b3 = 0.6237767133439941

stats = n.train(ground_truth_dataset, 500001, 0.001)
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
