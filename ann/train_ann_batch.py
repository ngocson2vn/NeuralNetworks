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
# n.w1 = 0.024739923179843564
# n.w2 = 0.782086203350603
# n.w3 = -0.7531968177831182
# n.w4 = -0.1328632433367266
# n.w5 = 0.7521920897144088
# n.w6 = -0.2568310157160393
# n.b1 = -0.5874532717161877
# n.b2 = -0.9232275728505895
# n.b3 = 0.75530633226724
n.w1 = 0.27985946828331204
n.w2 = 1.113212641433306
n.w3 = -1.8082799142509993
n.w4 = 1.572218879949482
n.w5 = 4.689412543076026
n.w6 = -4.6976243908676985
n.b1 = -3.6516812511529557
n.b2 = -0.5814582597051287
n.b3 = -2.1295699805201465

print("""Initial hyperparameters:
n.w1 = {}
n.w2 = {}
n.w3 = {}
n.w4 = {}
n.w5 = {}
n.w6 = {}
n.b1 = {}
n.b2 = {}
n.b3 = {}
""".format(n.w1, n.w2, n.w3, n.w4, n.w5, n.w6, n.b1, n.b2, n.b3))

stats = n.train_batch(ground_truth_dataset, 2000001, 0.001)
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
