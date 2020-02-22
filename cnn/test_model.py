import mnist
import numpy as np
from cnn import CNN
import matplotlib.pyplot as plt
from termcolor import colored
import pickle

test_images = (mnist.test_images() / 255) - 0.5
test_labels = mnist.test_labels()

with open("model.bin", "rb") as model:
  cnn = pickle.load(model)

i = 0
acc = 0
losses = []
confidences = []
drifts = []

permutation = np.random.permutation(len(test_images))
test_images = test_images[permutation]
test_labels = test_labels[permutation]

for img, label in zip(test_images[:100], test_labels[:100]):
  O = cnn.feedforward(img)
  inference = np.argmax(O)
  acc += 1 if inference == label else 0
  losses.append(-np.log(O[label]))
  confidence = O[label] * 100
  confidences.append(confidence)
  correct = colored("True", "green") if inference == label else colored("False", "red")
  print("Label: {}, inference: {}, confidence: {:02.2f}%, correct: {}".format(label, inference, confidence, correct))
  i += 1

print("=" * 100)
print("Accuracy: {}%".format(acc))

fig = plt.figure()
g = fig.add_subplot(2, 1, 1, ylabel="Loss", xlabel="Image Number")
g.plot(list(range(100)), losses, label="Loss")

# g = fig.add_subplot(2, 1, 2, ylabel="Confidence", xlabel="Image Number")
# g.plot(list(range(100)), confidences, label="Confidence")
plt.show()
