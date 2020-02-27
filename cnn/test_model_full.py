import mnist
import numpy as np
import argparse
from cnn import CNN
import matplotlib.pyplot as plt
from termcolor import colored
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index", help="The index of a test image.", type=int, required=True)
index = int(vars(parser.parse_args())["index"])

test_images = (mnist.test_images() / 255) - 0.5
test_labels = mnist.test_labels()
img = test_images[index]
label = test_labels[index]

with open("artifacts/model_full.bin", "rb") as model:
  cnn = pickle.load(model)

fig = plt.figure(figsize=(4, 5))

O = cnn.feedforward(img)
inference = np.argmax(O)
confidence = O[label] * 100
correct = "True" if inference == label else "False"
cmap = plt.get_cmap("gray") if inference == label else plt.get_cmap("Reds")
color = "blue" if inference == label else "red"

title = "Inferred Number: {}\nConfidence: {:02.2f}%\nCorrect: {}".format(inference, confidence, correct)
sub = fig.add_subplot(1, 1, 1, xlabel="Label: {}".format(label))
sub.text(14, -1, title, ha="center", va="bottom", size="large", color=color)
sub.imshow(img, cmap=cmap)

plt.show()
