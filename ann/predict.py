import sys
import pickle
import argparse
from ann import NeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cough", required=True, help="The number of coughs per hour.")
parser.add_argument("-s", "--sneeze", required=True, help="The number of sneezes per hour.")
args = vars(parser.parse_args())

x = [int(args["cough"]), int(args["sneeze"])]
if x[0] < 0 or x[1] < 0:
  print("One of arguments is minus!")
  sys.exit(1)

with open("model.bin", "rb") as model:
  n = pickle.load(model)
  print("""Hyperparameters:
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
  p = n.feedforward(x)
  percent = int(p * 100)
  if percent >= 50:
    print("Warning! You are {}% POSITIVE with Coronavirus.".format(percent))
  else:
    print("Congratulation! You are NEGATIVE with Coronavirus.".format(percent))
