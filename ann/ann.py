import numpy as np

def sigmoid(x):
  # Sigmoid activation function: sigmoid(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
  # Derivative of sigmoid activation function: d_sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
  return sigmoid(x) * (1 - sigmoid(x))

def L(y_true, y_pred):
  # Loss function
  return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
  """
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  """

  def __init__(self):
    # Initialize weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    
    # Initialize biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, ground_truth_dataset, epoch, lr):
    # ground_truth_dataset: has shape (n, 3), where n is the number of items.
    # epoch: the number of times to loop through the entire ground truth dataset
    # lr: learning rate

    epochs = []
    min_losses = []
    avg_losses = []
    max_losses = []
    y_trues = np.array(ground_truth_dataset)[:, 2]

    for ep in range(epoch):
      losses = []
      for item in ground_truth_dataset:
        # input
        x1, x2 = item[:2]

        # real result
        y_true = item[2]

        
        # ====== Feed forward ======
        # Neuron h1
        h1_sum = self.w1 * x1 + self.w2 * x2 + self.b1
        h1 = sigmoid(h1_sum)

        # Neuron h2
        h2_sum = self.w3 * x1 + self.w4 * x2 + self.b2
        h2 = sigmoid(h2_sum)

        # Neuron o1
        o1_sum = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(o1_sum)
        y_pred = o1

        loss = L(y_true, y_pred)
        losses.append(loss)

        # ====== Back propagation ======
        # Calculate gradients for OUTPUT layer
        dL_dy_pred = -2 * (y_true - y_pred)

        dL_dw5 = dL_dy_pred * d_sigmoid(o1_sum) * h1
        dL_dw6 = dL_dy_pred * d_sigmoid(o1_sum) * h2
        dL_db3 = dL_dy_pred * d_sigmoid(o1_sum)

        # Calculate gradients for HIDDEN layer
        dL_dw1 = dL_dy_pred * self.w5 * d_sigmoid(o1_sum) * d_sigmoid(h1_sum) * x1
        dL_dw2 = dL_dy_pred * self.w5 * d_sigmoid(o1_sum) * d_sigmoid(h1_sum) * x2
        dL_dw3 = dL_dy_pred * self.w6 * d_sigmoid(o1_sum) * d_sigmoid(h2_sum) * x1
        dL_dw4 = dL_dy_pred * self.w6 * d_sigmoid(o1_sum) * d_sigmoid(h2_sum) * x2
        dL_db1 = dL_dy_pred * self.w5 * d_sigmoid(o1_sum) * d_sigmoid(h1_sum)
        dL_db2 = dL_dy_pred * self.w6 * d_sigmoid(o1_sum) * d_sigmoid(h2_sum)

        # Update weights and biases
        self.w1 -= lr * dL_dw1
        self.w2 -= lr * dL_dw2
        self.w3 -= lr * dL_dw3
        self.w4 -= lr * dL_dw4
        self.w5 -= lr * dL_dw5
        self.w6 -= lr * dL_dw6

        self.b1 -= lr * dL_db1
        self.b2 -= lr * dL_db2
        self.b3 -= lr * dL_db3

      epochs.append(ep)
      min_losses.append(min(losses))
      avg_losses.append(sum(losses) / len(losses))
      max_losses.append(max(losses))
      if ep % 100 == 0:
        print("Epoch {}: min_loss = {}, avg_loss = {}, max_loss = {}".format(
          ep, min_losses[ep], avg_losses[ep], max_losses[ep]))

    return (epochs, min_losses, avg_losses, max_losses)
