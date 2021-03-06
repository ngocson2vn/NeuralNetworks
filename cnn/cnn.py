import math
import mnist
import numpy as np
from util import softmax, loss
from convolution import Convolution2D, Convolution3D
from pooling import MaxPool

class CNN:
  def __init__(self, N, M):
    u1 = 0.1
    u2 = 0.1
    u = 0.1
    self.k1 = np.random.uniform(low=-u1, high=u1, size=(N, 5, 5))
    self.b1 = np.random.uniform(low=-u1, high=u1, size=(N))
    self.k2 = np.random.uniform(low=-u2, high=u2, size=(M, N, 5, 5))
    self.b2 = np.random.uniform(low=-u2, high=u2, size=(M))
    self.w = np.random.uniform(low=-u, high=u, size=(10, M * 16))
    self.b = np.random.uniform(low=-u, high=u, size=(10))


  def feedforward(self, img):
    ## Convolution Layer C1
    conv1 = Convolution2D(self.k1, self.b1, stride=1, padding=0)
    C1, dC1S1 = conv1.feedforward(img)

    ## Pooling Layer P1
    maxpool = MaxPool(size=2)
    P1, I1 = maxpool.feedforward(C1)

    ## Convolution Layer C2
    conv2 = Convolution3D(self.k2, self.b2, stride=1, padding=0)
    C2, dC2S2, dS2P1 = conv2.feedforward(P1)

    ## Pooling Layer P2
    P2, I2 = maxpool.feedforward(C2)

    ## FC Layer
    f = P2.flatten()
    O = softmax(np.dot(self.w, f) + self.b)

    return O


  def train(self, train_images, train_labels, test_images, test_labels, epoch, lr):
    N = self.k1.shape[0]
    M = self.k2.shape[0]
    epochs = []
    avg_losses = []
    accuracies = []

    for ep in range(epoch):
      # Shuffle the training data
      permutation = np.random.permutation(len(train_images))
      train_images = train_images[permutation]
      train_labels = train_labels[permutation]
      for img, label in zip(train_images, train_labels):

        ############################################################
        # Feedforward phase                                        #
        ############################################################

        ## Convolution Layer C1
        conv1 = Convolution2D(self.k1, self.b1, stride=1, padding=0)
        C1, dC1S1 = conv1.feedforward(img)

        ## Pooling Layer P1
        maxpool = MaxPool(size=2)
        P1, I1 = maxpool.feedforward(C1)

        ## Convolution Layer C2
        conv2 = Convolution3D(self.k2, self.b2, stride=1, padding=0)
        C2, dC2S2, dS2P1 = conv2.feedforward(P1)

        ## Pooling Layer P2
        P2, I2 = maxpool.feedforward(C2)

        ## FC Layer
        f = P2.flatten()
        O = softmax(np.dot(self.w, f) + self.b)


        ############################################################
        # Backpropagation phase                                    #
        ############################################################

        ## 1. Calculate gradients of parameters in the FC layer
        dLS = np.copy(O)
        dLS[label] = O[label] - 1
        dLb = np.copy(dLS)
        
        dLw = np.zeros(O.shape + f.shape, dtype=np.float64)
        for i in range(O.shape[0]):
          dLw[i, :] = O[i] * f
        dLw[label, :] = (O[label] - 1) * f

        dLf = np.zeros(f.shape, dtype=np.float64)
        for j in range(f.shape[0]):
          dLf[j] = np.sum(dLS * self.w[:, j])

        dLP2 = dLf.reshape(P2.shape)



        ## 2. Calculate gradients of parameters in the C2 layer

        ### 2.1. Calculate dLC2
        dLC2 = np.zeros(C2.shape, dtype=np.float64)

        for m in range(P2.shape[0]):
           for x in range(P2.shape[1]):
             for y in range(P2.shape[2]):
               umax, vmax = I2[m, x, y]
               dLC2[m, umax, vmax] = dLP2[m, x, y]

        ### 2.2. Calculate dLS2
        dLS2 = dLC2 * dC2S2

        ### 2.3. Calculate dLb2 and dLk2
        dLb2 = np.zeros(self.k2.shape[0], dtype=np.float64)
        dLk2 = np.zeros(self.k2.shape, dtype=np.float64)
        for m in range(self.k2.shape[0]):
          dLb2[m] = np.sum(dLS2[m])
          for n in range(self.k2.shape[1]):
            for p in range(self.k2.shape[2]):
              for q in range(self.k2.shape[3]):
                dLk2[m, n, p, q] = np.sum(dLS2[m] * P1[n][p:(p + C2.shape[1]), q:(q + C2.shape[2])])

        

        ## 3. Calculate gradients of parameters in the C1 layer

        ### 3.1. Calculate dLP1
        dLP1 = np.zeros(P1.shape, dtype=np.float64)
        for n in range(P1.shape[0]):
          for r in range(P1.shape[1]):
            for s in range(P1.shape[2]):
              dLP1[n, r, s] = np.sum(dLS2 * dS2P1[n, r, s])


        ### 3.2. Calculate dLC1
        dLC1 = np.zeros(C1.shape, dtype=np.float64)
        for n in range(P1.shape[0]):
           for r in range(P1.shape[1]):
             for s in range(P1.shape[2]):
               imax, jmax = I1[n, r, s]
               dLC1[n, imax, jmax] = dLP1[n, r, s]

        ### 3.3. Calculate dLS1
        dLS1 = dLC1 * dC1S1

        ### 3.4. Calculate dLb1 and dLk1
        dLb1 = np.zeros(self.k1.shape[0], dtype=np.float64)
        dLk1 = np.zeros(self.k1.shape, dtype=np.float64)
        for n in range(self.k1.shape[0]):
          dLb1[n] = np.sum(dLS1[n])
          for g in range(self.k1.shape[1]):
            for h in range(self.k1.shape[2]):
              dLk1[n, g, h] = np.sum(dLS1[n] * img[g:(g + C1.shape[1]), h:(h + C1.shape[2])])


        ## 4. Update parameters
        self.k1 = self.k1 - lr * dLk1
        self.b1 = self.b1 - lr * dLb1

        self.k2 = self.k2 - lr * dLk2
        self.b2 = self.b2 - lr * dLb2

        self.w  = self.w  - lr * dLw
        self.b  = self.b  - lr * dLb


      losses = []
      acc = 0
      for img, label in zip(test_images, test_labels):
        O = self.feedforward(img)
        losses.append(-np.log(O[label]))
        acc += 1 if np.argmax(O) == label else 0
      losses = np.array(losses)
      
      epochs.append(ep + 1)
      avg_losses.append(losses.mean())
      accuracy = 100 * acc / len(test_labels)
      accuracies.append(accuracy)
      print("Epoch: {}, validate_average_loss: {}, validate_accuracy: {:02.2f}%".format(ep + 1, losses.mean(), accuracy))

    return (epochs, avg_losses, accuracies)
