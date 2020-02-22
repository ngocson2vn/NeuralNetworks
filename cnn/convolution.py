import numpy as np
from util import relu

class Convolution2D:
  def __init__(self, k1, b1, stride=1, padding=0):
    """
      - k: (6, 5, 5) kernels
      - b: (6) bias
    """
    self.k1 = k1
    self.b1 = b1
    self.stride = stride
    self.padding = padding

  def feedforward(self, img):
    """
      - img: (28, 28) image
    """

    N, k1_height, _ =  self.k1.shape

    C1_height = int((img.shape[0] - k1_height + 2 * self.padding) / self.stride) + 1

    # Initialize C
    C1 = np.zeros((N, C1_height, C1_height), dtype=np.float64)
    dC1S1 = np.zeros(C1.shape, dtype=np.float64)

    for n in range(N):
      for i in range(C1_height):
        for j in range(C1_height):
          region = img[i:(i + k1_height), j:(j + k1_height)]
          S1_nij = np.sum(region * self.k1[n]) + self.b1[n]
          C1[n, i, j] = relu(S1_nij)
          dC1S1[n, i, j] = 1 if S1_nij > 0 else 0

    return C1, dC1S1


class Convolution3D:
  def __init__(self, k2, b2, stride=1, padding=0):
    """
      - k:   (12, 6, 5, 5) kernels
      - b:   (12) bias
    """
    self.k2 = k2
    self.b2 = b2
    self.stride = stride
    self.padding = padding

  def feedforward(self, P1):
    """
      - P1: (6, 12, 12) image
    """
    M, N, k2_height, _ = self.k2.shape
    C2_height = int((P1.shape[1] - k2_height + 2 * self.padding) / self.stride) + 1

    C2 = np.zeros((M, C2_height, C2_height), dtype=np.float64)
    dC2S2 = np.zeros(C2.shape, dtype=np.float64)
    dS2P1 = np.zeros(P1.shape + C2.shape, dtype=np.float64)

    for m in range(M):
      for u in range(C2_height):
        for v in range(C2_height):
          region = P1[0:N, u:(u + k2_height), v:(v + k2_height)]
          S2_muv = np.sum(region * self.k2[m]) + self.b2[m]
          C2[m, u, v] = relu(S2_muv)
          dC2S2[m, u, v] = 1 if S2_muv > 0 else 0
          dS2P1[0:N, u:(u + k2_height), v:(v + k2_height), m, u, v] = self.k2[m]

    return C2, dC2S2, dS2P1
