import numpy as np

class MaxPool:
  def __init__(self, size=2):
    self.size = 2

  def feedforward(self, C):
    N, C_height, _ = C.shape
    P_height = int(C_height / self.size)

    P = np.zeros((N, P_height, P_height), dtype=np.float64)
    indices = np.zeros((N, P_height, P_height), dtype=(np.int64, 2))

    for n in range(N):
      for i in range(P_height):
        for j in range(P_height):
          region = C[n, (2 * i):(2 * i + 2), (2 * j):(2 * j + 2)]
          P[n, i, j] = np.max(region)
          local_indices = np.unravel_index(np.argmax(region), region.shape)
          indices[n, i, j] = [2 * i + local_indices[0], 2 * j + local_indices[1]]

    return P, indices
