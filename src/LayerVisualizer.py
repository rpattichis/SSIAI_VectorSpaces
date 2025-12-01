import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.ma import minimum_fill_value

class LayerVisualizer:
    """
    NOTES/ASSUMPTIONS: 
      assuming square image
      We want the form out = W (input) -- W.shape == (10 x (28*28))
      weights are before the activation function
    Inputs:
      model: the entire NN model
      hidden_layer: the hidden layer desired
      layer_num: layer number within the hidden layer
      input_dim: the row/col length of the input (in our case, 28 for a square input image)
      layer_num: is the layer is sequential, which part of the weights to visualize
    """
    def __init__(self, model, hidden_layer, layer_num, input_dim=28):
        self.model = model
        self.input_dim = input_dim
        self.weights = None

        # for FCNN example
        if layer_num != None: 
          hidden_layer = hidden_layer[layer_num]
          self.weights = hidden_layer.weight.detach().numpy()

        # for CNN / ResNet example
        else: 
          self.weights = hidden_layer.weight.detach().numpy()
          in_channels, out_channels, kern_size, _ = hidden_layer.weight.detach().numpy().shape
          self.weights = np.reshape(self.weights, (in_channels * out_channels, kern_size * kern_size))

    """
    Goal: this will reshape the matrix from a 2D numpy array to a 3D array. 
          Meant to be used in the visualize_space function for the FCNN first layer. 
          The rows get transformed to the shape of the input image (in our example, 28, 28).
    NOTE: this only works under strict assumptions 
    """
    def reshape_space(self, subsp_mat):
        nrows, ncols = subsp_mat.shape
        if math.sqrt(nrows).is_integer():
          return np.reshape(subsp_mat, (int(math.sqrt(nrows)), int(math.sqrt(nrows)), ncols))
        return np.reshape(subsp_mat, (nrows, 1, ncols))

    """ 
    Goal: Set the attributes for the rank and different spaces of weights of the hidden layer.
          The vectors are stored in the columns of the resulting matrices. 
    """
    def extract_spaces(self):
        U, S, Vh = np.linalg.svd(self.weights)
        self.rank = np.linalg.matrix_rank(self.weights)
        self.S = S
        self.row_space = Vh[:self.rank, :].T
        self.null_space = Vh[self.rank: :].T
        self.col_space = U[:, :self.rank]
        self.left_null = U[:, :self.rank]

    """
    Goal: Visualize any given subspace (i.e., average images, weights, residuals, etc.).
    NOTE/ASSUMPTIONS: 
      this is just the first layer of our model that takes in a square image (so that we can reshape 
        the subspace matrix). 
      Assume that all the vectors are stored in the columns. 
    Inputs:
      subsp_mat: subspace we want to visualize (should be 3D or 4D numpy array)
        - 3D: should be of size (IMG_SIZE, IMG_SIZE, N_CLASSES)
        - 4D: should be of size (N_SPACES, IMG_SIZE, IMG_SIZE, N_CLASSES)
      min_v, max_v: meant for normalization of the plot
    """
    def visualize_space(self, subsp_mat, min_v=None, max_v=None):
      ncols = subsp_mat.shape[-1]

      # there's only one subspace we want to visualize
      if subsp_mat.ndim == 3:
        plt.figure(figsize=(ncols * 1.2, 1.2))
        for col in range(ncols):
          plt.subplot(1, ncols, col + 1)
          if min_v is None or max_v is None:
            min_v = np.min(subsp_mat)
            max_v = np.max(subsp_mat)
          M = subsp_mat[:, :, col]
          plt.imshow(M, cmap="gray", interpolation="nearest", vmin=min_v, vmax=max_v)
          plt.axis('off')
        plt.show()

      # there are multiple subspaces -- take the min and max to normalize along each column of the graph
      elif subsp_mat.ndim == 4:
        nrows = len(subsp_mat)
        plt.figure(figsize=(nrows, ncols))
        fig, axs = plt.subplots(nrows, ncols)

        for col in range(ncols):
          MIN_V, MAX_V = np.min(subsp_mat[:, :, :, col]), np.max(subsp_mat[:, :, :, col])
          for row in range(nrows):
            axs[row, col].imshow(subsp_mat[row, :, :, col], cmap="gray", interpolation="nearest", vmin=MIN_V, vmax=MAX_V)
            axs[row, col].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
      return

    def residual_space_proj(self, input_image, proj_out):
      return np.subtract(input_image, proj_out)

    # weights example is 10, 7.. (output, input)
    def proj_weights(self, input_image, weights):
      output_shape, input_shape = weights.shape
      proj_out = np.zeros((output_shape, input_shape))
      for i in range(output_shape):
        reshaped_weights = weights[i, :].flatten()
        num = np.dot(reshaped_weights.T, input_image.flatten())
        denom = np.dot(reshaped_weights.T, reshaped_weights)
        proj_out[i, :] = np.multiply(num / denom, reshaped_weights)
      return proj_out