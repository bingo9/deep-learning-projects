import numpy as np

class FullyConnectedLayer(object):
    """A fully connected layer.
    Args:
        num_inputs (int): The input vector size / number of input values.
        layer_size (int): The output vector size / number of neurons in the layer.
        activation_fn (callable): The activation function defining this layer.
    Attributes:
        W (ndarray): The weight values for each input.
        b (float): The bias value, added to the weighted sum.
        size (int): The layer size / number of neurons.
        activation_fn (callable): The activation function computing the layer's output.
    """
    def __init__(self, num_inputs, layer_size, activation_fn):
        super().__init__()
        # Randomly initializing the weight matrix and the bias value (using a normal distribution this time):
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn

    def forward(self, x):
        """Forward input signal through the layer"""
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)

np.random.seed(42)
# Random input column array of 2 values (shape = (1,2))
x1 = np.random.uniform(-1, 1, 2).reshape(1, 2)
# > [[0.25091976 0.90142861]]
x2 = np.random.uniform(-1, 1, 2).reshape(1, 2)
# > [[0.46398788 0.19731697]]

# Instantiating a Fully Connected Layer (simple layer with ReLU function):
relu_fn = lambda y: np.maximum(y, 0) # Defining our activation function
layer = FullyConnectedLayer(2, 3, relu_fn)

# Our layer can process x1 and x2 separately...
out1 = layer.forward(x1)
# > [[0.28712364 0.         0.33478571]]
out2 = layer.forward(x2)
# > [[0.         0.         1.08175419]]

# ... or together:
x12 = np.concatenate((x1, x2))  # stack of input vectors, of shape `(2, 2)`
out12 = layer.forward(x12)
# > [[0.28712364 0.         0.33478571]
#    [0.         0.         1.08175419]]
print(out12)