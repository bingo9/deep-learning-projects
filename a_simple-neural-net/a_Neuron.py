import numpy as np

class Neuron(object):
    """A simple feed forward artificial neuron.
    Args:
        num_inputs (int): The input vector size / number of input values.
        activation_fn (callable): The activation function defining this neuron.
    Attributes:
        W (ndarray): The weight values for each input.
        b (float): The bias value, added to the weighted sum.
        activation_fn (callable): The activation function computing the neuron's output.
    """
    def __init__(self, num_inputs, activation_fn):
        super().__init__()
        self.W = np.random.rand(num_inputs)
        self.b = np.random.rand(1)
        self.activation_fn = activation_fn

    def forward(self, x):
        """Forward input signal through the neuron"""
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)

# Fixing the random number generator's seed to get reproducible results:
np.random.seed(42)
# Random input column array of 3 values (shape = (1,3))
x = np.random.rand(3).reshape(1, 3)
# > [[0.37454012 0.95071431 0.73199394]]

# Instantiating a Perceptron (simple neuron with step function):
step_fn = lambda y: 0 if y <= 0 else 1

perceptron = Neuron(num_inputs=x.size, activation_fn=step_fn)
# > perceptron.W    = [0.59865848 0.15601864 0.15599452]
# > perceptron.b    = [0.05808361]

out = perceptron.forward(x)
# > 1
print(out)