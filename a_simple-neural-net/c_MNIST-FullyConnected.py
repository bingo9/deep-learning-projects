import numpy as np
import mnist
mnist.datasets_url = 'http://yann.lecun.com/exdb/mnist/'

np.random.seed(42)

# Loading the MNIST dataset:
X_train, y_train = mnist.train_images(), mnist.train_labels()
X_test, y_test = mnist.test_images(), mnist.test_labels()
num_classes = 10

# We transform the images into column vectors (as inputs for our NN)
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
# We "one-hot" encode the labels (as outputs for our NN) for instance, transform
# the label 4 into [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
y_train = np.eye(num_classes)[y_train]



from p35_FullyConnectedImpl import FullyConnectedLayer

def sigmoid(x):  # Apply the sigmoid function to the elements of x
    return 1 / (1 + np.exp(-x)) # y

class SimpleNetwork(object):
    """A simple fully-connected NN.
    Args:
        num_inputs (int): The input vector size / number of input values.
        num_outputs (int): The output vector size.
        hidden_layers_sizes (list): A list of sizes for each hidden layer to add to the network
    Attributes:
        layers (list): The list of layers forming this simple network.
    """
    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32)):
        super().__init__()
        # We build the list of layers composing the network, according to the provided arguments:
        sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(sizes[i], sizes[i + 1], sigmoid)
            for i in range(len(sizes) - 1)
        ]
    
    def forward(self, x):
        """Forward the input vector through the layers, returning the output vector."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, x):
        """Compute the output corresponding to input `x`, and return the index of the largest output value."""
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class

    def evaluate_accuracy(self, X_val, y_val):
        """Evaluate the accuracy of the network on the validation set."""
        num_corrects = 0
        for i in range(len(X_val)):
            if self.predict(X_val[i]) == y_val[i]:
                num_corrects += 1
        return num_corrects / len(X_val)

# Network for MNIST images, with 2 hidden layers of 64 and 32 neurons each:
mnist_classifier = SimpleNetwork(X_train.shape[1], num_classes, [64, 32])

# ... and we evaluate its accuracy on the test set:
accuracy = mnist_classifier.evaluate_accuracy(X_test, y_test)
print(f"Accuracy = {accuracy * 100:.2f}%")
