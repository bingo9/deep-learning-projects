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
    def __init__(self, num_inputs, layer_size, activation_fn, d_activation_fn):
        super().__init__()
        # Randomly initializing the weight matrix and the bias value (using a normal distribution this time):
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn
        self.d_activation_fn = d_activation_fn

    def forward(self, x):
        """Forward input signal through the layer"""
        z = np.dot(x, self.W) + self.b
        self.y = self.activation_fn(z)
        self.x =x  # we store values for back-propogation
        return self.y

    def backward(self, dL_dy):
        """Back-propogate the losss."""
        dy_dz = self.d_activation_fn(self.y)  # =f'
        dL_dz = (dL_dy * dy_dz) # dL/dz = dL/dy * dy/dz = l'_{k+1} * f'
        dz_dw = self.x.T
        dz_dx = self.W.T
        dz_db = np.ones(dL_dy.shape[0]) # dz/db = d(W.x + b)/db = 0 + db/db = "ones"-vector

        # Computing the derivatives with respect to the layer's parameters, and storing them for opt. optimization:
        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        # Computing the derivative with respect to the input, to be passed to the previous layers (their `dL_dy`):
        dL_dx = np.dot(dL_dz, dz_dx)
        return dL_dx

    def optimize(self, epsilon):
        """Optimize the layer's parameters, using the stored derivative values."""
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db

def sigmoid(x):  # Apply the sigmoid function to the elements of x
    return 1 / (1 + np.exp(-x)) # y

def derivated_sigmoid(y):  # Sigmoid derivative function
    return y * (1-y)

def loss_L2(pred, target):    # L2 loss function
    return np.sum(np.square(pred - target)) / pred.shape[0] # opt. we divide by the batch size

def derivated_loss_L2(pred, target):    # L2 derivative function
    return 2 * (pred - target)


class SimpleNetwork(object):
    """A simple fully-connected NN.
    Args:
        num_inputs (int): The input vector size / number of input values.
        num_outputs (int): The output vector size.
        hidden_layers_sizes (list): A list of sizes for each hidden layer to add to the network
    Attributes:
        layers (list): The list of layers forming this simple network.
    """
    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32), loss_fn=loss_L2, d_loss_fn=derivated_loss_L2):
        super().__init__()
        # We build the list of layers composing the network, according to the provided arguments:
        sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(sizes[i], sizes[i + 1], sigmoid, derivated_sigmoid)
            for i in range(len(sizes) - 1)
        ]
        self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn
    
    def forward(self, x):
        """Forward the input vector through the layers, returning the output vector."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dL_dy):
        """Back-propogate the loss derivatives from last to 1st layer."""
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy
    
    def optimize(self, epsilon):
        """Optimize the network's parameters, using the stored derivative values (stored stored gradients)."""
        for layer in self.layers:
            layer.optimize(epsilon)

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, num_epochs=100, learning_rate=5e-3):
        """Train (and evaluate) the network on the training and validation sets."""
        num_batches_per_epoch = len(X_train) // batch_size
        loss, accuracy = [], []
        for i in range(num_epochs):  # For each training epoch
            epoch_loss = 0
            for b in range(num_batches_per_epoch):  # For each batch composing the dataset
                # Get batch
                b_idx = b * batch_size
                b_idx_e = b_idx + batch_size
                x, y_true = X_train[b_idx: b_idx_e], y_train[b_idx: b_idx_e]
                # Optimize on batch
                y = self.forward(x)  # forward pass
                epoch_loss += self.loss_fn(y, y_true)  # loss
                dL_dy = self.d_loss_fn(y, y_true)  # loss derivation
                self.backward(dL_dy)  # back-propagation pass
                self.optimize(learning_rate)  # optimization of the NN
            loss.append(epoch_loss / num_batches_per_epoch)
            # After each epoch, we evaluate the accuracy on the validation set
            accuracy.append(self.evaluate_accuracy(X_val, y_val))
            print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(i, loss[i], accuracy[i] * 100))
        return loss, accuracy

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

# Network for MNIST images, with 2 hidden layers of 64 and 32 neurons each:
mnist_classifier = SimpleNetwork(X_train.shape[1], num_classes, [64, 32])

losses, accuracies = mnist_classifier.train(X_train, y_train, X_test, y_test, batch_size=30, num_epochs=500)