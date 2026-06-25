# Simple Neural Net

**Maturity:** Notebook study — progressive NumPy implementations from a single neuron to a trainable MNIST classifier.

## Problem

How does a neural network actually work? Before using high-level frameworks, this project builds the core pieces from scratch: a neuron, a fully connected layer, a multi-layer network, and full training with backpropagation.

## Approach

Files are ordered to follow a learning path:

| File | What it covers |
|------|----------------|
| `a_Neuron.py` | Single feed-forward neuron with configurable activation (step function / perceptron) |
| `b_FullyConnected.py` | Matrix-based fully connected layer with ReLU; batch forward pass |
| `c_MNIST-FullyConnected.py` | Multi-layer network on MNIST — forward pass and accuracy evaluation only |
| `d_SimpleNetworkAndTrain.py` | Full training loop: L2 loss, sigmoid activations, backprop, SGD optimization |

**Architecture (MNIST classifier):** 784 → 64 → 32 → 10, sigmoid activations, L2 loss.

**Key implementation details:**
- Weights initialized from a standard normal distribution
- Backprop stores intermediate activations during forward pass for gradient computation
- Mini-batch SGD with configurable batch size, learning rate, and epochs

## Results

When trained with default settings in `d_SimpleNetworkAndTrain.py` (500 epochs, batch size 30, learning rate 5e-3), the network learns to classify MNIST digits. Exact accuracy depends on run time and hardware; training logs validation accuracy per epoch:

```
Epoch    0: training loss = ... | val accuracy = ...%
...
```

`c_MNIST-FullyConnected.py` evaluates **untrained** random weights — expect ~10% accuracy (chance level for 10 classes).

## Run it

```bash
pip install numpy mnist

# Step through the progression
python a_Neuron.py
python b_FullyConnected.py
python d_SimpleNetworkAndTrain.py   # full training (~ several minutes)
```

> **Note:** `c_MNIST-FullyConnected.py` references an external module name from course material (`p35_FullyConnectedImpl`). Use the `FullyConnectedLayer` from `b_FullyConnected.py` or the self-contained implementation in `d_SimpleNetworkAndTrain.py`.

## Takeaways

- Forward pass is straightforward matrix math; backprop is where the pieces connect.
- Building training from scratch makes framework abstractions (`model.fit`, `loss`, `optimizer`) much clearer.
- A simple fully connected network can learn MNIST, but sigmoid + L2 is not state of the art — later projects explore better architectures and APIs.
