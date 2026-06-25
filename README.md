# Deep Learning Projects

A collection of deep learning implementations, from first principles in NumPy through Keras/TensorFlow. Each folder is a self-contained project covering a specific topic in computer vision — neurons and backpropagation, image classification baselines, and classic CNN architectures.

**Focus areas:** neural network fundamentals, backpropagation, fully connected and convolutional architectures, Keras API patterns.

## Projects

| Project | Topic | Stack | Maturity | Highlights |
|---------|-------|-------|----------|------------|
| [Simple Neural Net](./a_simple-neural-net/) | Fundamentals | NumPy | From scratch | Neuron → layer → network → backprop + MNIST training |
| [Simple Vision Model](./b_simple-vision-model/) | Image classification | TensorFlow / Keras | Baseline | Flatten + dense classifier on MNIST |
| [LeNet Models](./c_lenet-models/) | Convolutional networks | TensorFlow / Keras | Architecture | LeNet-5 via Sequential and Model subclass APIs |

## How to explore

1. Start with **[Simple Neural Net](./a_simple-neural-net/)** — files are ordered `a_` through `d_` and build on each other conceptually.
2. Move to **[Simple Vision Model](./b_simple-vision-model/)** for a high-level Keras baseline on the same dataset.
3. See **[LeNet Models](./c_lenet-models/)** for a classic CNN architecture and two Keras construction styles.

Each project folder has its own README with problem context, approach, and run instructions.

## Requirements

Dependencies vary by project. At minimum:

```bash
pip install numpy tensorflow mnist
```

See individual project READMEs for details.

## About

Core deep learning building blocks implemented end to end: forward pass, backpropagation, mini-batch training, and convolutional architectures on MNIST. NumPy implementations establish the mechanics; Keras and TensorFlow apply the same concepts through production-style APIs.
