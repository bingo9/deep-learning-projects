# Deep Learning Projects

A collection of deep learning explorations, from first principles in NumPy through Keras/TensorFlow implementations. Each folder is a self-contained study in a specific topic, building toward practical image classification on MNIST.

**Focus areas:** neural network fundamentals, backpropagation, fully connected and convolutional architectures, Keras API patterns.

## Projects

| Project | Topic | Stack | Maturity | Highlights |
|---------|-------|-------|----------|------------|
| [Simple Neural Net](./a_simple-neural-net/) | Fundamentals | NumPy | Notebook study | Neuron → layer → network → backprop + MNIST training |
| [Simple Vision Model](./b_simple-vision-model/) | Image classification | TensorFlow / Keras | Exploration | Flatten + dense baseline on MNIST |
| [LeNet Models](./c_lenet-models/) | Convolutional networks | TensorFlow / Keras | Exploration | LeNet-5 via Sequential and Model subclass APIs |

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

These projects document hands-on learning — understanding how neural networks work under the hood before leaning on frameworks. Suitable for reviewing fundamentals, architecture choices, and progression from scratch implementations to production-style APIs.
