# Simple Vision Model

**Maturity:** Exploration — minimal Keras baseline for MNIST image classification.

## Problem

After implementing networks from scratch in NumPy, what's the simplest way to classify MNIST images using a modern deep learning framework?

## Approach

A two-layer fully connected classifier built with the Keras Sequential API:

```
Flatten(28×28) → Dense(128, ReLU) → Dense(10, softmax)
```

**Training setup:**
- **Data:** MNIST via `tf.keras.datasets.mnist`, pixel values normalized to [0, 1]
- **Optimizer:** SGD
- **Loss:** Sparse categorical crossentropy (integer labels, no one-hot encoding needed)
- **Metrics:** Accuracy
- **Epochs:** 5, with validation on the test set each epoch

This is intentionally minimal — no convolutions, no data augmentation — to establish a framework baseline before moving to LeNet-style architectures.

## Results

With 5 epochs of SGD, expect test accuracy in the **~95–97%** range (typical for this architecture on MNIST). Run locally to see exact numbers printed during `model.fit`.

## Run it

```bash
pip install tensorflow

python a_Keras-simple-vision-model.py
```

Training progress and per-epoch validation metrics print to the console. `model.summary()` shows parameter counts before training begins.

## Takeaways

- Keras reduces a full training pipeline to a handful of lines compared to the NumPy implementation.
- A plain dense network works reasonably on MNIST but ignores spatial structure in images — motivation for convolutional layers in the next project.
- `sparse_categorical_crossentropy` pairs naturally with integer labels; no manual one-hot encoding required.
