# LeNet Models

**Maturity:** Exploration — LeNet-5 architecture study using two Keras construction patterns.

## Problem

How was one of the earliest successful convolutional neural networks structured, and how does Keras let you express the same architecture in different ways?

## Approach

Both files implement **LeNet-5** (LeCun et al., 1998) for 10-class classification on 28×28 inputs:

```
Conv2D(6, 5×5, same) → MaxPool(2×2)
→ Conv2D(16, 5×5)    → MaxPool(2×2)
→ Flatten
→ Dense(120, ReLU) → Dense(84, ReLU) → Dense(10, softmax)
```

| File | Style | Description |
|------|-------|-------------|
| `a_LeNet-w-Keras.py` | **Sequential API** | Layer-by-layer `model.add(...)` — straightforward for linear stacks |
| `b_LeNet-w-Keras-OOP.py` | **Model subclass** | Custom `LeNet5(Model)` class with layers defined in `__init__` and forward logic in `call()` — flexible for complex or multi-input models |

These scripts define and summarize the architecture. Add data loading, `compile`, and `fit` to train (see [Simple Vision Model](../b_simple-vision-model/) for a complete Keras training example).

## Results

Architecture-only at this stage — run `model.summary()` to inspect layer output shapes and parameter counts. When trained on MNIST with appropriate preprocessing (e.g., reshaping to `(28, 28, 1)` and normalizing pixels), LeNet-style models typically exceed the simple dense baseline in accuracy and parameter efficiency.

## Run it

```bash
pip install tensorflow

python a_LeNet-w-Keras.py      # Sequential LeNet-5
python b_LeNet-w-Keras-OOP.py  # Subclass LeNet-5
```

Each script prints the model summary. To train, extend with MNIST loading and `model.compile` / `model.fit` following the pattern in [Simple Vision Model](../b_simple-vision-model/).

## Takeaways

- **Sequential vs. subclass:** Sequential is ideal for simple stacks; subclassing gives explicit control over the forward pass and scales to custom layers, residual connections, and multi-branch models.
- **Convolutions + pooling** exploit spatial locality — the natural next step after flatten-and-dense baselines.
- LeNet-5 remains a useful reference architecture for understanding how modern CNNs evolved.
