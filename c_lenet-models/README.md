# LeNet Models

**Maturity:** Trained CNN on MNIST — architecture definitions and end-to-end training.

## Problem

How was one of the earliest successful convolutional neural networks structured, and how does Keras let you express the same architecture in different ways? The model is trained end-to-end on MNIST to measure CNN performance against the dense baseline in [Simple Vision Model](../b_simple-vision-model/).

## Approach

All files implement **LeNet-5** (LeCun et al., 1998) for 10-class classification on 28×28 inputs:

```
Conv2D(6, 5×5, same) → MaxPool(2×2)
→ Conv2D(16, 5×5)    → MaxPool(2×2)
→ Flatten
→ Dense(120, ReLU) → Dense(84, ReLU) → Dense(10, softmax)
```

| File | Style | Description |
|------|-------|-------------|
| `a_LeNet-w-Keras.py` | Sequential API | Architecture definition + `model.summary()` |
| `b_LeNet-w-Keras-OOP.py` | Model subclass | Same architecture via custom `LeNet5` class |
| `c_LeNet-w-Keras-train-MNIST.py` | Sequential + training | Full pipeline: MNIST load, normalize, compile, fit |

Files `a_` and `b_` define structure only. File `c_` adds the training pipeline and is the script to run for results.

**Training setup** (`c_LeNet-w-Keras-train-MNIST.py`):

- **Input:** MNIST via `tf.keras.datasets.mnist`, pixels normalized to [0, 1], `input_shape=(28, 28, 1)`
- **Optimizer:** SGD; **loss:** `sparse_categorical_crossentropy`
- **Callbacks:** `EarlyStopping(patience=3, monitor='val_loss')`, `TensorBoard(log_dir='./logs')`
- **Batch size:** 32; **epochs:** 80 (early stopping may finish sooner)

## Results

| Metric | Value |
|--------|-------|
| Trainable parameters | 61,706 (~241 KB) |
| Peak validation accuracy | **98.81%** (epoch 12) |
| Validation accuracy at epoch 15 | 98.69% |

Training curve snapshot:

| Epoch | Val accuracy | Val loss |
|-------|--------------|----------|
| 1 | 95.81% | 0.1395 |
| 5 | 98.16% | 0.0606 |
| 10 | 98.77% | 0.0411 |
| 12 | **98.81%** | 0.0375 |

Compared to the flatten-and-dense baseline in [Simple Vision Model](../b_simple-vision-model/) (~95–97% on the same dataset), LeNet reaches ~99% validation accuracy with ~62K parameters — roughly 2–3 percentage points higher with a more parameter-efficient architecture.

Exact epoch count may vary slightly on re-run if EarlyStopping triggers at a different point.

## Run it

```bash
pip install tensorflow

python a_LeNet-w-Keras.py                  # architecture only
python b_LeNet-w-Keras-OOP.py              # OOP architecture only
python c_LeNet-w-Keras-train-MNIST.py      # train on MNIST (~minutes)
```

Running `c_` writes TensorBoard logs to `./logs`. View with `tensorboard --logdir=./logs`.

A Keras warning about passing `input_shape` directly to a layer is harmless — cosmetic in TF 2.x when using Sequential models.

## Takeaways

- LeNet's conv + pool design reaches **~99% validation accuracy** on MNIST with ~62K params — better accuracy and more parameter-efficient than a flatten-and-dense stack.
- **Sequential vs. subclass:** Sequential is ideal for simple stacks; subclassing gives explicit control over the forward pass and scales to custom layers, residual connections, and multi-branch models.
- **Convolutions + pooling** exploit spatial locality — the natural next step after flatten-and-dense baselines.
- LeNet-5 remains a useful reference architecture for understanding how modern CNNs evolved.
