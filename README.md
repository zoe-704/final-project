# RPSNet: CNN Gesture Recognition from Scratch

A convolutional neural network (CNN) built entirely from scratch in NumPy that learns to recognize rock, paper, and scissors hand gestures in real time via webcam to play against the user.

- Trains a CNN from scratch on a labeled image dataset of hand gestures
- Classifies live webcam frames into rock, paper, or scissors in real time
- Plays a full game with countdown, scoring, and result display using OpenCV

---

## Project Structure

```
├── train.py          # Model architecture, training loop, evaluation, save/load
├── play.py           # Webcam game interface (OpenCV UI)
├── model.pkl         # Saved weights (generated after training)
└── train/            # Training data, organized as train/rock/, train/paper/, train/scissors/
    ├── rock/
    ├── paper/
    └── scissors/
```

---

## Architecture

A 4-block CNN followed by 4 dense layers — all implemented manually:

```
Input (1×64×64 grayscale)
  → Conv2D(8 filters, 3×3)  → Leaky ReLU → MaxPool(2×2)   # (8×32×32)
  → Conv2D(16 filters, 3×3) → Leaky ReLU → MaxPool(2×2)   # (16×16×16)
  → Conv2D(32 filters, 3×3) → Leaky ReLU → MaxPool(2×2)   # (32×8×8)
  → Conv2D(64 filters, 3×3) → Leaky ReLU → MaxPool(2×2)   # (64×4×4)
  → Flatten → Dense(1024→256) → Dense(256→128)
  → Dense(128→64) → Dense(64→3) → Softmax
```

- **Loss**: Cross-entropy
- **Optimizer**: SGD with gradient clipping (clip = 5.0)
- **Weight init**: He initialization
- **Activation**: Leaky ReLU (slope 0.01 for negatives)

---

## How to Run

### 1. Train the model

```bash
python train.py
```

Runs 30 epochs at `lr=0.001`, sampling up to 300 images per class per epoch. Omit `--resume` to train a fresh `model.pkl`.

### 2. Finetune the model

```bash
python train.py --lr 0.0005 --resume --samples_per_class 500
```

Runs 30 epochs at `lr=0.0005`, sampling up to 500 images per class per epoch. Use `--resume` to continue from a saved `model.pkl`.

> **Note:** Use `--epochs xyz` to change the number of epochs. Each training run takes approximately 10–12 hours.

### 3. Play the game

```bash
python play.py
```

**Controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Start a round / advance from result screen |
| `R` | Reset scores |
| `Q` | Quit |

Place your hand in the green box on the right side of the screen. The model reads that region, classifies your gesture, and locks it in after a 3-second countdown.

---

## Implementation Details

All forward pass and backpropagation code is written from scratch — no deep learning libraries used for computation:

- **`conv2D`** — sliding kernel convolution over 4D tensors `(N, C, H, W)`
- **`max_pooling`** — 2×2 max pooling with configurable stride
- **`leaky_relu`** — element-wise activation
- **`flatten` / `dense` / `softmax`** — standard fully-connected layer operations
- **`cross_entropy`** — numerically stable loss
- **Full backprop** — analytic gradients for every layer including `conv2D_backward`, `max_pooling_backward`, `relu_backward`, `dense_backward`

Training runs one image at a time (batch size = 1). Each epoch randomly resamples the dataset for variety.

---

## Dataset

The model expects a `train/` directory structured as:

```
train/
  rock/       ← JPEG/PNG images of rock hand gesture
  paper/      ← JPEG/PNG images of paper hand gesture
  scissors/   ← JPEG/PNG images of scissors hand gesture
```

A compatible dataset can be found on Kaggle (search: "rock paper scissors dataset"). The model was trained on grayscale 64×64 crops.

---

## Notes & Known Limitations

- Inference is slow (~1–3 fps) because pure NumPy convolutions are not optimized. This is intentional — the goal was to understand the math, not maximize speed.
- The model is sensitive to lighting and background. For best results, use a plain background with good, even lighting.
- The AI opponent picks randomly — the neural network is only used for gesture recognition, not strategy.
