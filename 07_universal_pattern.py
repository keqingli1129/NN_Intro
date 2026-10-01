"""
Section 7 - Scaling up: activation functions, loss functions, and the universal loop.

GPT-4 and our 5-weight net run the same 5 steps:
  1. forward pass  2. loss  3. backprop  4. gradient descent  5. repeat
Bigger models just have longer chains, nicer activations, and task-specific losses.
"""

import math
from tiny_net import DATA, INITIAL_WEIGHTS, forward, loss, backward, step, total_loss

# ---------------------------------------------------------------------------
# Popular activation functions (each one just needs a derivative we can compute).
# ---------------------------------------------------------------------------
def relu(z):       return max(0.0, z)                 # passes positives, zeros out negatives
def relu_grad(z):  return 1.0 if z > 0 else 0.0       # slope is 1 or 0 - never explodes

def sigmoid(z):      return 1 / (1 + math.exp(-z))    # squashes anything into (0, 1) - a probability
def sigmoid_grad(z): return sigmoid(z) * (1 - sigmoid(z))

def square(z):       return z ** 2                    # what our tiny net used
def square_grad(z):  return 2 * z                     # grows without limit -> can blow up

print("Activation functions and their slopes:")
print(f"{'z':>5} | {'relu':>6} {'slope':>6} | {'sigmoid':>7} {'slope':>6} | {'z^2':>6} {'slope':>6}")
for z in [-3, -1, 0, 1, 3, 10]:
    print(f"{z:>5} | {relu(z):>6.2f} {relu_grad(z):>6.2f} | {sigmoid(z):>7.4f} {sigmoid_grad(z):>6.4f} "
          f"| {square(z):>6.0f} {square_grad(z):>6.0f}")

# ---------------------------------------------------------------------------
# Loss functions: different formula, same job - one number to start the blame game.
# ---------------------------------------------------------------------------
def mse(pred, true):
    """Mean squared error - for predicting numbers (e.g. house prices)."""
    return sum((p - t) ** 2 for p, t in zip(pred, true)) / len(pred)

def cross_entropy(p_cat, is_cat):
    """Binary cross-entropy - for classification. p_cat = predicted probability of 'cat'."""
    return -(is_cat * math.log(p_cat) + (1 - is_cat) * math.log(1 - p_cat))

print(f"\nMSE of predictions [37, 33, 13] vs [24, 14, 11] = {mse([37, 33, 13], [24, 14, 11]):.1f}")
print(f"Cross-entropy, image IS a cat, model says 90% cat = {cross_entropy(0.9, 1):.3f} (good, small)")
print(f"Cross-entropy, image IS a cat, model says 10% cat = {cross_entropy(0.1, 1):.3f} (bad, large)")

# ---------------------------------------------------------------------------
# The universal pattern, written out explicitly on our tiny network.
# ---------------------------------------------------------------------------
w = INITIAL_WEIGHTS
lr = 0.0001
print("\nThe universal training loop:")
for epoch in range(5001):                              # 5. repeat
    grads = [0.0] * 5
    for (x1, x2), y_true in DATA:
        _, _, y = forward(w, x1, x2)                    # 1. forward pass: make a guess
        _ = loss(y, y_true)                             # 2. loss: how wrong was it?
        g = backward(w, x1, x2, y_true)                 # 3. backprop: blame for every weight
        grads = [a + b for a, b in zip(grads, g)]
    w = step(w, grads, lr)                              # 4. gradient descent: nudge the weights
    if epoch in (0, 10, 100, 1000, 5000):
        print(f"  epoch {epoch:>5}: total error = {total_loss(w):.4f}")

print("\nFinal predictions:")
for (x1, x2), y_true in DATA:
    print(f"  {(x1, x2)} -> {forward(w, x1, x2)[2]:.2f}   (true {y_true})")
