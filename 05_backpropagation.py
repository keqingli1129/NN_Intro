"""
Section 5 - Backpropagation.

Key insight: a neural network is just a giant nested function.
The inputs are fixed (they're the problem); the knobs we turn are the WEIGHTS.
So we run gradient descent on the weights, using the chain rule to get each weight's blame.
"""

from tiny_net import INITIAL_WEIGHTS, backward, forward, loss, step

x1, x2, y_true = 3, 2, 24               # the "crime scene": the first training example
w = INITIAL_WEIGHTS

h1, h2, y = forward(w, x1, x2)          # forward pass
print(f"Before: prediction = {y:.2f}, target = {y_true}, error = {loss(y, y_true):.2f}")

grads = backward(w, x1, x2, y_true)     # backward pass: blame for every weight
print(f"\ndL/dy = 2*({y:.0f} - {y_true}) = {2 * (y - y_true):.0f}  <- computed once, reused by every weight")
for name, g in zip(["w1", "w2", "w3", "w4", "w5"], grads):
    print(f"  blame for {name}: {g:>6.0f}")
# w5 = 26 * 1                    (short path: error -> prediction -> w5)
# w1 = 26 * w3 * 20 = 520        (long path:  error -> prediction -> h1 -> w1)

# One gradient-descent step. The narration says "0.001" but every number on screen
# (w1 -> 0.948, w4 -> 0.9688) matches 0.0001, so that's what we use.
lr = 0.0001
w_new = step(w, grads, lr)
print("\nWeight updates (new = old - lr * grad):")
for name, old, new in zip(["w1", "w2", "w3", "w4", "w5"], w, w_new):
    print(f"  {name}: {old:.4f} -> {new:.4f}")

# Moment of truth: forward pass again with the updated weights.
h1, h2, y = forward(w_new, x1, x2)
print(f"\nAfter one update: h1 = {h1:.2f}, h2 = {h2:.2f}, prediction = {y:.2f}")
print(f"Error went from {loss(37, y_true):.0f} to {loss(y, y_true):.2f}. The network learned.")

# Keep repeating on this one example and watch it close in on 24.
for i in range(1000):
    w_new = step(w_new, backward(w_new, x1, x2, y_true), lr)
print(f"After 1000 more updates: prediction = {forward(w_new, x1, x2)[2]:.4f} (target {y_true})")
