"""
PyTorch, Part 4 - Linear regression from scratch with raw tensors.

Model: y_hat = x @ w + b. Two knobs (w and b). The data secretly follows y = 2x + 1 + noise;
the model never sees the true values, only x and y, and must discover them.
We do every step by hand, except the gradients - autograd computes those.
"""

import torch

torch.manual_seed(42)                           # reproducible random data and starting weights

# ---------------------------------------------------------------------------
# Setup: fake data that follows a line.
# ---------------------------------------------------------------------------
N = 10                                          # number of data points in our batch
D_in, D_out = 1, 1                              # one input feature, one output value
X = torch.randn(N, D_in)                        # inputs, shape (10, 1)
true_W = torch.tensor([[2.0]])                  # the secret slope  (shape D_in x D_out)
true_b = torch.tensor(1.0)                      # the secret intercept
y_true = X @ true_W + true_b + 0.1 * torch.randn(N, D_out)   # targets = line + a little noise

# The model's brain: random starting values, with the magic switch on.
W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(f"Initial guess: W = {W.item():.4f}, b = {b.item():.4f}   (truth: W = 2, b = 1)\n")

# ---------------------------------------------------------------------------
# Step 1 - forward pass: make a guess.
# ---------------------------------------------------------------------------
y_hat = X @ W + b                               # y = xW + b, written with @ (lesson 16)
print("First guess vs truth (first 3 rows):")
print(torch.cat([y_hat[:3], y_true[:3]], dim=1).detach())   # side by side; detach = just the numbers
print("y_hat.grad_fn:", y_hat.grad_fn, " <- autograd is already recording\n")

# ---------------------------------------------------------------------------
# Step 2 - loss: mean squared error,  L = (1/N) * sum((y_hat - y)^2)
# ---------------------------------------------------------------------------
error = y_hat - y_true                          # difference for every point
squared_error = error ** 2                      # make them all positive, punish big misses
loss = squared_error.mean()                     # average -> one scorecard number
print(f"loss = {loss.item():.4f}")

# ---------------------------------------------------------------------------
# Step 3 - backward pass: who is to blame?
# ---------------------------------------------------------------------------
loss.backward()                                 # fills W.grad and b.grad (dL/dW and dL/db)
print(f"W.grad = {W.grad.item():+.4f}   b.grad = {b.grad.item():+.4f}")
print("Negative gradient = increasing that knob would LOWER the loss, so we'll increase it.")

# Check autograd against the derivative by hand:  dL/dW = (2/N) * sum(error * x),  dL/db = (2/N) * sum(error)
with torch.no_grad():
    print(f"by hand: W.grad = {(2 / N * (error * X).sum()).item():+.4f}   "
          f"b.grad = {(2 / N * error.sum()).item():+.4f}\n")

# ---------------------------------------------------------------------------
# Steps 1-5 in a loop: forward, loss, backward, update, reset.
# ---------------------------------------------------------------------------
learning_rate = 0.1
epochs = 100
W = torch.randn(D_in, D_out, requires_grad=True)   # fresh random start
b = torch.randn(1, requires_grad=True)

for epoch in range(epochs):
    y_hat = X @ W + b                           # 1. forward pass
    loss = ((y_hat - y_true) ** 2).mean()       # 2. loss
    loss.backward()                             # 3. backward: compute W.grad, b.grad

    with torch.no_grad():                       # 4. update - don't record this step in the graph
        W -= learning_rate * W.grad             #    theta_new = theta_old - lr * gradient
        b -= learning_rate * b.grad

    W.grad.zero_()                              # 5. reset: otherwise next epoch's gradients
    b.grad.zero_()                              #    would be ADDED to these old ones

    if epoch % 10 == 0:
        print(f"epoch {epoch:>3}: loss = {loss.item():.4f}, W = {W.item():.4f}, b = {b.item():.4f}")

print(f"\nLearned: W = {W.item():.4f}, b = {b.item():.4f}   (truth: W = 2, b = 1; noise keeps it from being exact)")

# What goes wrong without step 5? Gradients pile up.
w_demo = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    (w_demo * 2).backward()                     # gradient is 2 every time ...
    print(f"no zero_(): call {i + 1}, w_demo.grad = {w_demo.grad.item()}")  # ... but it accumulates 2, 4, 6
