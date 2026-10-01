"""
PyTorch, Part 2 - Autograd, requires_grad, and the computation graph.

requires_grad=True is the switch that says "this is a learnable parameter - record
every operation done to it". PyTorch then builds a graph of those operations, and
.backward() walks that graph backwards applying the chain rule for you (lessons 03 & 05).
"""

import torch

# ---------------------------------------------------------------------------
# Data vs. parameter.
# ---------------------------------------------------------------------------
x_data = torch.tensor([[1, 2], [3, 4]])                  # plain data: nobody will learn it
w = torch.tensor([1.0, 2.0], requires_grad=True)         # a parameter: track it!
print("x_data.requires_grad:", x_data.requires_grad)     # False (the default)
print("w.requires_grad     :", w.requires_grad, "\n")    # True

# ---------------------------------------------------------------------------
# Building a computation graph:  y = a + b,  z = x * y
# ---------------------------------------------------------------------------
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)

y = a + b                                       # PyTorch records an "add" node: a, b -> y
z = x * y                                       # ... and a "multiply" node: x, y -> z
print(f"y = a + b = {y.item()}")
print(f"z = x * y = {z.item()}\n")

# grad_fn is the breadcrumb pointing at the operation that created each tensor.
print("Breadcrumbs (grad_fn):")
print("  z.grad_fn:", z.grad_fn)                # MulBackward0 - z came from a multiplication
print("  y.grad_fn:", y.grad_fn)                # AddBackward0 - y came from an addition
print("  a.grad_fn:", a.grad_fn)                # None - a was created by us, not by an operation
print("  z.grad_fn.next_functions:", z.grad_fn.next_functions)  # the links back toward x and y
print()

# Walk the graph backwards: chain rule, done automatically.
z.backward()                                    # compute dz/d(every tensor with requires_grad)
print("Gradients after z.backward():")
print(f"  dz/dx = y = {x.grad.item()}")         # z = x*y      -> dz/dx = y = 5
print(f"  dz/da = x * dy/da = {a.grad.item()}") # chain rule:   dz/dy * dy/da = x * 1 = 4
print(f"  dz/db = x * dy/db = {b.grad.item()}\n")

# ---------------------------------------------------------------------------
# Payoff: autograd reproduces our hand-written backprop from lesson 05.
# ---------------------------------------------------------------------------
w1, w2, w3, w4, w5 = (torch.tensor(v, requires_grad=True) for v in [1.0, 2.0, 1.0, 1.0, 0.0])
x1, x2, y_true = 3.0, 2.0, 24.0                 # first training example of tiny_net

h1 = (x1 + w1 * x2) ** 2                        # same forward pass as tiny_net.forward
h2 = w2 * x1 * x2
y_pred = w3 * h1 + w4 * h2 + w5
loss = (y_pred - y_true) ** 2                   # squared error = 169

loss.backward()                                 # one line replaces the whole backward() function
print(f"tiny_net from lesson 05: prediction {y_pred.item():.0f}, loss {loss.item():.0f}")
print("Autograd gradients:", [p.grad.item() for p in (w1, w2, w3, w4, w5)])
print("Lesson 05 by hand  : [520, 156, 650, 312, 26]  <- identical")
